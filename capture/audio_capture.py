"""
Audio capture using PyAudioWPatch (WASAPI).
Captures system audio loopback and microphone, mixes into mono.
Uses VAD-based chunking: splits on natural speech pauses instead of fixed intervals.

Device discovery happens once at init (slow probe).
Session start/stop just opens/closes streams (instant).
"""

import threading
import numpy as np
import pyaudiowpatch as pyaudio

import config

NATIVE_RATE = 48000
FRAME_SIZE = 4800  # 100ms at 48kHz


def discover_devices():
    """
    Probe all WASAPI devices and select the best loopback + mic.
    Called once at server startup. Returns (loopback_idx, loopback_channels, mic_idx).
    """
    p = pyaudio.PyAudio()
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)

    lb_idx = None
    lb_ch = None
    mic_idx = None

    # Find first loopback device
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev["hostApi"] == wasapi["index"] and dev.get("isLoopbackDevice", False):
            lb_idx = i
            lb_ch = dev["maxInputChannels"]
            print(f"  Loopback: {dev['name']} ({lb_ch}ch)")
            break

    # Find best microphone
    if config.AUDIO_MIC_DEVICE:
        # User specified a device name
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if (dev["hostApi"] == wasapi["index"]
                    and not dev.get("isLoopbackDevice", False)
                    and dev["maxInputChannels"] > 0
                    and config.AUDIO_MIC_DEVICE.lower() in dev["name"].lower()):
                mic_idx = i
                print(f"  Microphone (configured): {dev['name']}")
                break
        if mic_idx is None:
            print(f"  WARNING: Configured mic '{config.AUDIO_MIC_DEVICE}' not found")

    if mic_idx is None:
        # Auto-detect: probe each mic for 0.5s, pick loudest
        candidates = []
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if (dev["hostApi"] == wasapi["index"]
                    and not dev.get("isLoopbackDevice", False)
                    and dev["maxInputChannels"] > 0):
                candidates.append((i, dev["name"]))

        if candidates:
            print("  Auto-detecting microphone...")
            best_peak = 0
            for idx, name in candidates:
                try:
                    stream = p.open(
                        format=pyaudio.paFloat32, channels=1, rate=NATIVE_RATE,
                        input=True, input_device_index=idx, frames_per_buffer=FRAME_SIZE,
                    )
                    frames = []
                    for _ in range(5):  # 0.5 second
                        data = stream.read(FRAME_SIZE, exception_on_overflow=False)
                        frames.append(np.frombuffer(data, dtype=np.float32))
                    stream.stop_stream()
                    stream.close()

                    audio = np.concatenate(frames)
                    peak = float(np.max(np.abs(audio)))
                    status = "active" if peak > 0.005 else "silent"
                    print(f"    [{idx}] {name}: peak={peak:.4f} ({status})")

                    if peak > best_peak:
                        best_peak = peak
                        mic_idx = idx
                except Exception as e:
                    print(f"    [{idx}] {name}: error - {e}")

            if mic_idx is not None:
                dev = p.get_device_info_by_index(mic_idx)
                print(f"  Selected mic: {dev['name']}")
            if best_peak < 0.005:
                print("  WARNING: All mics appear silent — set AUDIO_MIC_DEVICE in .env")

    p.terminate()
    return lb_idx, lb_ch, mic_idx


class AudioCapture:
    def __init__(self, on_chunk_ready, lb_idx, lb_ch, mic_idx):
        self.on_chunk_ready = on_chunk_ready
        self.target_rate = config.AUDIO_SAMPLE_RATE
        self._lb_device_idx = lb_idx
        self._lb_channels = lb_ch
        self._mic_device_idx = mic_idx

        # VAD chunking parameters
        self.min_samples = int(self.target_rate * config.AUDIO_MIN_CHUNK_SECONDS)
        self.max_samples = int(self.target_rate * config.AUDIO_MAX_CHUNK_SECONDS)
        self.silence_threshold = config.AUDIO_SILENCE_THRESHOLD
        self.silence_samples = int(self.target_rate * config.AUDIO_SILENCE_DURATION_MS / 1000)
        self.overlap_samples = int(self.target_rate * config.AUDIO_OVERLAP_SECONDS)

        # Window for RMS energy calculation (50ms)
        self.energy_window = int(self.target_rate * 0.05)

        self._pa = None
        self._lb_stream = None
        self._mic_stream = None
        self._stop_event = threading.Event()
        self._chunk_index = 0
        self._total_emitted_samples = 0

        self._lock = threading.Lock()
        self._lb_buf = np.array([], dtype=np.float32)
        self._mic_buf = np.array([], dtype=np.float32)
        self._has_mic = False

        self._mixer_thread = None

    def start(self):
        """Open audio streams and start capture. Instant — no device probing."""
        self._pa = pyaudio.PyAudio()
        self._stop_event.clear()
        self._chunk_index = 0
        self._total_emitted_samples = 0
        self._lb_buf = np.array([], dtype=np.float32)
        self._mic_buf = np.array([], dtype=np.float32)

        # Open loopback
        self._lb_stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self._lb_channels,
            rate=NATIVE_RATE,
            input=True,
            input_device_index=self._lb_device_idx,
            frames_per_buffer=FRAME_SIZE,
            stream_callback=self._make_lb_callback(self._lb_channels),
        )

        # Open microphone
        if self._mic_device_idx is not None:
            try:
                self._mic_stream = self._pa.open(
                    format=pyaudio.paFloat32,
                    channels=1,
                    rate=NATIVE_RATE,
                    input=True,
                    input_device_index=self._mic_device_idx,
                    frames_per_buffer=FRAME_SIZE,
                    stream_callback=self._mic_callback,
                )
                self._has_mic = True
            except Exception as e:
                print(f"Could not open mic: {e}")

        self._mixer_thread = threading.Thread(target=self._mix_loop, daemon=True)
        self._mixer_thread.start()
        print("Audio capture started")

    def stop(self):
        self._stop_event.set()

        if self._lb_stream:
            self._lb_stream.stop_stream()
            self._lb_stream.close()
            self._lb_stream = None

        if self._mic_stream:
            self._mic_stream.stop_stream()
            self._mic_stream.close()
            self._mic_stream = None

        if self._mixer_thread:
            self._mixer_thread.join(timeout=3)
            self._mixer_thread = None

        if self._pa:
            self._pa.terminate()
            self._pa = None

        print("Audio capture stopped")

    @property
    def is_running(self):
        return self._lb_stream is not None and self._lb_stream.is_active()

    def _make_lb_callback(self, channels):
        def callback(in_data, frame_count, time_info, status):
            raw = np.frombuffer(in_data, dtype=np.float32).reshape(-1, channels)
            mono = np.mean(raw[:, :2], axis=1)
            mono16k = mono[::3]  # 48kHz -> 16kHz
            with self._lock:
                self._lb_buf = np.concatenate([self._lb_buf, mono16k])
            return (None, pyaudio.paContinue)
        return callback

    def _mic_callback(self, in_data, frame_count, time_info, status):
        mono = np.frombuffer(in_data, dtype=np.float32)
        mono16k = mono[::3]
        with self._lock:
            self._mic_buf = np.concatenate([self._mic_buf, mono16k])
        return (None, pyaudio.paContinue)

    def _find_silence_boundary(self, audio):
        if len(audio) < self.min_samples:
            return -1

        n_windows = len(audio) // self.energy_window
        if n_windows == 0:
            return -1

        energies = np.array([
            np.sqrt(np.mean(audio[i * self.energy_window:(i + 1) * self.energy_window] ** 2))
            for i in range(n_windows)
        ])

        is_silent = energies < self.silence_threshold
        silence_windows_needed = max(1, self.silence_samples // self.energy_window)

        min_window = self.min_samples // self.energy_window
        best_split = -1

        for i in range(n_windows - 1, min_window - 1, -1):
            start = max(0, i - silence_windows_needed + 1)
            if np.all(is_silent[start:i + 1]):
                split_window = (start + i) // 2
                best_split = split_window * self.energy_window
                break

        return best_split

    def _emit_chunk(self, chunk):
        offset = self._total_emitted_samples / self.target_rate
        self.on_chunk_ready(chunk, self._chunk_index, offset)
        self._total_emitted_samples += len(chunk) - self.overlap_samples
        self._chunk_index += 1

    def _mix_loop(self):
        mixed_buffer = np.array([], dtype=np.float32)

        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=0.1)

            with self._lock:
                n_lb = len(self._lb_buf)
                n_mic = len(self._mic_buf)

                if self._has_mic and n_lb > 0 and n_mic > 0:
                    n = min(n_lb, n_mic)
                    lb = self._lb_buf[:n].copy()
                    mc = self._mic_buf[:n].copy()
                    self._lb_buf = self._lb_buf[n:]
                    self._mic_buf = self._mic_buf[n:]
                    new_audio = (lb + mc) * 0.5
                elif n_lb > 0 and not self._has_mic:
                    new_audio = self._lb_buf.copy()
                    self._lb_buf = np.array([], dtype=np.float32)
                else:
                    continue

            mixed_buffer = np.concatenate([mixed_buffer, new_audio])

            if len(mixed_buffer) >= self.min_samples:
                split = self._find_silence_boundary(mixed_buffer)

                if split > 0:
                    chunk = mixed_buffer[:split].copy()
                    keep_from = max(0, split - self.overlap_samples)
                    mixed_buffer = mixed_buffer[keep_from:]
                    self._emit_chunk(chunk)

                elif len(mixed_buffer) >= self.max_samples:
                    split = self._find_best_split_near_end(mixed_buffer)
                    chunk = mixed_buffer[:split].copy()
                    keep_from = max(0, split - self.overlap_samples)
                    mixed_buffer = mixed_buffer[keep_from:]
                    self._emit_chunk(chunk)

        if len(mixed_buffer) > self.target_rate:
            self._emit_chunk(mixed_buffer)

        print("Audio capture stopped")

    def _find_best_split_near_end(self, audio):
        search_start = max(self.min_samples, len(audio) - self.target_rate * 3)
        n_windows = (len(audio) - search_start) // self.energy_window

        if n_windows <= 0:
            return self.max_samples

        energies = np.array([
            np.sqrt(np.mean(
                audio[search_start + i * self.energy_window:
                      search_start + (i + 1) * self.energy_window] ** 2
            ))
            for i in range(n_windows)
        ])

        best_window = np.argmin(energies)
        return search_start + best_window * self.energy_window
