"""
Audio capture using PyAudioWPatch (WASAPI).
Captures BOTH system audio loopback (remote participants) AND microphone (local user),
mixes them into a single mono stream for transcription.
"""

import threading
import numpy as np
import pyaudiowpatch as pyaudio

import config

NATIVE_RATE = 48000  # WASAPI native sample rate


class AudioCapture:
    def __init__(self, on_chunk_ready):
        """
        on_chunk_ready: callable(audio_np_array, chunk_index, session_offset_seconds)
            Called from the capture thread when a chunk is ready.
        """
        self.on_chunk_ready = on_chunk_ready
        self.target_rate = config.AUDIO_SAMPLE_RATE
        self.chunk_samples = int(self.target_rate * config.AUDIO_CHUNK_SECONDS)
        self.overlap_samples = int(self.target_rate * config.AUDIO_OVERLAP_SECONDS)
        self.step_samples = self.chunk_samples - self.overlap_samples

        self._pa = None
        self._stop_event = threading.Event()
        self._chunk_index = 0

        self._lock = threading.Lock()
        self._loopback_buffer = np.array([], dtype=np.float32)
        self._mic_buffer = np.array([], dtype=np.float32)

        self._loopback_thread = None
        self._mic_thread = None
        self._mix_thread = None

        # Device info (discovered at start)
        self._loopback_idx = None
        self._loopback_channels = None
        self._mic_idx = None

    def start(self):
        self._pa = pyaudio.PyAudio()
        self._discover_devices()

        self._stop_event.clear()
        self._chunk_index = 0
        self._loopback_buffer = np.array([], dtype=np.float32)
        self._mic_buffer = np.array([], dtype=np.float32)

        self._loopback_thread = threading.Thread(target=self._loopback_loop, daemon=True)
        self._mic_thread = threading.Thread(target=self._mic_loop, daemon=True)
        self._mix_thread = threading.Thread(target=self._mix_loop, daemon=True)

        self._loopback_thread.start()
        self._mic_thread.start()
        self._mix_thread.start()

    def stop(self):
        self._stop_event.set()
        for t in [self._loopback_thread, self._mic_thread, self._mix_thread]:
            if t:
                t.join(timeout=5)
        self._loopback_thread = None
        self._mic_thread = None
        self._mix_thread = None

        if self._pa:
            self._pa.terminate()
            self._pa = None

    @property
    def is_running(self):
        return self._mix_thread is not None and self._mix_thread.is_alive()

    def _discover_devices(self):
        wasapi_info = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)

        # Find default speaker's loopback
        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            if dev["hostApi"] != wasapi_info["index"]:
                continue
            if dev.get("isLoopbackDevice", False):
                self._loopback_idx = i
                self._loopback_channels = dev["maxInputChannels"]
                print(f"Loopback device: {dev['name']} ({self._loopback_channels}ch)")
                break

        # Find default microphone
        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            if dev["hostApi"] != wasapi_info["index"]:
                continue
            if not dev.get("isLoopbackDevice", False) and dev["maxInputChannels"] > 0:
                self._mic_idx = i
                print(f"Microphone device: {dev['name']}")
                break

        if self._loopback_idx is None:
            raise RuntimeError("No WASAPI loopback device found")
        if self._mic_idx is None:
            print("WARNING: No microphone found — only capturing system audio")

    def _resample(self, audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        if from_rate == to_rate:
            return audio
        ratio = to_rate / from_rate
        new_len = int(len(audio) * ratio)
        indices = np.arange(new_len) / ratio
        indices = np.clip(indices, 0, len(audio) - 1).astype(int)
        return audio[indices]

    def _loopback_loop(self):
        try:
            frame_size = int(NATIVE_RATE * 0.1)  # 100ms reads
            stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=self._loopback_channels,
                rate=NATIVE_RATE,
                input=True,
                input_device_index=self._loopback_idx,
                frames_per_buffer=frame_size,
            )

            while not self._stop_event.is_set():
                data = stream.read(frame_size, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.float32)
                # Mix down to mono (average first 2 channels)
                audio = audio.reshape(-1, self._loopback_channels)
                mono = np.mean(audio[:, :2], axis=1)
                # Resample 48kHz -> 16kHz
                mono = self._resample(mono, NATIVE_RATE, self.target_rate)

                with self._lock:
                    self._loopback_buffer = np.concatenate([self._loopback_buffer, mono])

            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"Loopback capture error: {e}")
            import traceback
            traceback.print_exc()

    def _mic_loop(self):
        if self._mic_idx is None:
            return

        try:
            frame_size = int(NATIVE_RATE * 0.1)
            stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=NATIVE_RATE,
                input=True,
                input_device_index=self._mic_idx,
                frames_per_buffer=frame_size,
            )

            while not self._stop_event.is_set():
                data = stream.read(frame_size, exception_on_overflow=False)
                mono = np.frombuffer(data, dtype=np.float32)
                mono = self._resample(mono, NATIVE_RATE, self.target_rate)

                with self._lock:
                    self._mic_buffer = np.concatenate([self._mic_buffer, mono])

            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"Microphone capture error: {e}")
            import traceback
            traceback.print_exc()

    def _mix_loop(self):
        """Mix loopback + mic buffers into chunks and emit them."""
        mixed_buffer = np.array([], dtype=np.float32)

        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=0.05)

            with self._lock:
                n_loopback = len(self._loopback_buffer)
                n_mic = len(self._mic_buffer)

                if self._mic_idx is not None:
                    # Both sources: mix the shorter of the two
                    n = min(n_loopback, n_mic)
                    if n == 0:
                        continue
                    loopback_chunk = self._loopback_buffer[:n]
                    mic_chunk = self._mic_buffer[:n]
                    self._loopback_buffer = self._loopback_buffer[n:]
                    self._mic_buffer = self._mic_buffer[n:]
                else:
                    # Loopback only
                    if n_loopback == 0:
                        continue
                    loopback_chunk = self._loopback_buffer.copy()
                    self._loopback_buffer = np.array([], dtype=np.float32)
                    mic_chunk = None

            if mic_chunk is not None:
                mixed = (loopback_chunk + mic_chunk) * 0.5
            else:
                mixed = loopback_chunk

            mixed_buffer = np.concatenate([mixed_buffer, mixed])

            while len(mixed_buffer) >= self.chunk_samples:
                chunk = mixed_buffer[: self.chunk_samples].copy()
                mixed_buffer = mixed_buffer[self.step_samples :]

                if self._chunk_index > 0:
                    offset = (self._chunk_index * self.step_samples) / self.target_rate
                else:
                    offset = 0.0

                self.on_chunk_ready(chunk, self._chunk_index, offset)
                self._chunk_index += 1

        print("Audio capture stopped")
