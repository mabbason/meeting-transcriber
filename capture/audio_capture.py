"""
Audio capture using PyAudioWPatch (WASAPI).
Captures system audio loopback and microphone, mixes into mono for transcription.
Uses callback-based streams for reliable concurrent capture.
"""

import threading
import numpy as np
import pyaudiowpatch as pyaudio

import config

NATIVE_RATE = 48000
FRAME_SIZE = 4800  # 100ms at 48kHz


class AudioCapture:
    def __init__(self, on_chunk_ready):
        self.on_chunk_ready = on_chunk_ready
        self.target_rate = config.AUDIO_SAMPLE_RATE
        self.chunk_samples = int(self.target_rate * config.AUDIO_CHUNK_SECONDS)
        self.overlap_samples = int(self.target_rate * config.AUDIO_OVERLAP_SECONDS)
        self.step_samples = self.chunk_samples - self.overlap_samples

        self._pa = None
        self._lb_stream = None
        self._mic_stream = None
        self._stop_event = threading.Event()
        self._chunk_index = 0

        self._lock = threading.Lock()
        self._lb_buf = np.array([], dtype=np.float32)
        self._mic_buf = np.array([], dtype=np.float32)
        self._has_mic = False

        self._mixer_thread = None

    def start(self):
        self._pa = pyaudio.PyAudio()
        self._stop_event.clear()
        self._chunk_index = 0

        wasapi = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)

        # Find and open loopback stream
        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            if dev["hostApi"] == wasapi["index"] and dev.get("isLoopbackDevice", False):
                lb_ch = dev["maxInputChannels"]
                print(f"Loopback: {dev['name']} ({lb_ch}ch)")
                self._lb_stream = self._pa.open(
                    format=pyaudio.paFloat32,
                    channels=lb_ch,
                    rate=NATIVE_RATE,
                    input=True,
                    input_device_index=i,
                    frames_per_buffer=FRAME_SIZE,
                    stream_callback=self._make_lb_callback(lb_ch),
                )
                break

        if not self._lb_stream:
            raise RuntimeError("No WASAPI loopback device found")

        # Find and open microphone stream
        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            if (dev["hostApi"] == wasapi["index"]
                    and not dev.get("isLoopbackDevice", False)
                    and dev["maxInputChannels"] > 0):
                print(f"Microphone: {dev['name']}")
                try:
                    self._mic_stream = self._pa.open(
                        format=pyaudio.paFloat32,
                        channels=1,
                        rate=NATIVE_RATE,
                        input=True,
                        input_device_index=i,
                        frames_per_buffer=FRAME_SIZE,
                        stream_callback=self._mic_callback,
                    )
                    self._has_mic = True
                except Exception as e:
                    print(f"Could not open mic: {e}")
                break

        # Start the mixer thread
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
                    mixed = (lb + mc) * 0.5
                elif n_lb > 0 and not self._has_mic:
                    mixed = self._lb_buf.copy()
                    self._lb_buf = np.array([], dtype=np.float32)
                else:
                    continue

            mixed_buffer = np.concatenate([mixed_buffer, mixed])

            while len(mixed_buffer) >= self.chunk_samples:
                chunk = mixed_buffer[:self.chunk_samples].copy()
                mixed_buffer = mixed_buffer[self.step_samples:]

                offset = (self._chunk_index * self.step_samples / self.target_rate
                          if self._chunk_index > 0 else 0.0)

                self.on_chunk_ready(chunk, self._chunk_index, offset)
                self._chunk_index += 1
