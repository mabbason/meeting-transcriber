"""
In-process audio capture using WASAPI loopback via soundcard.
Captures system audio output in a background thread and fills a chunk buffer.
"""

import threading
import numpy as np
import soundcard as sc

import config


class AudioCapture:
    def __init__(self, on_chunk_ready):
        """
        on_chunk_ready: callable(audio_np_array, chunk_index, session_offset_seconds)
            Called from the capture thread when a chunk is ready.
        """
        self.on_chunk_ready = on_chunk_ready
        self.sample_rate = config.AUDIO_SAMPLE_RATE
        self.chunk_samples = int(self.sample_rate * config.AUDIO_CHUNK_SECONDS)
        self.overlap_samples = int(self.sample_rate * config.AUDIO_OVERLAP_SECONDS)
        self.step_samples = self.chunk_samples - self.overlap_samples

        self._thread = None
        self._stop_event = threading.Event()
        self._chunk_index = 0
        self._total_samples = 0

    def start(self):
        self._stop_event.clear()
        self._chunk_index = 0
        self._total_samples = 0
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def list_devices(self):
        speakers = sc.all_speakers()
        return [{"name": s.name, "id": str(s.name)} for s in speakers]

    def _capture_loop(self):
        try:
            default_speaker = sc.default_speaker()
            print(f"Capturing loopback from: {default_speaker.name}")

            loopback = sc.get_microphone(
                id=str(default_speaker.name), include_loopback=True
            )

            buffer = np.array([], dtype=np.float32)
            read_size = int(self.sample_rate * 0.1)  # Read 100ms at a time for responsiveness

            with loopback.recorder(samplerate=self.sample_rate, channels=1) as recorder:
                while not self._stop_event.is_set():
                    audio = recorder.record(numframes=read_size)
                    mono = audio[:, 0] if audio.ndim > 1 else audio
                    buffer = np.concatenate([buffer, mono.astype(np.float32)])

                    while len(buffer) >= self.chunk_samples:
                        chunk = buffer[:self.chunk_samples].copy()
                        # Step forward by (chunk - overlap) so next chunk overlaps
                        buffer = buffer[self.step_samples:]

                        offset = self._chunk_index * config.AUDIO_CHUNK_SECONDS
                        # Adjust offset for overlap after first chunk
                        if self._chunk_index > 0:
                            offset = (self._chunk_index * self.step_samples) / self.sample_rate

                        self.on_chunk_ready(chunk, self._chunk_index, offset)
                        self._chunk_index += 1

        except Exception as e:
            print(f"Audio capture error: {e}")
            import traceback
            traceback.print_exc()

        print("Audio capture stopped")
