"""
Real-time transcription pipeline.
Orchestrates audio capture, transcription, diarization, and WebSocket broadcasting.
Runs as a single process — audio capture in a background thread, ML in a thread pool.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

import config
from capture.audio_capture import AudioCapture
from transcriber.transcription import Transcriber
from transcriber.diarization import Diarizer


class TranscriptionPipeline:
    def __init__(self):
        self.transcriber = Transcriber()
        self.diarizer = Diarizer()
        self.capture = None
        self.session = None
        self.websocket_clients = set()
        self._loop = None
        self._processing_lock = None

    def load_models(self):
        self.transcriber.load_model()
        self.diarizer.load_model()

    def start_session(self) -> dict:
        now = datetime.now()
        session_id = now.strftime("%Y%m%d_%H%M%S")
        session_dir = config.SESSIONS_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        self.session = {
            "id": session_id,
            "started_at": now.isoformat(),
            "ended_at": None,
            "segments": [],
            "dir": session_dir,
        }

        self._loop = asyncio.get_event_loop()
        self._processing_lock = asyncio.Lock()

        # Start audio capture thread
        self.capture = AudioCapture(on_chunk_ready=self._on_chunk_from_thread)
        self.capture.start()

        print(f"Session started: {session_id}")
        return {"id": session_id, "started_at": self.session["started_at"]}

    def stop_session(self) -> dict | None:
        if not self.session:
            return None

        # Stop audio capture
        if self.capture:
            self.capture.stop()
            self.capture = None

        self.session["ended_at"] = datetime.now().isoformat()

        # Save final transcript
        self._save_transcript()

        result = {
            "id": self.session["id"],
            "started_at": self.session["started_at"],
            "ended_at": self.session["ended_at"],
            "segment_count": len(self.session["segments"]),
        }

        print(f"Session stopped: {self.session['id']} ({len(self.session['segments'])} segments)")
        self.session = None
        return result

    def _on_chunk_from_thread(self, audio: np.ndarray, chunk_index: int, offset: float):
        """Called from the capture thread — schedules async processing on the event loop."""
        if self._loop and self.session:
            asyncio.run_coroutine_threadsafe(
                self._process_chunk(audio, chunk_index, offset),
                self._loop,
            )

    async def _process_chunk(self, audio: np.ndarray, chunk_index: int, offset: float):
        if not self.session:
            return

        async with self._processing_lock:
            # Save raw audio chunk
            chunk_path = self.session["dir"] / f"chunk_{chunk_index:04d}.wav"
            sf.write(str(chunk_path), audio, config.AUDIO_SAMPLE_RATE)

            # Check if chunk has actual audio (not silence)
            if np.max(np.abs(audio)) < 0.001:
                return

            loop = asyncio.get_event_loop()

            # Transcribe (runs in thread pool — CPU-bound CUDA call)
            segments = await loop.run_in_executor(
                None, self.transcriber.transcribe, audio, offset
            )

            if not segments:
                return

            # Diarize — pass chunk-relative times to pyannote
            chunk_segments = []
            for seg in segments:
                chunk_segments.append({
                    **seg,
                    "start": seg["start"] - offset,
                    "end": seg["end"] - offset,
                })

            diarized = await loop.run_in_executor(
                None, self.diarizer.diarize, audio, chunk_segments
            )

            # Restore absolute timestamps
            for seg in diarized:
                seg["start"] = round(seg["start"] + offset, 1)
                seg["end"] = round(seg["end"] + offset, 1)
                self.session["segments"].append(seg)
                await self._broadcast(seg)

    async def _broadcast(self, segment: dict):
        message = json.dumps({
            "type": "segment",
            "data": {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker": segment.get("speaker", "Speaker"),
            },
        })

        dead = set()
        for ws in self.websocket_clients:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        self.websocket_clients -= dead

    def _save_transcript(self):
        if not self.session:
            return
        transcript_path = self.session["dir"] / "transcript.json"
        save_data = {
            "id": self.session["id"],
            "started_at": self.session["started_at"],
            "ended_at": self.session["ended_at"],
            "segments": self.session["segments"],
        }
        transcript_path.write_text(json.dumps(save_data, indent=2))

    def get_sessions(self) -> list[dict]:
        sessions = []
        if not config.SESSIONS_DIR.exists():
            return sessions

        for session_dir in sorted(config.SESSIONS_DIR.iterdir(), reverse=True):
            transcript = session_dir / "transcript.json"
            if transcript.exists():
                data = json.loads(transcript.read_text())
                sessions.append({
                    "id": data["id"],
                    "started_at": data["started_at"],
                    "ended_at": data["ended_at"],
                    "segment_count": len(data.get("segments", [])),
                })
        return sessions

    def get_session_transcript(self, session_id: str) -> dict | None:
        transcript = config.SESSIONS_DIR / session_id / "transcript.json"
        if not transcript.exists():
            return None
        return json.loads(transcript.read_text())
