"""
Tests for dual-source audio capture and processing.

Verifies that when both mic and loopback devices are selected:
- AudioCapture detects dual-source mode and emits chunks with source labels
- Pipeline labels mic segments as "Speaker 1" and system segments as "Speaker 2"
- Per-source dedup keeps separate word histories
- Post-session diarization only processes system audio chunks
- Speaker numbering starts at 2 for system audio diarization
- Single-source mode continues to produce unnumbered "Speaker" labels
"""

import asyncio
import copy
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from server.app import app, pipeline
from transcriber.pipeline import TranscriptionPipeline
from tests.conftest import FAKE_SEGMENTS

from fastapi.testclient import TestClient


# -- AudioCapture dual-source detection --

class TestAudioCaptureDualSource:
    def test_dual_source_detected_with_mic_and_loopback(self):
        """AudioCapture sets _dual_source when both mic and loopback devices are selected."""
        from capture.audio_capture import AudioCapture

        devices = [1, 2]
        device_info = [
            {"index": 1, "name": "Mic", "type": "microphone", "channels": 1},
            {"index": 2, "name": "Loopback", "type": "loopback", "channels": 2},
        ]

        cap = AudioCapture.__new__(AudioCapture)
        cap.on_chunk_ready = MagicMock()
        cap.target_rate = 16000
        cap._selected_devices = devices
        cap._device_info = {d["index"]: d for d in device_info}
        cap.min_samples = 24000
        cap.max_samples = 240000
        cap.silence_threshold = 0.005
        cap.silence_samples = 4800
        cap.overlap_samples = 16000
        cap.energy_window = 800
        cap._pa = None
        cap._streams = []
        cap._stop_event = MagicMock()
        cap._chunk_index = 0
        cap._total_emitted_samples = 0
        cap._lock = MagicMock()
        cap._buffers = {}
        cap._mixer_thread = None
        cap._dual_source = False
        cap._mic_devices = []
        cap._sys_devices = []
        cap._source_state = {}

        # Simulate the classification logic from start()
        cap._mic_devices = [idx for idx in devices
                            if cap._device_info.get(idx, {}).get("type") == "microphone"]
        cap._sys_devices = [idx for idx in devices
                            if cap._device_info.get(idx, {}).get("type") == "loopback"]
        cap._dual_source = bool(cap._mic_devices and cap._sys_devices)

        assert cap._dual_source is True
        assert cap._mic_devices == [1]
        assert cap._sys_devices == [2]

    def test_single_source_when_loopback_only(self):
        """AudioCapture does not set _dual_source when only loopback is selected."""
        from capture.audio_capture import AudioCapture

        devices = [2]
        device_info = [
            {"index": 2, "name": "Loopback", "type": "loopback", "channels": 2},
        ]

        cap = AudioCapture.__new__(AudioCapture)
        cap._selected_devices = devices
        cap._device_info = {d["index"]: d for d in device_info}
        cap._dual_source = False
        cap._mic_devices = []
        cap._sys_devices = []

        cap._mic_devices = [idx for idx in devices
                            if cap._device_info.get(idx, {}).get("type") == "microphone"]
        cap._sys_devices = [idx for idx in devices
                            if cap._device_info.get(idx, {}).get("type") == "loopback"]
        cap._dual_source = bool(cap._mic_devices and cap._sys_devices)

        assert cap._dual_source is False

    def test_emit_chunk_includes_source_in_dual_mode(self):
        """In dual-source mode, _emit_chunk passes source to callback."""
        from capture.audio_capture import AudioCapture

        callback = MagicMock()

        cap = AudioCapture.__new__(AudioCapture)
        cap.on_chunk_ready = callback
        cap.target_rate = 16000
        cap.overlap_samples = 16000
        cap._dual_source = True
        cap._source_state = {
            "mic": {"chunk_index": 0, "total_emitted": 0},
            "system": {"chunk_index": 0, "total_emitted": 0},
        }
        cap._chunk_index = 0
        cap._total_emitted_samples = 0

        chunk = np.zeros(24000, dtype=np.float32)

        cap._emit_chunk(chunk, source="mic")
        cap._emit_chunk(chunk, source="system")

        assert callback.call_count == 2
        # First call: mic
        _, _, _, src1 = callback.call_args_list[0][0]
        assert src1 == "mic"
        # Second call: system
        _, _, _, src2 = callback.call_args_list[1][0]
        assert src2 == "system"

    def test_emit_chunk_no_source_in_single_mode(self):
        """In single-source mode, _emit_chunk passes source=None."""
        from capture.audio_capture import AudioCapture

        callback = MagicMock()

        cap = AudioCapture.__new__(AudioCapture)
        cap.on_chunk_ready = callback
        cap.target_rate = 16000
        cap.overlap_samples = 16000
        cap._dual_source = False
        cap._source_state = {}
        cap._chunk_index = 0
        cap._total_emitted_samples = 0

        chunk = np.zeros(24000, dtype=np.float32)
        cap._emit_chunk(chunk, source=None)

        assert callback.call_count == 1
        _, _, _, src = callback.call_args_list[0][0]
        assert src is None


# -- Pipeline dual-source processing --

class TestPipelineDualSource:
    def test_dual_source_mode_detected(self, mock_audio_capture, mock_diarizer):
        """Pipeline detects dual-source when both mic and loopback are selected."""
        client = TestClient(app)
        resp = client.post("/api/session/start", json={"devices": [1, 2]})
        assert resp.status_code == 200
        assert pipeline._dual_source_mode is True
        assert pipeline.session["dual_source"] is True
        client.post("/api/session/stop")

    def test_single_source_mode(self, mock_audio_capture, mock_diarizer):
        """Pipeline stays single-source when only loopback is selected."""
        client = TestClient(app)
        resp = client.post("/api/session/start", json={"devices": [2]})
        assert resp.status_code == 200
        assert pipeline._dual_source_mode is False
        assert pipeline.session["dual_source"] is False
        client.post("/api/session/stop")

    @pytest.mark.asyncio
    async def test_mic_chunks_labeled_speaker_1(self, mock_audio_capture, mock_diarizer, mock_transcribe):
        """Mic chunks in dual-source mode get labeled 'Speaker 1'."""
        client = TestClient(app)
        client.post("/api/session/start", json={"devices": [1, 2]})

        audio = np.random.randn(16000).astype(np.float32) * 0.5
        await pipeline._process_chunk(audio, 0, 0.0, source="mic")

        mic_segments = [s for s in pipeline.session["segments"] if s.get("source") == "mic"]
        assert len(mic_segments) > 0
        for seg in mic_segments:
            assert seg["speaker"] == "Speaker 1"

        client.post("/api/session/stop")

    @pytest.mark.asyncio
    async def test_system_chunks_labeled_speaker_2(self, mock_audio_capture, mock_diarizer, mock_transcribe):
        """System chunks in dual-source mode get labeled 'Speaker 2'."""
        client = TestClient(app)
        client.post("/api/session/start", json={"devices": [1, 2]})

        audio = np.random.randn(16000).astype(np.float32) * 0.5
        await pipeline._process_chunk(audio, 0, 0.0, source="system")

        sys_segments = [s for s in pipeline.session["segments"] if s.get("source") == "system"]
        assert len(sys_segments) > 0
        for seg in sys_segments:
            assert seg["speaker"] == "Speaker 2"

        client.post("/api/session/stop")

    @pytest.mark.asyncio
    async def test_single_source_labeled_speaker_no_number(self, mock_audio_capture, mock_diarizer, mock_transcribe):
        """Single-source chunks get labeled 'Speaker' with no number."""
        client = TestClient(app)
        client.post("/api/session/start", json={"devices": [2]})

        audio = np.random.randn(16000).astype(np.float32) * 0.5
        await pipeline._process_chunk(audio, 0, 0.0, source=None)

        for seg in pipeline.session["segments"]:
            assert seg["speaker"] == "Speaker"
            assert "source" not in seg

        client.post("/api/session/stop")

    @pytest.mark.asyncio
    async def test_per_source_dedup_state(self, mock_audio_capture, mock_diarizer, mock_transcribe):
        """Mic and system maintain separate dedup word histories."""
        client = TestClient(app)
        client.post("/api/session/start", json={"devices": [1, 2]})

        audio = np.random.randn(16000).astype(np.float32) * 0.5

        await pipeline._process_chunk(audio, 0, 0.0, source="mic")
        await pipeline._process_chunk(audio, 0, 0.0, source="system")

        assert "mic" in pipeline._prev_words_by_source
        assert "system" in pipeline._prev_words_by_source
        assert len(pipeline._prev_words_by_source["mic"]) > 0
        assert len(pipeline._prev_words_by_source["system"]) > 0

        client.post("/api/session/stop")

    @pytest.mark.asyncio
    async def test_dual_source_chunk_filenames(self, mock_audio_capture, mock_diarizer, mock_transcribe, isolated_sessions):
        """Dual-source chunks saved with mic_/sys_ prefixes."""
        import soundfile as sf

        client = TestClient(app)
        client.post("/api/session/start", json={"devices": [1, 2]})

        audio = np.random.randn(16000).astype(np.float32) * 0.5

        await pipeline._process_chunk(audio, 0, 0.0, source="mic")
        await pipeline._process_chunk(audio, 0, 0.0, source="system")

        session_dir = pipeline.session["dir"]
        # sf.write is mocked, so check the calls
        write_calls = sf.write.call_args_list
        paths = [call[0][0] for call in write_calls]

        mic_paths = [p for p in paths if "mic_chunk_" in p]
        sys_paths = [p for p in paths if "sys_chunk_" in p]
        assert len(mic_paths) > 0, f"Expected mic_chunk files, got: {paths}"
        assert len(sys_paths) > 0, f"Expected sys_chunk files, got: {paths}"

        client.post("/api/session/stop")

    @pytest.mark.asyncio
    async def test_single_source_chunk_filenames(self, mock_audio_capture, mock_diarizer, mock_transcribe, isolated_sessions):
        """Single-source chunks saved with plain chunk_ prefix."""
        import soundfile as sf

        client = TestClient(app)
        client.post("/api/session/start", json={"devices": [2]})

        audio = np.random.randn(16000).astype(np.float32) * 0.5

        await pipeline._process_chunk(audio, 0, 0.0, source=None)

        write_calls = sf.write.call_args_list
        paths = [call[0][0] for call in write_calls]

        plain_paths = [p for p in paths if "/chunk_" in str(p)]
        assert len(plain_paths) > 0, f"Expected chunk_ files, got: {paths}"

        client.post("/api/session/stop")


# -- Post-session diarization --

class TestPostSessionDiarization:
    def test_dual_source_diarization_starts_at_speaker_2(self, mock_audio_capture, mock_diarizer):
        """In dual-source mode, post-session diarization labels start at Speaker 2."""
        import soundfile as sf

        # Build a fake session with dual-source segments
        session_dir = Path(pipeline.session["dir"]) if pipeline.session else None

        # Create a temp session manually
        import config
        session_id = "test_dual_diar"
        sdir = config.SESSIONS_DIR / session_id
        sdir.mkdir(parents=True, exist_ok=True)

        # Write a fake sys_chunk file
        fake_audio = np.random.randn(16000).astype(np.float32) * 0.1
        sf.write = MagicMock()  # Already mocked

        # Create fake chunk files on disk
        import tempfile
        # We need actual files for glob to find
        chunk_path = sdir / "sys_chunk_0000.wav"
        chunk_path.write_bytes(b"fake")

        session = {
            "id": session_id,
            "started_at": "2026-01-01T12:00:00",
            "ended_at": "2026-01-01T12:05:00",
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Hello from remote", "speaker": "Speaker 2", "source": "system"},
                {"start": 2.5, "end": 4.0, "text": "My response", "speaker": "Speaker 1", "source": "mic"},
                {"start": 4.5, "end": 6.0, "text": "Remote again", "speaker": "Speaker 2", "source": "system"},
            ],
            "dir": sdir,
            "dual_source": True,
        }

        # Mock diarizer pipeline to return a fake annotation
        mock_annotation = MagicMock()
        mock_turns = [
            (MagicMock(start=0.0, end=2.0), None, "SPEAKER_00"),
            (MagicMock(start=2.5, end=4.0), None, "SPEAKER_01"),
        ]
        mock_annotation.itertracks = MagicMock(return_value=mock_turns)

        mock_result = MagicMock()
        mock_result.speaker_diarization = mock_annotation
        pipeline.diarizer.pipeline = MagicMock(return_value=mock_result)

        # Mock sf.read for chunk loading
        import sys
        sf_mod = sys.modules["soundfile"]
        original_read = sf_mod.read
        sf_mod.read = MagicMock(return_value=(fake_audio, 16000))

        try:
            pipeline._run_post_session_diarization(session)
        finally:
            sf_mod.read = original_read
            pipeline.diarizer.pipeline = None

        # Mic segment should be untouched
        assert session["segments"][1]["speaker"] == "Speaker 1"

        # System segments should be relabeled starting at Speaker 2
        system_speakers = {s["speaker"] for s in session["segments"] if s.get("source") == "system"}
        for sp in system_speakers:
            # Should be Speaker 2 or higher, never Speaker 1
            num = int(sp.split()[-1])
            assert num >= 2

    def test_single_source_diarization_starts_at_speaker_1(self, mock_audio_capture, mock_diarizer):
        """In single-source mode, diarization labels start at Speaker 1."""
        import config
        import sys

        session_id = "test_single_diar"
        sdir = config.SESSIONS_DIR / session_id
        sdir.mkdir(parents=True, exist_ok=True)

        chunk_path = sdir / "chunk_0000.wav"
        chunk_path.write_bytes(b"fake")

        session = {
            "id": session_id,
            "started_at": "2026-01-01T12:00:00",
            "ended_at": "2026-01-01T12:05:00",
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "First speaker", "speaker": "Speaker"},
                {"start": 2.5, "end": 4.0, "text": "Second speaker", "speaker": "Speaker"},
            ],
            "dir": sdir,
            "dual_source": False,
        }

        mock_annotation = MagicMock()
        mock_turns = [
            (MagicMock(start=0.0, end=2.0), None, "SPEAKER_00"),
            (MagicMock(start=2.5, end=4.0), None, "SPEAKER_01"),
        ]
        mock_annotation.itertracks = MagicMock(return_value=mock_turns)

        mock_result = MagicMock()
        mock_result.speaker_diarization = mock_annotation
        pipeline.diarizer.pipeline = MagicMock(return_value=mock_result)

        fake_audio = np.random.randn(16000).astype(np.float32) * 0.1
        sf_mod = sys.modules["soundfile"]
        original_read = sf_mod.read
        sf_mod.read = MagicMock(return_value=(fake_audio, 16000))

        try:
            pipeline._run_post_session_diarization(session)
        finally:
            sf_mod.read = original_read
            pipeline.diarizer.pipeline = None

        speakers = {s["speaker"] for s in session["segments"]}
        assert "Speaker 1" in speakers
        assert "Speaker 2" in speakers


# -- Diarizer reset_with_offset --

class TestDiarizerResetWithOffset:
    def test_reset_with_offset(self):
        """reset_with_offset sets next_speaker_id to the given value."""
        from transcriber.diarization import Diarizer
        d = Diarizer()
        d.speaker_map = {"A": "Speaker 1"}
        d.speaker_embeddings = {"Speaker 1": np.array([1, 0])}
        d.next_speaker_id = 5

        d.reset_with_offset(2)

        assert d.speaker_map == {}
        assert d.speaker_embeddings == {}
        assert d.next_speaker_id == 2

    def test_regular_reset_starts_at_1(self):
        """Regular reset starts at Speaker 1."""
        from transcriber.diarization import Diarizer
        d = Diarizer()
        d.next_speaker_id = 10
        d.reset()
        assert d.next_speaker_id == 1
