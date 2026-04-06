"""
File transcription endpoint tests.

Tests the upload → poll → result workflow and the local-path variant.
"""

import io
import json
import time
from unittest.mock import MagicMock, patch

import numpy as np

from tests.conftest import pipeline, file_jobs, FAKE_SEGMENTS


def _mock_file_models(monkeypatch):
    """Set up mocks for the lazy-loaded file transcription models."""
    from server import app as app_module
    import copy

    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = copy.deepcopy(FAKE_SEGMENTS)
    mock_transcriber.load_model.return_value = None

    mock_diarizer = MagicMock()
    mock_diarizer.pipeline = None  # diarization disabled
    mock_diarizer.load_model.return_value = None

    monkeypatch.setattr(app_module, "_file_transcriber", mock_transcriber)
    monkeypatch.setattr(app_module, "_file_diarizer", mock_diarizer)

    # Mock load_audio to return fake audio
    monkeypatch.setattr(
        app_module,
        "load_audio",
        lambda path: np.zeros(16000 * 10, dtype=np.float32),
    )

    return mock_transcriber, mock_diarizer


def test_upload_starts_job(client, monkeypatch):
    _mock_file_models(monkeypatch)

    file_data = io.BytesIO(b"fake audio content")
    resp = client.post(
        "/api/transcribe-file",
        files={"file": ("test.wav", file_data, "audio/wav")},
        data={"language": "en"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "processing"
    assert data["filename"] == "test.wav"


def test_upload_poll_and_result(client, monkeypatch):
    _mock_file_models(monkeypatch)

    file_data = io.BytesIO(b"fake audio")
    resp = client.post(
        "/api/transcribe-file",
        files={"file": ("episode.mp3", file_data, "audio/mpeg")},
        data={"language": "en"},
    )
    job_id = resp.json()["job_id"]

    # Poll until complete (background thread should finish quickly with mocks)
    for _ in range(20):
        time.sleep(0.1)
        resp = client.get(f"/api/transcribe-file/{job_id}/status")
        if resp.json()["status"] == "completed":
            break
    else:
        raise AssertionError("Job did not complete within 2 seconds")

    status = resp.json()
    assert status["status"] == "completed"

    # Get result as JSON
    resp = client.get(f"/api/transcribe-file/{job_id}/result")
    assert resp.status_code == 200
    result = resp.json()
    assert "segments" in result
    assert len(result["segments"]) >= 1
    # Both fake segments have the same speaker so merge_adjacent_segments
    # combines them into one
    assert "Hello, this is a test." in result["segments"][0]["text"]


def test_result_txt_format(client, monkeypatch):
    _mock_file_models(monkeypatch)

    file_data = io.BytesIO(b"fake audio")
    resp = client.post(
        "/api/transcribe-file",
        files={"file": ("test.wav", file_data, "audio/wav")},
    )
    job_id = resp.json()["job_id"]

    for _ in range(20):
        time.sleep(0.1)
        if client.get(f"/api/transcribe-file/{job_id}/status").json()["status"] == "completed":
            break

    resp = client.get(f"/api/transcribe-file/{job_id}/result?format=txt")
    assert resp.status_code == 200
    assert "Hello, this is a test." in resp.text
    assert "[00:" in resp.text


def test_result_srt_format(client, monkeypatch):
    _mock_file_models(monkeypatch)

    file_data = io.BytesIO(b"fake audio")
    resp = client.post(
        "/api/transcribe-file",
        files={"file": ("test.wav", file_data, "audio/wav")},
    )
    job_id = resp.json()["job_id"]

    for _ in range(20):
        time.sleep(0.1)
        if client.get(f"/api/transcribe-file/{job_id}/status").json()["status"] == "completed":
            break

    resp = client.get(f"/api/transcribe-file/{job_id}/result?format=srt")
    assert resp.status_code == 200
    assert "-->" in resp.text
    assert "Hello, this is a test." in resp.text


def test_result_before_completion(client, monkeypatch):
    """Requesting result for an in-progress job returns 400."""
    _mock_file_models(monkeypatch)

    # Manually create a processing job
    file_jobs["test-incomplete"] = {
        "job_id": "test-incomplete",
        "status": "processing",
        "progress": "Transcribing...",
        "filename": "test.wav",
        "duration": None,
        "speakers": {},
        "result": None,
    }

    resp = client.get("/api/transcribe-file/test-incomplete/result")
    assert resp.status_code == 400


def test_save_to_disk(client, monkeypatch, tmp_path):
    """POST /api/transcribe-file/{job_id}/save writes JSON + TXT + SRT."""
    _mock_file_models(monkeypatch)

    file_data = io.BytesIO(b"fake audio")
    resp = client.post(
        "/api/transcribe-file",
        files={"file": ("test.wav", file_data, "audio/wav")},
    )
    job_id = resp.json()["job_id"]

    for _ in range(20):
        time.sleep(0.1)
        if client.get(f"/api/transcribe-file/{job_id}/status").json()["status"] == "completed":
            break

    output_dir = str(tmp_path / "output")
    resp = client.post(
        f"/api/transcribe-file/{job_id}/save",
        json={"output_dir": output_dir, "basename": "my_meeting"},
    )
    assert resp.status_code == 200
    saved = resp.json()["saved"]
    assert len(saved) == 3
    assert any("my_meeting_transcript.json" in p for p in saved)
    assert any("my_meeting_transcript.txt" in p for p in saved)
    assert any("my_meeting_transcript.srt" in p for p in saved)
