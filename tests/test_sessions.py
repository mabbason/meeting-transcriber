"""
Session lifecycle tests.

Covers start → status → stop → list → rename → export → delete.
Critically tests that stop_session() returns fast (no blocking diarization).
"""

import json
import time
from unittest.mock import MagicMock, patch

import numpy as np


def test_start_and_stop_session(client):
    """Basic session lifecycle: start, verify active, stop, verify idle."""
    # Start
    resp = client.post("/api/session/start", json={"devices": [1, 2]})
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    session_id = data["id"]

    # Status should be active
    resp = client.get("/api/session/status")
    assert resp.json()["active"] is True
    assert resp.json()["id"] == session_id

    # Stop
    resp = client.post("/api/session/stop")
    assert resp.status_code == 200
    stop_data = resp.json()
    assert stop_data["id"] == session_id
    assert "ended_at" in stop_data

    # Status should be idle
    resp = client.get("/api/session/status")
    assert resp.json()["active"] is False


def test_stop_returns_quickly(client, monkeypatch):
    """
    CRITICAL: Catches the blocking-diarization regression.
    stop_session() must return in under 500ms even with diarization enabled.
    """
    from tests.conftest import pipeline, FAKE_SEGMENTS
    import copy

    # Enable diarization with a slow mock that would block if called synchronously
    def slow_diarization(session):
        time.sleep(5)  # Simulate slow diarization

    monkeypatch.setattr(pipeline.diarizer, "pipeline", MagicMock())

    # Start session
    resp = client.post("/api/session/start", json={"devices": [1]})
    session_id = resp.json()["id"]

    # Inject some segments so diarization would run
    pipeline.session["segments"] = copy.deepcopy(FAKE_SEGMENTS)

    # Patch the background diarization to be slow
    monkeypatch.setattr(pipeline, "_run_post_session_diarization", slow_diarization)

    # Stop and measure time
    t0 = time.time()
    resp = client.post("/api/session/stop")
    elapsed = time.time() - t0

    assert resp.status_code == 200
    assert elapsed < 0.5, f"stop_session() took {elapsed:.2f}s — diarization is blocking the response"


def test_stop_saves_transcript_immediately(client, isolated_sessions, monkeypatch):
    """Transcript should be saved before diarization runs."""
    from tests.conftest import pipeline, FAKE_SEGMENTS
    import copy

    resp = client.post("/api/session/start", json={"devices": [1]})
    session_id = resp.json()["id"]

    pipeline.session["segments"] = copy.deepcopy(FAKE_SEGMENTS)

    resp = client.post("/api/session/stop")
    assert resp.status_code == 200

    # Transcript file should exist immediately
    transcript_path = isolated_sessions / session_id / "transcript.json"
    assert transcript_path.exists(), "Transcript not saved on stop"

    data = json.loads(transcript_path.read_text())
    assert len(data["segments"]) == 2
    assert data["segments"][0]["text"] == "Hello, this is a test."


def test_session_appears_in_list(client, session_on_disk):
    resp = client.get("/api/sessions")
    sessions = resp.json()
    assert len(sessions) == 1
    assert sessions[0]["id"] == session_on_disk
    assert sessions[0]["title"] == "Test Session"
    assert sessions[0]["segment_count"] == 2


def test_get_session_transcript(client, session_on_disk):
    resp = client.get(f"/api/sessions/{session_on_disk}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == session_on_disk
    assert len(data["segments"]) == 2


def test_rename_session(client, session_on_disk):
    resp = client.patch(
        f"/api/sessions/{session_on_disk}",
        json={"title": "Renamed Session"},
    )
    assert resp.status_code == 200

    # Verify rename persisted
    resp = client.get(f"/api/sessions/{session_on_disk}")
    assert resp.json()["title"] == "Renamed Session"


def test_rename_empty_title_rejected(client, session_on_disk):
    resp = client.patch(
        f"/api/sessions/{session_on_disk}",
        json={"title": "   "},
    )
    assert resp.status_code == 400


def test_delete_session(client, session_on_disk, isolated_sessions):
    resp = client.delete(f"/api/sessions/{session_on_disk}")
    assert resp.status_code == 200

    # Should be gone
    assert not (isolated_sessions / session_on_disk).exists()
    resp = client.get(f"/api/sessions/{session_on_disk}")
    assert resp.status_code == 404


def test_export_json(client, session_on_disk):
    resp = client.get(f"/api/sessions/{session_on_disk}/export/json")
    assert resp.status_code == 200
    data = resp.json()
    assert "segments" in data


def test_export_txt(client, session_on_disk):
    resp = client.get(f"/api/sessions/{session_on_disk}/export/txt")
    assert resp.status_code == 200
    assert "Hello, this is a test." in resp.text
    assert "[00:00:00.0]" in resp.text


def test_export_srt(client, session_on_disk):
    resp = client.get(f"/api/sessions/{session_on_disk}/export/srt")
    assert resp.status_code == 200
    assert "-->" in resp.text
    assert "Hello, this is a test." in resp.text


def test_export_unknown_format(client, session_on_disk):
    resp = client.get(f"/api/sessions/{session_on_disk}/export/csv")
    assert resp.status_code == 400


def test_empty_session_cleanup(client, isolated_sessions):
    """Sessions with zero segments should be cleaned up on stop."""
    resp = client.post("/api/session/start", json={"devices": [1]})
    session_id = resp.json()["id"]
    session_dir = isolated_sessions / session_id

    # Don't add any segments
    resp = client.post("/api/session/stop")
    assert resp.status_code == 200
    assert resp.json()["segment_count"] == 0

    # Empty session dir should be removed
    assert not session_dir.exists()


def test_double_stop_is_safe(client):
    """Stopping when no session is active returns an error, not a crash."""
    resp = client.post("/api/session/stop")
    assert resp.status_code == 400
    assert "error" in resp.json()
