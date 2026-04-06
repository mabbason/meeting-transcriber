"""
TranscriptionPipeline logic tests.

Tests chunk processing, word deduplication, segment rebuilding, and
session persistence — the core transcription logic that doesn't need
real audio hardware.
"""

import copy
import json
from unittest.mock import MagicMock

import numpy as np

from tests.conftest import pipeline, FAKE_SEGMENTS
from transcriber.pipeline import TranscriptionPipeline


class TestWordDedup:
    """Word deduplication between overlapping audio chunks."""

    def test_no_overlap(self):
        pipeline._prev_words = []
        words = [
            {"start": 0.0, "end": 0.5, "word": "hello"},
            {"start": 0.6, "end": 1.0, "word": " world"},
        ]
        result = pipeline._dedup_words(words)
        assert len(result) == 2

    def test_timestamp_overlap(self):
        pipeline._prev_words = [
            {"start": 0.0, "end": 0.5, "word": "hello"},
            {"start": 0.6, "end": 1.0, "word": " world"},
        ]
        words = [
            {"start": 0.7, "end": 1.0, "word": " world"},  # overlaps with prev
            {"start": 1.1, "end": 1.5, "word": " foo"},
        ]
        result = pipeline._dedup_words(words)
        # " world" should be skipped (timestamp before prev end)
        assert len(result) == 1
        assert result[0]["word"] == " foo"

    def test_sequence_alignment(self):
        pipeline._prev_words = [
            {"start": 0.0, "end": 0.3, "word": "the"},
            {"start": 0.4, "end": 0.7, "word": " quick"},
            {"start": 0.8, "end": 1.0, "word": " brown"},
        ]
        words = [
            {"start": 0.4, "end": 0.7, "word": " quick"},
            {"start": 0.8, "end": 1.0, "word": " brown"},
            {"start": 1.1, "end": 1.5, "word": " fox"},
        ]
        result = pipeline._dedup_words(words)
        assert len(result) == 1
        assert result[0]["word"] == " fox"

    def test_empty_prev_passes_through(self):
        pipeline._prev_words = []
        words = [{"start": 0.0, "end": 0.5, "word": "test"}]
        result = pipeline._dedup_words(words)
        assert result == words

    def test_empty_new_returns_empty(self):
        pipeline._prev_words = [{"start": 0.0, "end": 0.5, "word": "test"}]
        result = pipeline._dedup_words([])
        assert result == []


class TestWordsToSegments:
    """Rebuilding segments from deduped word lists."""

    def test_words_within_segment(self):
        words = [
            {"start": 0.0, "end": 0.5, "word": "Hello"},
            {"start": 0.6, "end": 1.0, "word": " world"},
        ]
        orig_segments = [{"start": 0.0, "end": 1.0, "text": "Hello world"}]
        result = pipeline._words_to_segments(words, orig_segments)
        assert len(result) == 1
        assert result[0]["text"] == "Hello world"

    def test_empty_words(self):
        result = pipeline._words_to_segments([], [])
        assert result == []

    def test_remaining_unmatched_words(self):
        words = [
            {"start": 5.0, "end": 5.5, "word": "Extra"},
        ]
        orig_segments = [{"start": 0.0, "end": 1.0, "text": "Original"}]
        result = pipeline._words_to_segments(words, orig_segments)
        assert len(result) == 1
        assert result[0]["text"] == "Extra"


class TestSessionPersistence:
    """Transcript saving and loading."""

    def test_save_and_load(self, client):
        resp = client.post("/api/session/start", json={"devices": [1]})
        session_id = resp.json()["id"]

        # Inject segments
        pipeline.session["segments"] = copy.deepcopy(FAKE_SEGMENTS)

        # Stop saves transcript
        client.post("/api/session/stop")

        # Load it back via API
        resp = client.get(f"/api/sessions/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["segments"]) == 2
        assert data["title"].startswith("Untitled")

    def test_untitled_numbering(self, client):
        """Successive sessions get incrementing Untitled NNN names."""
        import time

        titles = []
        for _ in range(3):
            resp = client.post("/api/session/start", json={"devices": [1]})
            sid = resp.json()["id"]
            pipeline.session["segments"] = copy.deepcopy(FAKE_SEGMENTS)
            client.post("/api/session/stop")

            resp = client.get(f"/api/sessions/{sid}")
            titles.append(resp.json()["title"])
            time.sleep(1.1)  # Session IDs are timestamp-based, need 1s gap

        assert titles == ["Untitled 001", "Untitled 002", "Untitled 003"]

    def test_default_device_selection(self):
        """Default devices: first loopback + first mic."""
        pipeline.available_devices = [
            {"index": 10, "name": "Mic A", "type": "microphone", "channels": 1},
            {"index": 20, "name": "Loopback A", "type": "loopback", "channels": 2},
            {"index": 30, "name": "Mic B", "type": "microphone", "channels": 1},
        ]
        defaults = pipeline._default_devices()
        assert 20 in defaults  # first loopback
        assert 10 in defaults  # first mic
        assert 30 not in defaults
