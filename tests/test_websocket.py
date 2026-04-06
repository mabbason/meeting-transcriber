"""
WebSocket tests.

The /ws endpoint is the backbone of live transcription. These tests ensure
it exists, accepts connections, and delivers the expected message types.
"""

import json
from unittest.mock import MagicMock

from tests.conftest import pipeline, FAKE_SEGMENTS


def test_websocket_connects(client):
    """WebSocket handshake succeeds."""
    with client.websocket_connect("/ws") as ws:
        ws.close()


def test_websocket_ping_pong(client):
    """Server responds to ping with pong."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "ping"})
        resp = ws.receive_json()
        assert resp["type"] == "pong"


def test_websocket_receives_session_state(client):
    """When a session is active, new WS clients get current segments."""
    import copy

    # Start a session and inject segments
    resp = client.post("/api/session/start", json={"devices": [1]})
    assert resp.status_code == 200
    pipeline.session["segments"] = copy.deepcopy(FAKE_SEGMENTS)

    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "session_state"
        assert msg["data"]["active"] is True
        assert len(msg["data"]["segments"]) == 2
        assert msg["data"]["segments"][0]["text"] == "Hello, this is a test."

    # Cleanup
    pipeline.session["segments"] = []
    client.post("/api/session/stop")


def test_websocket_no_state_when_idle(client):
    """When no session is active, WS clients don't get session_state."""
    with client.websocket_connect("/ws") as ws:
        # Send a ping to verify the connection works
        ws.send_json({"type": "ping"})
        resp = ws.receive_json()
        assert resp["type"] == "pong"
        # No session_state message was sent before our ping


def test_websocket_client_tracking(client):
    """Connected clients are tracked and cleaned up on disconnect."""
    assert len(pipeline.websocket_clients) == 0

    with client.websocket_connect("/ws") as ws:
        assert len(pipeline.websocket_clients) == 1
        ws.close()

    assert len(pipeline.websocket_clients) == 0
