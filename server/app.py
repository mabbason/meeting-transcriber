"""
FastAPI web server with WebSocket support for real-time transcription display.
"""

import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from transcriber.pipeline import TranscriptionPipeline

app = FastAPI(title="Meeting Transcriber")
pipeline = TranscriptionPipeline()

STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/session/start")
async def start_session():
    result = pipeline.start_session()
    return JSONResponse(result)


@app.post("/api/session/stop")
async def stop_session():
    result = pipeline.stop_session()
    if result is None:
        return JSONResponse({"error": "No active session"}, status_code=400)
    return JSONResponse(result)


@app.get("/api/session/status")
async def session_status():
    if pipeline.session:
        capturing = pipeline.capture is not None and pipeline.capture.is_running
        return JSONResponse({
            "active": True,
            "id": pipeline.session["id"],
            "started_at": pipeline.session["started_at"],
            "segment_count": len(pipeline.session["segments"]),
            "audio_capturing": capturing,
        })
    return JSONResponse({"active": False, "audio_capturing": False})


@app.get("/api/sessions")
async def list_sessions():
    return JSONResponse(pipeline.get_sessions())


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    data = pipeline.get_session_transcript(session_id)
    if data is None:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return JSONResponse(data)


@app.get("/api/sessions/{session_id}/export/{fmt}")
async def export_session(session_id: str, fmt: str):
    data = pipeline.get_session_transcript(session_id)
    if data is None:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    segments = data.get("segments", [])

    if fmt == "json":
        return JSONResponse(data)

    elif fmt == "txt":
        lines = []
        for seg in segments:
            ts = format_timestamp(seg["start"])
            speaker = seg.get("speaker", "Speaker")
            lines.append(f"[{ts}] {speaker}: {seg['text']}")
        return PlainTextResponse(
            "\n".join(lines),
            headers={"Content-Disposition": f"attachment; filename={session_id}.txt"},
        )

    elif fmt == "srt":
        srt_lines = []
        for i, seg in enumerate(segments, 1):
            start = format_srt_time(seg["start"])
            end = format_srt_time(seg["end"])
            speaker = seg.get("speaker", "Speaker")
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start} --> {end}")
            srt_lines.append(f"{speaker}: {seg['text']}")
            srt_lines.append("")
        return PlainTextResponse(
            "\n".join(srt_lines),
            headers={"Content-Disposition": f"attachment; filename={session_id}.srt"},
        )

    return JSONResponse({"error": f"Unknown format: {fmt}"}, status_code=400)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    pipeline.websocket_clients.add(ws)
    print(f"WebSocket client connected ({len(pipeline.websocket_clients)} total)")

    try:
        # Send current session state if active
        if pipeline.session:
            await ws.send_text(json.dumps({
                "type": "session_state",
                "data": {
                    "active": True,
                    "id": pipeline.session["id"],
                    "segments": [
                        {
                            "start": s["start"],
                            "end": s["end"],
                            "text": s["text"],
                            "speaker": s.get("speaker", "Speaker"),
                        }
                        for s in pipeline.session["segments"]
                    ],
                },
            }))

        # Keep connection alive, handle client messages
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        pass
    finally:
        pipeline.websocket_clients.discard(ws)
        print(f"WebSocket client disconnected ({len(pipeline.websocket_clients)} total)")


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 10)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms}"


def format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
