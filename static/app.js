let ws = null;
let activeSessionId = null;
let isRecording = false;
let timerInterval = null;
let startTime = null;
let viewingSessionId = null;

function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws`);

    ws.onopen = () => console.log('WebSocket connected');

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === 'segment' && isRecording) {
            appendSegment(msg.data);
        } else if (msg.type === 'session_state') {
            if (msg.data.active) {
                msg.data.segments.forEach(seg => appendSegment(seg));
            }
        }
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected, reconnecting...');
        setTimeout(connectWebSocket, 2000);
    };
}

async function startSession() {
    const resp = await fetch('/api/session/start', { method: 'POST' });
    const data = await resp.json();

    activeSessionId = data.id;
    viewingSessionId = data.id;
    isRecording = true;

    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    document.getElementById('status').className = 'status recording';
    document.getElementById('status').textContent = 'Recording';
    document.getElementById('export-bar').style.display = 'none';

    const transcript = document.getElementById('transcript');
    transcript.innerHTML = '';

    startTime = Date.now();
    timerInterval = setInterval(updateTimer, 1000);

    loadSessions();
}

async function stopSession() {
    const resp = await fetch('/api/session/stop', { method: 'POST' });
    const data = await resp.json();

    isRecording = false;
    activeSessionId = null;

    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-stop').disabled = true;
    document.getElementById('status').className = 'status idle';
    document.getElementById('status').textContent = 'Idle';

    if (viewingSessionId) {
        document.getElementById('export-bar').style.display = 'flex';
    }

    clearInterval(timerInterval);
    loadSessions();
}

function updateTimer() {
    if (!startTime) return;
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const h = String(Math.floor(elapsed / 3600)).padStart(2, '0');
    const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
    const s = String(elapsed % 60).padStart(2, '0');
    document.getElementById('timer').textContent = `${h}:${m}:${s}`;
}

function appendSegment(seg) {
    const transcript = document.getElementById('transcript');
    const placeholder = transcript.querySelector('.placeholder');
    if (placeholder) placeholder.remove();

    const div = document.createElement('div');
    div.className = 'segment';
    div.dataset.speaker = seg.speaker;

    const ts = formatTimestamp(seg.start);
    div.innerHTML = `
        <div class="meta">
            <span class="timestamp">[${ts}]</span>
            <span class="speaker">${seg.speaker}</span>
        </div>
        <div class="text">${escapeHtml(seg.text)}</div>
    `;

    transcript.appendChild(div);
    transcript.scrollTop = transcript.scrollHeight;
}

function formatTimestamp(seconds) {
    const h = String(Math.floor(seconds / 3600)).padStart(2, '0');
    const m = String(Math.floor((seconds % 3600) / 60)).padStart(2, '0');
    const s = String(Math.floor(seconds % 60)).padStart(2, '0');
    const ms = Math.floor((seconds % 1) * 10);
    return `${h}:${m}:${s}.${ms}`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function loadSessions() {
    const resp = await fetch('/api/sessions');
    const sessions = await resp.json();
    const list = document.getElementById('session-list');

    if (sessions.length === 0) {
        list.innerHTML = '<p class="empty">No sessions yet</p>';
        return;
    }

    list.innerHTML = sessions.map(s => {
        const date = new Date(s.started_at);
        const dateStr = date.toLocaleDateString('en-US', {
            month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
        });
        const active = s.id === viewingSessionId ? ' active' : '';
        return `
            <div class="session-item${active}">
                <div class="session-row" onclick="viewSession('${s.id}')">
                    <div class="date">${dateStr}</div>
                    <div class="meta">${s.segment_count} segments</div>
                </div>
                <button class="btn-delete" onclick="event.stopPropagation(); deleteSession('${s.id}')" title="Delete session">&times;</button>
            </div>
        `;
    }).join('');
}

async function viewSession(sessionId) {
    if (isRecording && sessionId !== activeSessionId) {
        return; // Don't switch away from live session while recording
    }

    const resp = await fetch(`/api/sessions/${sessionId}`);
    const data = await resp.json();

    viewingSessionId = sessionId;
    const transcript = document.getElementById('transcript');
    transcript.innerHTML = '';

    data.segments.forEach(seg => appendSegment(seg));

    document.getElementById('export-bar').style.display = 'flex';
    loadSessions(); // Refresh to update active highlight
}

async function deleteSession(sessionId) {
    if (!confirm('Delete this recording?')) return;
    await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
    if (viewingSessionId === sessionId) {
        viewingSessionId = null;
        document.getElementById('transcript').innerHTML = '<p class="placeholder">Start a recording or select a past session to view the transcript.</p>';
        document.getElementById('export-bar').style.display = 'none';
    }
    loadSessions();
}

function exportSession(fmt) {
    if (!viewingSessionId) return;
    window.open(`/api/sessions/${viewingSessionId}/export/${fmt}`, '_blank');
}

async function pollStatus() {
    try {
        const resp = await fetch('/api/session/status');
        const data = await resp.json();

        const audioEl = document.getElementById('audio-status');
        if (data.active && data.audio_capturing) {
            audioEl.innerHTML = '<span class="connected">Capturing</span>';
        } else if (data.active) {
            audioEl.innerHTML = '<span class="disconnected">Starting...</span>';
        } else {
            audioEl.innerHTML = 'Idle';
        }
    } catch (e) {
        // Server not reachable
    }
}

// Initialize
connectWebSocket();
loadSessions();
setInterval(pollStatus, 3000);
pollStatus();
