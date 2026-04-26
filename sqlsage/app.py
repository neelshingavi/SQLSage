"""FastAPI application for SQLSage (Meta OpenEnv HTTP/WebSocket API)."""

from __future__ import annotations

import atexit
import threading
from datetime import datetime, timezone
from typing import Any

from fastapi import BackgroundTasks
from fastapi.responses import HTMLResponse
from openenv.core.env_server.http_server import create_fastapi_app

from .openenv_bridge import SQLSageOpenEnvironment
from .openenv_types import SQLSageServerObservation, SQLSageStepAction
from train import run_training

_singleton: SQLSageOpenEnvironment | None = None
_training_lock = threading.Lock()
_training_status: dict[str, Any] = {
    "running": False,
    "progress": 0.0,
    "message": "idle",
    "started_at": None,
    "finished_at": None,
    "result": None,
    "error": None,
}


def _sqlsage_factory() -> SQLSageOpenEnvironment:
    """Single long-lived env instance (HTTP handlers call close() after each request)."""
    global _singleton
    if _singleton is None:
        _singleton = SQLSageOpenEnvironment()
    return _singleton


def _shutdown_sqlsage() -> None:
    global _singleton
    if _singleton is not None:
        _singleton.shutdown()
        _singleton = None


atexit.register(_shutdown_sqlsage)

app = create_fastapi_app(
    _sqlsage_factory,
    SQLSageStepAction,
    SQLSageServerObservation,
)


def _now_iso() -> str:
    """Return UTC timestamp string for status payloads."""
    return datetime.now(tz=timezone.utc).isoformat()


def _progress_callback(progress: float, message: str) -> None:
    """Update in-memory training progress state."""
    with _training_lock:
        _training_status["progress"] = float(max(0.0, min(1.0, progress)))
        _training_status["message"] = str(message)


def _run_training_task() -> None:
    """Execute training in background and capture terminal status."""
    with _training_lock:
        _training_status["running"] = True
        _training_status["progress"] = 0.0
        _training_status["message"] = "starting"
        _training_status["started_at"] = _now_iso()
        _training_status["finished_at"] = None
        _training_status["result"] = None
        _training_status["error"] = None
    try:
        result = run_training(progress_callback=_progress_callback)
        with _training_lock:
            _training_status["result"] = result
            _training_status["progress"] = 1.0
            _training_status["message"] = "completed"
            _training_status["finished_at"] = _now_iso()
            _training_status["running"] = False
    except Exception as error:
        with _training_lock:
            _training_status["error"] = str(error)
            _training_status["message"] = "failed"
            _training_status["finished_at"] = _now_iso()
            _training_status["running"] = False


@app.post("/train/start")
def start_training(background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Start GRPO training in background if not currently running."""
    with _training_lock:
        if _training_status["running"]:
            return {"ok": False, "status": dict(_training_status), "message": "training already running"}
    background_tasks.add_task(_run_training_task)
    return {"ok": True, "message": "training started"}


@app.get("/train/status")
def training_status() -> dict[str, Any]:
    """Fetch current training status."""
    with _training_lock:
        return dict(_training_status)


@app.get("/train", response_class=HTMLResponse)
def training_page() -> str:
    """Simple HF-visible training dashboard and trigger page."""
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>SQLSage Training Control</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 2rem; background: #0f172a; color: #e2e8f0; }
      button { background: #2563eb; color: white; border: none; padding: 0.6rem 1rem; border-radius: 8px; cursor: pointer; }
      button:disabled { background: #64748b; cursor: not-allowed; }
      .box { margin-top: 1rem; padding: 1rem; border: 1px solid #334155; border-radius: 10px; background: #111827; }
      code { color: #93c5fd; }
    </style>
  </head>
  <body>
    <h2>SQLSage GRPO Training</h2>
    <p>Trigger training from this page in HF Space runtime.</p>
    <button id="startBtn" onclick="startTraining()">Start Training</button>
    <div class="box"><pre id="status">Loading status...</pre></div>
    <script>
      async function refreshStatus() {
        const res = await fetch('/train/status');
        const data = await res.json();
        const pct = Math.round((data.progress || 0) * 100);
        document.getElementById('status').textContent =
          JSON.stringify({...data, progress_percent: pct}, null, 2);
        document.getElementById('startBtn').disabled = !!data.running;
      }
      async function startTraining() {
        const res = await fetch('/train/start', { method: 'POST' });
        const data = await res.json();
        await refreshStatus();
        if (!data.ok) alert(data.message || 'Unable to start training');
      }
      refreshStatus();
      setInterval(refreshStatus, 3000);
    </script>
  </body>
</html>
"""
