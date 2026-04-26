"""FastAPI application for SQLSage (Meta OpenEnv HTTP/WebSocket API)."""

from __future__ import annotations

import atexit
import os
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


def _training_env_snapshot() -> dict[str, Any]:
    """Return non-secret training env readiness details for UI/status."""
    env_url = os.environ.get("SQLSAGE_ENV_URL", "").strip()
    wandb_project = os.environ.get("WANDB_PROJECT", "").strip()
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", "").strip()
    model_name = os.environ.get("SQLSAGE_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct").strip()
    output_dir = os.environ.get("SQLSAGE_OUTPUT_DIR", "sqlsage-trained").strip()
    merged_dir = os.environ.get("SQLSAGE_MERGED_DIR", "sqlsage-trained-merged").strip()
    wandb_key_present = bool(os.environ.get("WANDB_API_KEY", "").strip())
    return {
        "SQLSAGE_ENV_URL": env_url,
        "WANDB_PROJECT": wandb_project,
        "WANDB_RUN_NAME": wandb_run_name,
        "SQLSAGE_MODEL_NAME": model_name,
        "SQLSAGE_OUTPUT_DIR": output_dir,
        "SQLSAGE_MERGED_DIR": merged_dir,
        "WANDB_API_KEY_SET": wandb_key_present,
        "ready": bool(env_url and wandb_key_present and wandb_project and wandb_run_name),
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


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for Colab / load balancers (must return 200 when Space is up)."""
    return {"status": "ok", "service": "sqlsage-openenv"}


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
        payload = dict(_training_status)
    payload["env"] = _training_env_snapshot()
    return payload


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
      body { font-family: Arial, sans-serif; margin: 2rem; background: #0f172a; color: #e2e8f0; line-height: 1.45; }
      button { background: #2563eb; color: white; border: none; padding: 0.6rem 1rem; border-radius: 8px; cursor: pointer; margin-right: 0.5rem; }
      button:disabled { background: #64748b; cursor: not-allowed; }
      .box { margin-top: 1rem; padding: 1rem; border: 1px solid #334155; border-radius: 10px; background: #111827; }
      code { color: #93c5fd; }
      .ok { color: #34d399; }
      .warn { color: #fbbf24; }
      .err { color: #f87171; }
      ul { margin-top: 0.3rem; }
    </style>
  </head>
  <body>
    <h2>SQLSage GRPO Training</h2>
    <p>Trigger GRPO training from this page in the current HF Space runtime.</p>
    <div class="box">
      <strong>What happens after Start:</strong>
      <ul>
        <li>Loads Unsloth + TRL + model weights</li>
        <li>Calls your SQLSage env via <code>SQLSAGE_ENV_URL</code></li>
        <li>Runs GRPO training and logs metrics to W&amp;B</li>
        <li>Saves adapter and merged 16-bit model outputs</li>
      </ul>
      <strong>Required env vars:</strong>
      <ul>
        <li><code>SQLSAGE_ENV_URL</code> (required)</li>
        <li><code>WANDB_API_KEY</code> (secret, required)</li>
        <li><code>WANDB_PROJECT</code> and <code>WANDB_RUN_NAME</code> (required)</li>
        <li><code>SQLSAGE_MODEL_NAME</code>, <code>SQLSAGE_OUTPUT_DIR</code>, <code>SQLSAGE_MERGED_DIR</code> (optional)</li>
      </ul>
      <p>Tip: If status shows <code>failed</code>, check Space logs and W&amp;B run page.</p>
    </div>
    <button id="startBtn" onclick="startTraining()">Start Training</button>
    <button onclick="refreshStatus()">Refresh</button>
    <div class="box"><pre id="preflight">Checking environment...</pre></div>
    <div class="box"><pre id="status">Loading status...</pre></div>
    <script>
      function fmt(v) { return (v === undefined || v === null || v === '') ? '<missing>' : String(v); }
      function yesNo(ok) { return ok ? 'OK' : 'MISSING'; }
      async function refreshStatus() {
        const res = await fetch('/train/status');
        const data = await res.json();
        const pct = Math.round((data.progress || 0) * 100);
        const env = data.env || {};
        const preflight = {
          ready: !!env.ready,
          SQLSAGE_ENV_URL: fmt(env.SQLSAGE_ENV_URL),
          WANDB_API_KEY_SET: yesNo(!!env.WANDB_API_KEY_SET),
          WANDB_PROJECT: fmt(env.WANDB_PROJECT),
          WANDB_RUN_NAME: fmt(env.WANDB_RUN_NAME),
          SQLSAGE_MODEL_NAME: fmt(env.SQLSAGE_MODEL_NAME),
          SQLSAGE_OUTPUT_DIR: fmt(env.SQLSAGE_OUTPUT_DIR),
          SQLSAGE_MERGED_DIR: fmt(env.SQLSAGE_MERGED_DIR),
        };
        document.getElementById('preflight').textContent = JSON.stringify(preflight, null, 2);
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
