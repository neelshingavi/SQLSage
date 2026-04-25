"""FastAPI wrapper for SQLSage environment."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .env import SQLSageEnv

app = FastAPI(title="SQLSage Environment", version="0.1.0")
env: SQLSageEnv | None = None


def get_env() -> SQLSageEnv:
    global env
    if env is None:
        try:
            env = SQLSageEnv()
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=503, detail=f"database_unavailable: {exc}") from exc
    return env


class StepRequest(BaseModel):
    action: str = Field(..., description="Action type, e.g. push_filter")
    rewritten_query: str = Field(..., description="Rewritten SQL text")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(seed: int | None = None) -> dict[str, Any]:
    state = get_env().reset(seed=seed)
    return {"observation": asdict(state)}


@app.post("/step")
def step(payload: StepRequest) -> dict[str, Any]:
    try:
        state, reward, done, info = get_env().step(payload.action, payload.rewritten_query)
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"observation": asdict(state), "reward": reward, "done": done, "info": info}


@app.get("/state")
def state() -> dict[str, Any]:
    try:
        return {"observation": asdict(get_env().state())}
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
