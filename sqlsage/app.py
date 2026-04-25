"""FastAPI wrapper for SQLSage environment."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .env import SQLSageEnv

app = FastAPI(title="SQLSage Environment", version="0.1.0")
env = SQLSageEnv()


class StepRequest(BaseModel):
    action: str = Field(..., description="Action type, e.g. push_filter")
    rewritten_query: str = Field(..., description="Rewritten SQL text")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(seed: int | None = None) -> dict[str, Any]:
    state = env.reset(seed=seed)
    return {"observation": asdict(state)}


@app.post("/step")
def step(payload: StepRequest) -> dict[str, Any]:
    try:
        state, reward, done, info = env.step(payload.action, payload.rewritten_query)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"observation": asdict(state), "reward": reward, "done": done, "info": info}


@app.get("/state")
def state() -> dict[str, Any]:
    try:
        return {"observation": asdict(env.state())}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
