"""Pytest fixtures for SQLSage (optional live PostgreSQL)."""

from __future__ import annotations

import os

import pytest


def _postgres_params() -> dict[str, object]:
    return {
        "host": os.getenv("POSTGRES_HOST", "127.0.0.1"),
        "port": int(os.getenv("POSTGRES_PORT", "5433")),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "sqlsage"),
        "dbname": os.getenv("POSTGRES_DB", "sqlsage"),
    }


@pytest.fixture(scope="session")
def postgres_conn():
    """Live connection when DB is reachable; else skip."""
    try:
        import psycopg2
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"psycopg2 not installed: {exc}")

    try:
        conn = psycopg2.connect(**_postgres_params())
    except Exception as exc:
        pytest.skip(f"PostgreSQL not available for integration tests: {exc}")
    try:
        yield conn
    finally:
        conn.close()
