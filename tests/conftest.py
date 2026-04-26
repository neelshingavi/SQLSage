"""Shared pytest fixtures for PostgreSQL-backed tests."""

import psycopg2
import pytest


@pytest.fixture(scope="session")
def postgres_conn():
    """Create and close a real PostgreSQL connection for tests."""
    connection = psycopg2.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="sqlsage",
        dbname="sqlsage",
    )
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture(scope="module")
def conn(postgres_conn):
    """Backward-compatible module-scoped alias fixture for parser tests."""
    yield postgres_conn
