"""Check PostgreSQL connectivity for SQLSage manually."""

from __future__ import annotations

import os

import psycopg2


def main() -> None:
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "sqlsage"),
        dbname=os.getenv("POSTGRES_DB", "sqlsage"),
    )
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        row = cur.fetchone()
    conn.close()
    print(row[0])


if __name__ == "__main__":
    main()
