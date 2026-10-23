import os

import psycopg
from dotenv import load_dotenv

load_dotenv()


def get_pg_connection():
    return psycopg.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "enterprise_graphrag"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD"),
        connect_timeout=5,
    )


def check_postgres():
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), version()")
            database, version = cur.fetchone()
            cur.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )
            row = cur.fetchone()

    if row is None:
        raise RuntimeError(
            "pgvector is not enabled in the project database."
        )

    return {
        "database": database,
        "postgres_version": version.split(",")[0],
        "pgvector_version": row[0],
    }
