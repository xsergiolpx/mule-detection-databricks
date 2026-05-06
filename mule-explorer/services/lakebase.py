"""Lakebase connection service with OAuth token rotation."""
import os
import uuid
import psycopg
from databricks.sdk import WorkspaceClient

INSTANCE_NAME = "crr-demo-lakebase"


def get_connection():
    w = WorkspaceClient()
    cred = w.database.generate_database_credential(
        request_id=str(uuid.uuid4()),
        instance_names=[INSTANCE_NAME],
    )
    return psycopg.connect(
        host=os.environ.get("PGHOST"),
        dbname=os.environ.get("PGDATABASE", "databricks_postgres"),
        user=os.environ.get("PGUSER"),
        password=cred.token,
        port=os.environ.get("PGPORT", "5432"),
        sslmode=os.environ.get("PGSSLMODE", "require"),
        autocommit=True,
    )


def query(sql, params=None):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        return [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()
