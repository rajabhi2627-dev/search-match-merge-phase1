"""
Local SQLite storage (data/matching.db). Uses Python's built-in sqlite3 module.

Tables
  master_records : the ~1,000 master entities
  test_records   : the 100 generated incoming records + ground truth
  match_results  : output of running the matcher on the test records
  monitoring_results : 6 monthly batches used for Week 4 performance monitoring
  meta           : which dataset the database was built from
"""
import sqlite3

import pandas as pd

import config as C
from data_generator import generate_master_records

MASTER_COLUMNS = ["entity_id"] + C.FIELDS


def get_connection():
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(C.DB_PATH)


def _stored_dataset(conn):
    if not _table_exists(conn, "meta"):
        return None
    row = conn.execute("SELECT value FROM meta WHERE key = 'dataset'").fetchone()
    return row[0] if row else None


def init_db():
    """Create the database and fill the master table if it does not exist yet.

    If the database was built from a different dataset, all tables are rebuilt.
    """
    with get_connection() as conn:
        if _stored_dataset(conn) != C.DATASET_NAME:
            for table in ["master_records", "test_records", "match_results", "monitoring_results", "meta"]:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("INSERT INTO meta VALUES ('dataset', ?)", (C.DATASET_NAME,))
        conn.execute(
            """CREATE TABLE IF NOT EXISTS master_records (
                   entity_id     INTEGER PRIMARY KEY,
                   name          TEXT,
                   email         TEXT,
                   phone         TEXT,
                   address       TEXT,
                   date_of_birth TEXT,
                   company       TEXT,
                   city          TEXT
               )"""
        )
        count = conn.execute("SELECT COUNT(*) FROM master_records").fetchone()[0]
        if count == 0:
            records = generate_master_records()
            conn.executemany(
                f"INSERT INTO master_records ({', '.join(MASTER_COLUMNS)}) VALUES ({', '.join('?' * len(MASTER_COLUMNS))})",
                [tuple(r[c] for c in MASTER_COLUMNS) for r in records],
            )
    return master_count()


def master_count():
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM master_records").fetchone()[0]


def load_master():
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM master_records ORDER BY entity_id", conn)


def get_master_record(entity_id):
    with get_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM master_records WHERE entity_id = ?", conn, params=(int(entity_id),))
    return df.iloc[0].to_dict() if len(df) else None


def _table_exists(conn, table):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def save_test_records(records):
    """Replace the stored test records (and clear old results, which no longer apply)."""
    df = pd.DataFrame(records)
    df["true_entity_id"] = df["true_entity_id"].astype("Int64")
    with get_connection() as conn:
        df.to_sql("test_records", conn, if_exists="replace", index=False)
        conn.execute("DROP TABLE IF EXISTS match_results")


def load_test_records():
    with get_connection() as conn:
        if not _table_exists(conn, "test_records"):
            return pd.DataFrame()
        df = pd.read_sql_query("SELECT * FROM test_records ORDER BY test_record_id", conn)
    df["true_entity_id"] = df["true_entity_id"].astype("Int64")
    return df


def save_match_results(df):
    with get_connection() as conn:
        df.to_sql("match_results", conn, if_exists="replace", index=False)


def _load_results(table, order_by):
    with get_connection() as conn:
        if not _table_exists(conn, table):
            return pd.DataFrame()
        df = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY {order_by}", conn)
    for col in ["true_entity_id", "predicted_entity_id", "best_candidate_id"]:
        df[col] = df[col].astype("Int64")
    return df


def load_match_results():
    return _load_results("match_results", "test_record_id")


def save_monitoring_results(df):
    with get_connection() as conn:
        df.to_sql("monitoring_results", conn, if_exists="replace", index=False)


def load_monitoring_results():
    df = _load_results("monitoring_results", "rowid")
    if not df.empty:
        df["disputed"] = df["disputed"].astype(bool)
    return df
