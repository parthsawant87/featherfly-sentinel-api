# db_logger.py — SENTINEL / EduScope SQLite Logger + Claude API Budget
# Used by: api_server_sentinel.py, api_server_eduscope.py, claude_reporter.py
#
# Functions:
#   init_db()              — create tables on startup (safe to call repeatedly)
#   log_prediction(result) — log one classification result
#   can_call_claude()      — True if daily budget not exhausted
#   record_claude_call()   — increment today's counter after a Claude call
#   get_recent(n)          — last n predictions, newest first
#   get_daily_stats(date)  — yield % and defect rate for a date
#   get_claude_usage_today() — budget status dict

import sqlite3, json, time, os
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional
import config as cfg

CLAUDE_DAILY_LIMIT = 20   # calls per device per day — edit here to change
DB_PATH            = cfg.DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    date_str        TEXT    NOT NULL,
    predicted_class TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    severity        TEXT,
    filename        TEXT,
    module          TEXT    DEFAULT 'sentinel',
    result_json     TEXT
);
CREATE TABLE IF NOT EXISTS claude_usage (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date_str    TEXT    NOT NULL,
    call_count  INTEGER NOT NULL DEFAULT 0,
    UNIQUE(date_str)
);
CREATE INDEX IF NOT EXISTS idx_pred_date  ON predictions(date_str);
CREATE INDEX IF NOT EXISTS idx_pred_class ON predictions(predicted_class);
"""


def _get_conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist. Safe to call on every startup."""
    with _get_conn() as conn:
        conn.executescript(SCHEMA)
    print(f"[db] ✓ Database ready: {DB_PATH}")


def log_prediction(result: dict, module: str = "sentinel"):
    """Log a classification result.
    result: the dict returned by /predict or /identify endpoint
    module: 'sentinel' or 'eduscope'
    """
    try:
        with _get_conn() as conn:
            conn.execute(
                """INSERT INTO predictions
                   (timestamp, date_str, predicted_class, confidence,
                    severity, filename, module, result_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(),
                    str(date.today()),
                    result.get("predicted_class") or result.get("specimen", "UNKNOWN"),
                    result.get("confidence", 0.0),
                    result.get("severity", ""),
                    result.get("filename", ""),
                    module,
                    json.dumps(result),
                )
            )
    except Exception as e:
        print(f"[db] log_prediction failed: {e}")


def can_call_claude() -> bool:
    """Returns True if today's Claude call budget is not exhausted.
    Call this BEFORE making any Anthropic API request.
    Returns False when 20 calls have been made today — use static fallback.
    """
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT call_count FROM claude_usage WHERE date_str = ?",
                (str(date.today()),)
            ).fetchone()
            count = row["call_count"] if row else 0
            return count < CLAUDE_DAILY_LIMIT   # ← fixed: < was stripped by HTML renderer
    except Exception as e:
        print(f"[db] can_call_claude error: {e}")
        return True   # fail open — allow call if DB error


def record_claude_call():
    """Increment today's Claude API call counter.
    Call this AFTER a successful Anthropic API call.
    """
    try:
        with _get_conn() as conn:
            conn.execute(
                """INSERT INTO claude_usage (date_str, call_count) VALUES (?, 1)
                   ON CONFLICT(date_str) DO UPDATE SET call_count = call_count + 1""",
                (str(date.today()),)
            )
    except Exception as e:
        print(f"[db] record_claude_call error: {e}")


def get_claude_usage_today() -> dict:
    """Return budget status for today. Used by /health endpoint."""
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT call_count FROM claude_usage WHERE date_str = ?",
                (str(date.today()),)
            ).fetchone()
            used = row["call_count"] if row else 0
            return {
                "used":      used,
                "remaining": max(0, CLAUDE_DAILY_LIMIT - used),
                "limit":     CLAUDE_DAILY_LIMIT,
                "pct":       round(used / CLAUDE_DAILY_LIMIT * 100),
            }
    except:
        return {"used":0, "remaining":CLAUDE_DAILY_LIMIT, "limit":CLAUDE_DAILY_LIMIT, "pct":0}


def get_recent(n: int = 20, module: str = None) -> List[dict]:
    """Return the last n predictions, newest first.
    module: filter by 'sentinel' or 'eduscope', or None for all.
    """
    try:
        with _get_conn() as conn:
            if module:
                rows = conn.execute(
                    "SELECT * FROM predictions WHERE module=? ORDER BY timestamp DESC LIMIT ?",
                    (module, n)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", (n,)
                ).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"[db] get_recent error: {e}")
        return []


def get_daily_stats(date_str: str = None) -> dict:
    """Return class distribution + yield for a given date (default: today).
    Used by spc_monitor.py for daily defect rate tracking.
    """
    d = date_str or str(date.today())
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                """SELECT predicted_class, COUNT(*) as cnt
                   FROM predictions WHERE date_str = ?
                   GROUP BY predicted_class""",
                (d,)
            ).fetchall()
            counts = {r["predicted_class"]: r["cnt"] for r in rows}
            total  = sum(counts.values())
            clean  = counts.get("CLEAN", 0)
            return {
                "date":        d,
                "total":       total,
                "clean":       clean,
                "defects":     total - clean,
                "yield_pct":   round(clean / total * 100, 2) if total > 0 else 0,
                "defect_rate": round((total - clean) / total, 4) if total > 0 else 0,
                "counts":      counts,
            }
    except Exception as e:
        print(f"[db] get_daily_stats error: {e}")
        return {"date":d,"total":0,"clean":0,"defects":0,"yield_pct":0,"defect_rate":0,"counts":{}}


def get_all_predictions(module: str = None) -> List[dict]:
    """Return all predictions ordered by time. Used by spc_monitor and benchmark."""
    try:
        with _get_conn() as conn:
            q = "SELECT * FROM predictions"
            params = ()
            if module:
                q += " WHERE module=?"
                params = (module,)
            q += " ORDER BY timestamp ASC"
            return [dict(r) for r in conn.execute(q, params).fetchall()]
    except:
        return []


# Auto-initialise on import
init_db()