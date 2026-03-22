# active_learner.py — SENTINEL Active Learning Queue
# Strategy: Uncertainty sampling — flag predictions below confidence threshold
# These images are most valuable to add to your training set
#
# Usage:
#   from active_learner import flag_if_uncertain   ← call after every /predict
#   python active_learner.py                       ← print current queue stats
#   python active_learner.py --export              ← export queue to CSV
#   python active_learner.py --import FILE.csv     ← import reviewed labels

import sqlite3, json, csv, argparse
from datetime import datetime, date
from pathlib import Path
from typing import List
import config as cfg
from db_logger import _get_conn

UNCERTAINTY_THRESHOLD = cfg.CONFIDENCE_THRESHOLD   # 0.60 default from config
MARGIN_THRESHOLD      = 0.15   # flag if top-1 minus top-2 probability < 0.15
MAX_QUEUE_SIZE        = 500    # max items in pending queue at any time

# ── SCHEMA (extends db_logger tables) ────────────────────────────────────────
AL_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_queue (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        REAL    NOT NULL,
    filename         TEXT    NOT NULL,
    predicted_class  TEXT    NOT NULL,
    confidence       REAL    NOT NULL,
    probabilities    TEXT,
    uncertainty_type TEXT,
    status           TEXT DEFAULT 'pending',
    true_label       TEXT,
    reviewed_at      TEXT,
    notes            TEXT
);
CREATE INDEX IF NOT EXISTS idx_rq_status ON review_queue(status);
CREATE INDEX IF NOT EXISTS idx_rq_conf   ON review_queue(confidence);
"""

def _init_al_db():
    with _get_conn() as conn:
        conn.executescript(AL_SCHEMA)

_init_al_db()


def flag_if_uncertain(result: dict) -> bool:
    """Check if a prediction result should be flagged for human review.
    Call this after every /predict or /identify call.

    Flags when:
      (1) max confidence < UNCERTAINTY_THRESHOLD (absolute low confidence)
      (2) top-1 minus top-2 margin < MARGIN_THRESHOLD (two classes nearly tied)

    Returns True if flagged, False otherwise.
    """
    conf  = result.get("confidence", 1.0)
    probs = result.get("probabilities", {})
    fname = result.get("filename", "")
    cls   = result.get("predicted_class") or result.get("specimen", "")

    uncertainty_type = None

    if conf < UNCERTAINTY_THRESHOLD:   # ← fixed: was 'conf UNCERTAINTY_THRESHOLD'
        uncertainty_type = "low_confidence"
    elif isinstance(probs, dict) and len(probs) >= 2:
        # Fixed: original code was cut off mid-sentence here
        sorted_probs = sorted(probs.values(), reverse=True)
        margin = sorted_probs[0] - sorted_probs[1]   # top-1 minus top-2
        if margin < MARGIN_THRESHOLD:
            uncertainty_type = "low_margin"

    if not uncertainty_type:
        return False

    try:
        with _get_conn() as conn:
            # Check queue isn't full
            count = conn.execute(
                "SELECT COUNT(*) FROM review_queue WHERE status='pending'"
            ).fetchone()[0]
            if count >= MAX_QUEUE_SIZE:
                return False

            # Check for duplicate (same filename already pending)
            dup = conn.execute(
                "SELECT id FROM review_queue WHERE filename=? AND status='pending'",
                (fname,)
            ).fetchone()
            if dup: return False

            conn.execute(
                """INSERT INTO review_queue
                   (timestamp, filename, predicted_class, confidence,
                    probabilities, uncertainty_type)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    __import__("time").time(),
                    fname, cls, conf,
                    json.dumps(probs),
                    uncertainty_type,
                )
            )
        return True
    except Exception as e:
        print(f"[al] flag_if_uncertain error: {e}")
        return False


def get_review_queue(status: str = "pending", limit: int = 50) -> List[dict]:
    """Get items from the review queue ordered by lowest confidence first."""
    try:
        with _get_conn() as conn:
            if status == "all":
                rows = conn.execute(
                    "SELECT * FROM review_queue ORDER BY confidence ASC LIMIT ?",
                    (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM review_queue WHERE status=? ORDER BY confidence ASC LIMIT ?",
                    (status, limit)
                ).fetchall()
            return [dict(r) for r in rows]
    except:
        return []


def mark_reviewed(item_id: int, true_label: str, notes: str = ""):
    """Mark a review queue item as reviewed with the correct label.
    After marking, copy image to data/dataset_raw/<class>/ for retraining.
    """
    try:
        with _get_conn() as conn:
            conn.execute(
                """UPDATE review_queue
                   SET status='reviewed', true_label=?, reviewed_at=?, notes=?
                   WHERE id=?""",
                (true_label, datetime.now().isoformat(), notes, item_id)
            )
    except Exception as e:
        print(f"[al] mark_reviewed error: {e}")


def get_queue_stats() -> dict:
    """Summary stats for the review queue."""
    try:
        with _get_conn() as conn:
            total    = conn.execute("SELECT COUNT(*) FROM review_queue").fetchone()[0]
            pending  = conn.execute("SELECT COUNT(*) FROM review_queue WHERE status='pending'").fetchone()[0]
            reviewed = conn.execute("SELECT COUNT(*) FROM review_queue WHERE status='reviewed'").fetchone()[0]
            lc_count = conn.execute(
                "SELECT COUNT(*) FROM review_queue WHERE uncertainty_type='low_confidence' AND status='pending'"
            ).fetchone()[0]
            # Fixed: lm_count query was incomplete (missing WHERE clause) in original
            lm_count = conn.execute(
                "SELECT COUNT(*) FROM review_queue WHERE uncertainty_type='low_margin' AND status='pending'"
            ).fetchone()[0]
            return {
                "total":          total,
                "pending":        pending,
                "reviewed":       reviewed,
                "low_confidence": lc_count,
                "low_margin":     lm_count,
                "queue_pct_full": round(pending / MAX_QUEUE_SIZE * 100),
            }
    except:
        return {}


def export_queue_csv(out_path: str = None):
    """Export pending review queue to CSV for human labelling spreadsheet."""
    out   = out_path or str(Path(cfg.RESULTS_DIR) / f"review_queue_{date.today()}.csv")
    items = get_review_queue("pending", limit=500)
    if not items:
        print("[al] Queue is empty — nothing to export."); return
    fields = ["id", "filename", "predicted_class", "confidence",
              "uncertainty_type", "true_label", "notes"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(items)
    print(f"[al] ✓ Exported {len(items)} items to {out}")
    print(f"[al] Fill in 'true_label' column in the CSV.")
    print(f"[al] Then run: python active_learner.py --import {out}")


def import_reviewed_csv(csv_path: str):
    """Import reviewed labels from CSV back into the database.
    Copies images to data/dataset_raw/<label>/ automatically.
    """
    import shutil
    updated = 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if not row.get("true_label"): continue
            true_label = row["true_label"].strip().upper()
            if true_label not in cfg.CLASS_NAMES:
                print(f"[al] skip row {row['id']}: '{true_label}' not in CLASS_NAMES")
                continue
            mark_reviewed(int(row["id"]), true_label, row.get("notes", ""))
            src = Path(row["filename"])
            if src.exists():
                dst_dir = Path(cfg.DATASET_RAW) / true_label
                dst_dir.mkdir(exist_ok=True)
                shutil.copy2(src, dst_dir / src.name)
                print(f"[al] ✓ Added {src.name} → {true_label}/")
            updated += 1
    print(f"[al] ✓ Imported {updated} reviewed labels.")
    print(f"[al] Re-run Split_dataset.py + PH3_transfer_train.py to retrain.")


def main():
    parser = argparse.ArgumentParser(description="SENTINEL Active Learner")
    parser.add_argument("--export", action="store_true", help="Export pending queue to CSV")
    parser.add_argument("--import", dest="import_csv", help="Import reviewed CSV back")
    args = parser.parse_args()

    if args.export:
        export_queue_csv()
    elif args.import_csv:
        import_reviewed_csv(args.import_csv)
    else:
        stats = get_queue_stats()
        print("\n[al] ═══ ACTIVE LEARNING QUEUE ═══")
        print(f"  Pending review:   {stats.get('pending', 0)}")
        print(f"  Already reviewed: {stats.get('reviewed', 0)}")
        print(f"  Low confidence:   {stats.get('low_confidence', 0)}")
        print(f"  Low margin:       {stats.get('low_margin', 0)}")
        print(f"  Queue % full:     {stats.get('queue_pct_full', 0)}%")
        print(f"\n  Run: python active_learner.py --export")
        print(f"  Fill true_label column in CSV")
        print(f"  Run: python active_learner.py --import results/review_queue_DATE.csv")
        print(f"  Then: python Split_dataset.py + python PH3_transfer_train.py")

if __name__ == "__main__":
    main()