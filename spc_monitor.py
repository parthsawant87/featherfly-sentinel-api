# spc_monitor.py — SENTINEL SPC (Statistical Process Control) Monitor
# Algorithm: EWMA (Exponentially Weighted Moving Average) on defect rate
# Triggers alert when defect rate crosses upper control limit (UCL)
#
# Usage:
#   python spc_monitor.py              — run once, print current status
#   python spc_monitor.py --watch      — watch mode, check every 60s
#   from spc_monitor import check_spc  — call from API /spc endpoint

import argparse, time, json
from datetime import datetime, date, timedelta
from pathlib import Path
import config as cfg
from db_logger import get_daily_stats, get_all_predictions

EWMA_LAMBDA    = 0.2   # smoothing factor — 0.2 is standard for fab SPC
UCL_SIGMA      = 3.0   # alert at 3-sigma above baseline (99.7% threshold)
LCL_SIGMA      = 3.0   # lower control limit (unusual cleanliness)
BASELINE_DAYS  = 7     # days of history to establish baseline defect rate
MIN_SAMPLES    = 5     # minimum predictions per day to include
ALERT_LOG_PATH = Path(cfg.LOG_DIR) / "spc_alerts.json"


def get_daily_defect_rates(n_days: int = 30) -> list:
    """Return list of (date_str, defect_rate) for last n_days.
    Only includes days with >= MIN_SAMPLES predictions.
    """
    rates = []
    for i in range(n_days, -1, -1):
        d = str(date.today() - timedelta(days=i))
        s = get_daily_stats(d)
        if s["total"] >= MIN_SAMPLES:
            rates.append((d, s["defect_rate"]))
    return rates


def compute_ewma(rates: list) -> list:
    """Apply EWMA smoothing to list of (date, rate) pairs.
    Returns list of (date, raw_rate, ewma_value).
    """
    if not rates: return []
    result = []
    ewma   = rates[0][1]   # initialise at first observation
    for d, r in rates:
        ewma = EWMA_LAMBDA * r + (1 - EWMA_LAMBDA) * ewma
        result.append((d, r, ewma))
    return result


def compute_control_limits(baseline_rates: list) -> dict:
    """Compute UCL and LCL from baseline period.
    baseline_rates: list of raw defect rate floats (not tuples).
    Returns dict with mean, std, ucl, lcl, ewma_std.
    """
    if not baseline_rates:
        return {"mean": 0.15, "std": 0.05, "ucl": 0.30, "lcl": 0.0, "ewma_std": 0.035}
    mean = sum(baseline_rates) / len(baseline_rates)
    # Population variance — fixed: was using 'var' before defining it
    var  = sum((r - mean) ** 2 for r in baseline_rates) / len(baseline_rates)
    std  = var ** 0.5
    # EWMA control limits are narrower than Shewhart by sqrt(lambda / (2 - lambda))
    ewma_std = std * ((EWMA_LAMBDA / (2 - EWMA_LAMBDA)) ** 0.5)
    return {
        "mean":     round(mean, 4),
        "std":      round(std, 4),
        "ucl":      round(min(1.0, mean + UCL_SIGMA * ewma_std), 4),
        "lcl":      round(max(0.0, mean - LCL_SIGMA * ewma_std), 4),
        "ewma_std": round(ewma_std, 4),
    }


def log_alert(alert: dict):
    """Append alert to the alerts log file. Keeps last 100 alerts."""
    ALERT_LOG_PATH.parent.mkdir(exist_ok=True)
    alerts = []
    if ALERT_LOG_PATH.exists():
        with open(ALERT_LOG_PATH) as f:
            try: alerts = json.load(f)
            except: alerts = []
    alerts.append(alert)
    alerts = alerts[-100:]
    with open(ALERT_LOG_PATH, "w") as f:
        json.dump(alerts, f, indent=2)


def check_spc() -> dict:
    """Run full SPC check. Returns status dict.
    Called by API /spc endpoint and watch loop.

    Returns:
        status:       'IN_CONTROL' | 'UCL_BREACH' | 'LCL_BREACH' | 'INSUFFICIENT_DATA'
        alert:        True if process is out of control
        current_rate: today's defect rate
        ewma_value:   current EWMA statistic
        ucl / lcl:    control limits
        message:      human-readable summary
    """
    all_rates = get_daily_defect_rates(n_days=30)

    if len(all_rates) < 2:   # ← fixed: was 'len(all_rates) 2'
        return {
            "status":         "INSUFFICIENT_DATA",
            "alert":          False,
            "message":        f"Need at least 2 days of data (have {len(all_rates)}). Keep running.",
            "days_available": len(all_rates),
        }

    # Baseline = first BASELINE_DAYS days, current = the rest
    baseline_tuples = all_rates[:BASELINE_DAYS]
    baseline_rates  = [r for _, r in baseline_tuples]
    limits          = compute_control_limits(baseline_rates)
    ewma_data       = compute_ewma(all_rates)
    latest_date, latest_raw, latest_ewma = ewma_data[-1]

    status  = "IN_CONTROL"
    alert   = False
    message = ""

    if latest_ewma > limits["ucl"]:
        status  = "UCL_BREACH"
        alert   = True
        message = (
            f"⚠ DEFECT RATE SPIKE: EWMA={latest_ewma:.3f} > UCL={limits['ucl']:.3f}. "
            f"Process out of control. Baseline rate was {limits['mean']:.3f}. "
            "Inspect recent lots immediately. Check: particle contamination, "
            "resist batch, tool maintenance schedule."
        )
    elif latest_ewma < limits["lcl"]:   # ← fixed: was 'elif latest_ewma limits["lcl"]'
        status  = "LCL_BREACH"
        alert   = False
        message = (
            f"✓ UNUSUALLY CLEAN: EWMA={latest_ewma:.3f} < LCL={limits['lcl']:.3f}. "
            "Check: model confidence threshold drift, or genuinely good process run."
        )
    else:
        message = (
            f"✓ IN CONTROL: EWMA={latest_ewma:.3f}. "
            f"UCL={limits['ucl']:.3f}  Mean={limits['mean']:.3f}  LCL={limits['lcl']:.3f}"
        )

    result = {
        "status":        status,
        "alert":         alert,
        "date":          latest_date,
        "current_rate":  round(latest_raw, 4),
        "ewma_value":    round(latest_ewma, 4),
        "ucl":           limits["ucl"],
        "lcl":           limits["lcl"],
        "baseline_mean": limits["mean"],
        "days_in_chart": len(ewma_data),
        "message":       message,
        "checked_at":    datetime.now().isoformat(),
        "ewma_series":   [
            {"date": d, "rate": round(r, 4), "ewma": round(e, 4)}
            for d, r, e in ewma_data
        ],
    }
    if alert:
        log_alert(result)
    return result


def main():
    parser = argparse.ArgumentParser(description="SENTINEL SPC Monitor")
    parser.add_argument("--watch", action="store_true", help="Watch mode — check every N seconds")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds (default: 60)")
    args = parser.parse_args()

    if args.watch:
        print(f"[spc] Watch mode — checking every {args.interval}s. Ctrl+C to stop.")
        while True:
            r = check_spc()
            print(f"[spc] {r['checked_at'][:19]}  {r['status']:20s}  {r['message'][:80]}")
            time.sleep(args.interval)
    else:
        r = check_spc()
        print(f"\n[spc] ═══ SPC STATUS ═══")
        print(f"  Status:       {r['status']}")
        print(f"  Alert:        {'YES — ACTION REQUIRED' if r['alert'] else 'No'}")
        print(f"  Today rate:   {r.get('current_rate', 'N/A')}")
        print(f"  EWMA:         {r.get('ewma_value', 'N/A')}")
        print(f"  UCL:          {r.get('ucl', 'N/A')}")
        print(f"  Baseline:     {r.get('baseline_mean', 'N/A')}")
        print(f"  LCL:          {r.get('lcl', 'N/A')}")
        print(f"  Days tracked: {r.get('days_in_chart', 0)}")
        print(f"  Message:      {r['message']}")

if __name__ == "__main__":
    main()