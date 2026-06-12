"""
alert_system.py
---------------
Reads the latest history entry and generates a traffic alert.
Alert levels:
  HIGH   — ≥ 20 vehicles
  MEDIUM — ≥ 10 vehicles
  LOW    — < 10 vehicles

Usage:
    python src/alert_system.py
    python src/alert_system.py --total 25
"""

import argparse
import os
from datetime import datetime

import pandas as pd


HISTORY_FILE = "outputs/history/traffic_history.csv"
ALERT_FILE   = "outputs/traffic_alerts.csv"

THRESHOLDS = {
    "HIGH":   20,
    "MEDIUM": 10,
}

COLORS = {
    "HIGH":   "\033[91m",   # Red
    "MEDIUM": "\033[93m",   # Yellow
    "LOW":    "\033[92m",   # Green
    "RESET":  "\033[0m",
}


def get_alert_level(total: int) -> str:
    if total >= THRESHOLDS["HIGH"]:
        return "HIGH"
    elif total >= THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def generate_alert(total_vehicles: int) -> dict:
    level = get_alert_level(total_vehicles)

    record = {
        "Timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Total Vehicles": int(total_vehicles),
        "Alert":          level,
        "Threshold HIGH": THRESHOLDS["HIGH"],
        "Threshold MED":  THRESHOLDS["MEDIUM"],
    }

    os.makedirs(os.path.dirname(ALERT_FILE), exist_ok=True)

    if os.path.exists(ALERT_FILE):
        df = pd.read_csv(ALERT_FILE)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    df.to_csv(ALERT_FILE, index=False)

    color = COLORS[level]
    reset = COLORS["RESET"]
    print(f"\n{color}{'='*36}{reset}")
    print(f"{color}  TRAFFIC ALERT GENERATED{reset}")
    print(f"{color}{'='*36}{reset}")
    print(f"  Timestamp     : {record['Timestamp']}")
    print(f"  Total vehicles: {record['Total Vehicles']}")
    print(f"{color}  Status        : {level} TRAFFIC{reset}")
    print(f"{'='*36}")
    print(f"[OK]  Saved to: {ALERT_FILE}  (row {len(df)})")

    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=None,
                        help="Override total — otherwise read from history CSV")
    args = parser.parse_args()

    if args.total is not None:
        total = args.total
    elif os.path.exists(HISTORY_FILE):
        df    = pd.read_csv(HISTORY_FILE)
        total = int(df.iloc[-1]["Total"])
        print(f"[INFO]  Read latest history: {total} vehicles")
    else:
        print("[ERROR] No history file found. Run vehicle_analytics.py first,")
        print("        or pass --total <number>.")
        raise SystemExit(1)

    generate_alert(total)