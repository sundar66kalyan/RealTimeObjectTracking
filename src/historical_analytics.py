"""
historical_analytics.py
------------------------
Appends a new analytics record to the rolling history CSV.
Provides a summary of trends (peak hour, busiest type, total sessions).

Usage:
    python src/historical_analytics.py
    python src/historical_analytics.py --counts '{"Car":6,"Bus":3,"Motorcycle":0,"Truck":0}'
"""

import argparse
import json
import os
from datetime import datetime

import pandas as pd


HISTORY_FILE = "outputs/history/traffic_history.csv"


def append_history(vehicle_counts: dict) -> pd.DataFrame:
    """Append a new session record to traffic_history.csv and return full history."""

    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

    total = sum(vehicle_counts.values())

    record = {
        "Timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Car":        vehicle_counts.get("Car", 0),
        "Motorcycle": vehicle_counts.get("Motorcycle", 0),
        "Bus":        vehicle_counts.get("Bus", 0),
        "Truck":      vehicle_counts.get("Truck", 0),
        "Total":      total,
    }

    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    df.to_csv(HISTORY_FILE, index=False)
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print trend summary from historical data."""

    print("\n" + "=" * 44)
    print("   HISTORICAL ANALYTICS SUMMARY")
    print("=" * 44)
    print(f"  Sessions recorded : {len(df)}")
    print(f"  Total vehicles    : {df['Total'].sum()}")
    print(f"  Avg per session   : {df['Total'].mean():.1f}")
    print(f"  Peak session      : {df['Total'].max()}  "
          f"({df.loc[df['Total'].idxmax(), 'Timestamp']})")
    print("-" * 44)
    for col in ["Car", "Motorcycle", "Bus", "Truck"]:
        print(f"  {col:<14}: total={df[col].sum():<6}  avg={df[col].mean():.1f}")
    print("=" * 44)
    print(f"\n  History file: {HISTORY_FILE}")
    print("\n  Last 5 records:")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--counts",
        default='{"Car": 6, "Motorcycle": 0, "Bus": 3, "Truck": 0}',
    )
    args = parser.parse_args()

    counts = json.loads(args.counts)
    df     = append_history(counts)
    print_summary(df)