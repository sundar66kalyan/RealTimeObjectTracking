"""
csv_export.py
-------------
Reads the latest vehicle_report.csv and generates a formatted
analytics report saved to outputs/analytics_report.txt.

Previously: analytics_report.py and csv_export.py were separate files
            doing nearly identical work. Now combined into one clean module.

Usage:
    python src/csv_export.py
    python src/csv_export.py --counts '{"Car":6,"Bus":3,"Motorcycle":0,"Truck":0}'
"""

import argparse
import json
import os
from datetime import datetime

import pandas as pd


def export_csv(vehicle_counts: dict, output_path: str = "outputs/vehicle_report.csv") -> pd.DataFrame:
    """Save vehicle counts dict to a CSV with a Total row."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = list(vehicle_counts.items())
    total = sum(vehicle_counts.values())
    rows.append(("Total", total))

    df = pd.DataFrame(rows, columns=["Vehicle Type", "Count"])
    df.to_csv(output_path, index=False)

    print(f"[OK]  CSV saved: {output_path}")
    return df


def generate_text_report(
    vehicle_counts: dict,
    report_path: str = "outputs/analytics_report.txt",
) -> str:
    """Generate a human-readable text report."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    total     = sum(vehicle_counts.values())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 42,
        "   TRAFFIC ANALYTICS REPORT",
        f"   Generated : {timestamp}",
        "=" * 42,
        "",
    ]
    for vtype, cnt in vehicle_counts.items():
        lines.append(f"  {vtype:<14}: {cnt}")
    lines += [
        "-" * 42,
        f"  {'Total':<14}: {total}",
        "=" * 42,
    ]

    report = "\n".join(lines)

    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\n[OK]  Report saved: {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--counts",
        default='{"Car": 6, "Motorcycle": 0, "Bus": 3, "Truck": 0}',
        help="JSON string of vehicle counts",
    )
    args = parser.parse_args()

    counts = json.loads(args.counts)
    export_csv(counts)
    generate_text_report(counts)