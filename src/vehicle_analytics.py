"""
vehicle_analytics.py
--------------------
Full analytics pipeline:
  - YOLOv8 detection + ByteTrack
  - Vehicle counting per class
  - Speed estimation
  - CSV export
  - Historical analytics

Usage:
    python src/vehicle_analytics.py
    python src/vehicle_analytics.py --source data/highway.mp4
"""

import argparse
import os
import sys
import time
from datetime import datetime
from collections import defaultdict

import cv2
import pandas as pd
from ultralytics import YOLO

# ── Constants ──────────────────────────────────────────────────────────────────

VEHICLE_NAMES   = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
FRAME_SIZE      = (640, 360)
PIXEL_TO_KMH    = 0.05
SPEED_HISTORY   = 5


# ── Output setup ───────────────────────────────────────────────────────────────

def ensure_outputs() -> None:
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/history", exist_ok=True)


# ── Export helpers ─────────────────────────────────────────────────────────────

def export_csv(vehicle_counts: dict, total: int) -> str:
    rows = list(vehicle_counts.items()) + [("Total", total)]
    df   = pd.DataFrame(rows, columns=["Vehicle Type", "Count"])
    path = "outputs/vehicle_report.csv"
    df.to_csv(path, index=False)
    return path


def update_history(vehicle_counts: dict, total: int) -> pd.DataFrame:
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **vehicle_counts,
        "Total": total,
    }
    path = "outputs/history/traffic_history.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])
    df.to_csv(path, index=False)
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def run_analytics(source: str, model_path: str, line_ratio: float) -> None:
    ensure_outputs()

    model = YOLO(model_path)
    cap   = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        sys.exit(1)

    frame_h = FRAME_SIZE[1]
    line_y  = int(frame_h * line_ratio)

    counted_ids:    set[int]   = set()
    vehicle_counts: dict       = {v: 0 for v in VEHICLE_NAMES.values()}
    prev_pos:       dict       = {}
    speed_hist:     dict       = defaultdict(list)
    speed_now:      dict       = {}

    cv2.namedWindow("Vehicle Analytics  |  Q to quit", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vehicle Analytics  |  Q to quit", 1000, 600)

    print(f"[INFO]  Source: {source}  |  Line: y={line_y}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.resize(frame, FRAME_SIZE)
        t_now   = time.perf_counter()
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            classes=list(VEHICLE_NAMES.keys()),
        )
        annotated = results[0].plot()

        # Counting line
        cv2.line(annotated, (0, line_y), (FRAME_SIZE[0], line_y), (0, 0, 255), 2)

        if results[0].boxes.id is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            ids     = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, tid, cls in zip(boxes, ids, classes):
                if cls not in VEHICLE_NAMES:
                    continue

                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Count crossing
                if abs(cy - line_y) < 12 and tid not in counted_ids:
                    counted_ids.add(tid)
                    vehicle_counts[VEHICLE_NAMES[cls]] += 1

                # Speed
                if tid in prev_pos:
                    px, py, pt = prev_pos[tid]
                    dt = t_now - pt
                    if dt > 0:
                        dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                        spd  = (dist / dt) * PIXEL_TO_KMH
                        speed_hist[tid].append(spd)
                        if len(speed_hist[tid]) > SPEED_HISTORY:
                            speed_hist[tid].pop(0)
                        speed_now[tid] = round(sum(speed_hist[tid]) / len(speed_hist[tid]), 1)

                        color = (0, 255, 255) if speed_now[tid] < 60 else (0, 100, 255)
                        cv2.putText(
                            annotated, f"{speed_now[tid]:.0f} km/h",
                            (int(cx) - 28, int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                        )
                prev_pos[tid] = (cx, cy, t_now)

        # Stats overlay
        panel_y = 22
        for vname, cnt in vehicle_counts.items():
            cv2.putText(annotated, f"{vname}: {cnt}", (10, panel_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            panel_y += 22

        total = len(counted_ids)
        cv2.putText(annotated, f"TOTAL: {total}", (10, panel_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Vehicle Analytics  |  Q to quit", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    total = len(counted_ids)

    # Summary
    print("\n" + "=" * 40)
    print("  VEHICLE ANALYTICS SUMMARY")
    print("=" * 40)
    for vname, cnt in vehicle_counts.items():
        print(f"  {vname:<12}: {cnt}")
    print("-" * 40)
    print(f"  Total       : {total}")
    if speed_now:
        speeds = list(speed_now.values())
        print(f"  Avg Speed   : {sum(speeds)/len(speeds):.1f} km/h")
        print(f"  Max Speed   : {max(speeds):.1f} km/h")
    print("=" * 40)

    csv_path = export_csv(vehicle_counts, total)
    print(f"\n[OK]  CSV saved   : {csv_path}")

    history_df = update_history(vehicle_counts, total)
    print(f"[OK]  History rows: {len(history_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/traffic1.mp4")
    parser.add_argument("--model",  default="yolov8n.pt")
    parser.add_argument("--line",   type=float, default=0.65)
    args = parser.parse_args()

    run_analytics(args.source, args.model, args.line)