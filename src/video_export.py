"""
video_export.py
---------------
Processes a video through the full analytics pipeline and saves
an annotated output video with:
  - Bounding boxes & track IDs
  - Vehicle classification labels
  - Speed overlays
  - Counting line
  - Live stats panel
  - Timestamp watermark

Usage:
    python src/video_export.py
    python src/video_export.py --source data/highway.mp4 --output outputs/result.mp4
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import cv2
from ultralytics import YOLO

VEHICLE_NAMES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
FRAME_W, FRAME_H = 640, 360
PIXEL_TO_KMH     = 0.05
SPEED_HISTORY    = 5


def export_analytics_video(source: str, output: str, model_path: str, line_ratio: float) -> None:

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    model = YOLO(model_path)
    cap   = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    line_y       = int(FRAME_H * line_ratio)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, src_fps, (FRAME_W, FRAME_H))

    print(f"[INFO]  Source   : {source}")
    print(f"[INFO]  Output   : {output}")
    print(f"[INFO]  Frames   : {total_frames}  |  FPS: {src_fps}")
    print(f"[INFO]  Line Y   : {line_y}px")
    print("[INFO]  Exporting... press Q to stop early.")

    counted_ids:    set[int] = set()
    vehicle_counts: dict     = {v: 0 for v in VEHICLE_NAMES.values()}
    prev_pos:       dict     = {}
    speed_hist:     dict     = defaultdict(list)
    speed_now:      dict     = {}
    frame_idx = 0

    cv2.namedWindow("Exporting  |  Q to stop", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Exporting  |  Q to stop", 1000, 600)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.resize(frame, (FRAME_W, FRAME_H))
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
        cv2.line(annotated, (0, line_y), (FRAME_W, line_y), (0, 0, 255), 2)

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

        # Stats panel
        cv2.rectangle(annotated, (6, 6), (185, 130), (0, 0, 0), -1)
        y_txt = 24
        for vname, cnt in vehicle_counts.items():
            cv2.putText(annotated, f"{vname}: {cnt}", (12, y_txt),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1)
            y_txt += 20
        cv2.putText(annotated, f"Total: {len(counted_ids)}", (12, y_txt + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        # Timestamp watermark
        cv2.putText(
            annotated,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            (FRAME_W - 218, FRAME_H - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1,
        )

        writer.write(annotated)
        cv2.imshow("Exporting  |  Q to stop", annotated)
        frame_idx += 1

        if frame_idx % 30 == 0:
            pct = frame_idx / max(total_frames, 1) * 100
            print(f"\r[INFO]  {pct:.0f}% ({frame_idx}/{total_frames})", end="", flush=True)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO]  Stopped early by user.")
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    size_mb = os.path.getsize(output) / 1_048_576 if os.path.exists(output) else 0
    print(f"\n[DONE]  Saved: {output}  ({size_mb:.1f} MB)")
    print(f"[DONE]  Frames written: {frame_idx}")
    print(f"[DONE]  Total vehicles : {len(counted_ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/traffic1.mp4")
    parser.add_argument("--output", default="outputs/traffic_analytics_output.mp4")
    parser.add_argument("--model",  default="yolov8n.pt")
    parser.add_argument("--line",   type=float, default=0.65)
    args = parser.parse_args()

    export_analytics_video(args.source, args.output, args.model, args.line)