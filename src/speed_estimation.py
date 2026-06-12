"""
speed_estimation.py
-------------------
Estimates vehicle speed from pixel displacement per second.
Uses a smoothed rolling average to reduce jitter.

NOTE: The pixel-to-km/h conversion factor (PIXEL_TO_KMH) is approximate.
      Calibrate it using a known reference distance in your video.

Usage:
    python src/speed_estimation.py
    python src/speed_estimation.py --source data/highway.mp4 --scale 0.05
"""

import argparse
import sys
import time
import cv2
from collections import defaultdict
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
FRAME_SIZE      = (640, 360)
SPEED_HISTORY   = 5   # frames to smooth over


def run_speed_estimation(source: str, model_path: str, pixel_scale: float) -> None:
    """
    Args:
        pixel_scale : Conversion factor — pixels/second → km/h.
                      Tune this to match your camera setup.
    """

    model = YOLO(model_path)
    cap   = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        sys.exit(1)

    prev_pos:   dict[int, tuple[float, float, float]] = {}   # id → (cx, cy, time)
    speed_hist: dict[int, list[float]]                = defaultdict(list)
    speed_now:  dict[int, float]                      = {}

    cv2.namedWindow("Speed Estimation  |  Q to quit", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Speed Estimation  |  Q to quit", 1000, 600)

    print(f"[INFO]  Scale factor: {pixel_scale} (pixels/s → km/h)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame    = cv2.resize(frame, FRAME_SIZE)
        t_now    = time.perf_counter()
        results  = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            classes=list(VEHICLE_CLASSES.keys()),
        )
        annotated = results[0].plot()

        if results[0].boxes.id is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            ids     = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, tid, cls in zip(boxes, ids, classes):
                if cls not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                if tid in prev_pos:
                    px, py, pt = prev_pos[tid]
                    dt = t_now - pt

                    if dt > 0:
                        dist_px = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                        speed   = (dist_px / dt) * pixel_scale

                        # Rolling average to smooth jitter
                        speed_hist[tid].append(speed)
                        if len(speed_hist[tid]) > SPEED_HISTORY:
                            speed_hist[tid].pop(0)
                        smoothed = sum(speed_hist[tid]) / len(speed_hist[tid])
                        speed_now[tid] = round(smoothed, 1)

                prev_pos[tid] = (cx, cy, t_now)

                # Display speed label on bounding box
                if tid in speed_now:
                    spd   = speed_now[tid]
                    color = (0, 255, 255) if spd < 60 else (0, 100, 255)
                    cv2.putText(
                        annotated, f"{spd:.0f} km/h",
                        (int(cx) - 30, int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                    )

        cv2.imshow("Speed Estimation  |  Q to quit", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if speed_now:
        speeds = list(speed_now.values())
        print(f"\n[DONE]  Vehicles tracked : {len(speed_now)}")
        print(f"[DONE]  Avg speed        : {sum(speeds)/len(speeds):.1f} km/h")
        print(f"[DONE]  Max speed        : {max(speeds):.1f} km/h")
        print(f"[DONE]  Min speed        : {min(speeds):.1f} km/h")
        print("\n--- Per Vehicle ---")
        for vid, spd in sorted(speed_now.items()):
            print(f"  Vehicle ID {vid:>3} : {spd:.1f} km/h")
    else:
        print("[INFO]  No speed data collected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/traffic1.mp4")
    parser.add_argument("--model",  default="yolov8n.pt")
    parser.add_argument("--scale",  type=float, default=0.05,
                        help="Pixel/sec to km/h conversion factor")
    args = parser.parse_args()

    run_speed_estimation(args.source, args.model, args.scale)