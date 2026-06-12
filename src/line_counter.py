"""
line_counter.py
---------------
Counts vehicles crossing a virtual line, tracking direction.
Vehicles moving DOWN  → Entry count
Vehicles moving UP    → Exit count

Usage:
    python src/line_counter.py
    python src/line_counter.py --source data/highway.mp4 --line 0.60
"""

import argparse
import sys
import cv2
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
FRAME_SIZE      = (640, 360)


def run_line_counter(source: str, model_path: str, line_ratio: float) -> None:

    model  = YOLO(model_path)
    cap    = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        sys.exit(1)

    frame_h   = FRAME_SIZE[1]
    line_y    = int(frame_h * line_ratio)
    tolerance = 12

    # Track each ID's previous y-position to determine direction
    prev_y:    dict[int, float] = {}
    entry_ids: set[int]         = set()
    exit_ids:  set[int]         = set()

    cv2.namedWindow("Line Counter  |  Q to quit", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Line Counter  |  Q to quit", 1000, 600)

    print(f"[INFO]  Line Y : {line_y}px  |  Down = Entry, Up = Exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.resize(frame, FRAME_SIZE)
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            classes=list(VEHICLE_CLASSES.keys()),
        )
        annotated = results[0].plot()

        # Counting line
        cv2.line(annotated, (0, line_y), (FRAME_SIZE[0], line_y), (0, 0, 255), 2)

        if results[0].boxes.id is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            ids     = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, tid, cls in zip(boxes, ids, classes):
                if cls not in VEHICLE_CLASSES:
                    continue

                _, y1, _, y2 = box
                cy = (y1 + y2) / 2

                if abs(cy - line_y) < tolerance:
                    if tid in prev_y:
                        if prev_y[tid] < line_y and cy >= line_y:
                            # Moving down → Entry
                            entry_ids.add(tid)
                        elif prev_y[tid] > line_y and cy <= line_y:
                            # Moving up → Exit
                            exit_ids.add(tid)

                prev_y[tid] = cy

        # Overlay panel
        entry = len(entry_ids)
        exit_ = len(exit_ids)

        cv2.rectangle(annotated, (8, 8), (200, 100), (0, 0, 0), -1)
        cv2.putText(annotated, f"Entry : {entry}", (14, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(annotated, f"Exit  : {exit_}", (14, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 100, 255), 2)
        cv2.putText(annotated, f"Net   : {entry - exit_}", (14, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        cv2.imshow("Line Counter  |  Q to quit", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[DONE]  Entry count : {len(entry_ids)}")
    print(f"[DONE]  Exit count  : {len(exit_ids)}")
    print(f"[DONE]  Net traffic : {len(entry_ids) - len(exit_ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/traffic1.mp4")
    parser.add_argument("--model",  default="yolov8n.pt")
    parser.add_argument("--line",   type=float, default=0.65)
    args = parser.parse_args()

    run_line_counter(args.source, args.model, args.line)