"""
vehicle_count.py
----------------
Counts total unique vehicles using ByteTrack IDs.
Uses a virtual counting line — only counts each vehicle once as it crosses.

Usage:
    python src/vehicle_count.py
    python src/vehicle_count.py --source data/highway.mp4 --line 0.60
"""

import argparse
import sys
import cv2
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
FRAME_SIZE      = (640, 360)


def count_vehicles(source: str, model_path: str, line_ratio: float) -> int:
    """
    Count vehicles crossing a virtual horizontal line.

    Args:
        source     : Video file path.
        model_path : YOLOv8 weights file.
        line_ratio : Vertical position of counting line as fraction of frame height (0–1).
    Returns:
        Total unique vehicles counted.
    """

    model = YOLO(model_path)
    cap   = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {source}")
        sys.exit(1)

    frame_h = FRAME_SIZE[1]
    line_y  = int(frame_h * line_ratio)
    tolerance = 12  # pixels either side of the line

    counted_ids: set[int] = set()

    cv2.namedWindow("Vehicle Counting  |  Q to quit", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vehicle Counting  |  Q to quit", 1000, 600)

    print(f"[INFO]  Source     : {source}")
    print(f"[INFO]  Line Y     : {line_y}px (ratio {line_ratio})")
    print(f"[INFO]  Tolerance  : ±{tolerance}px")

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

        # Draw counting line
        cv2.line(annotated, (0, line_y), (FRAME_SIZE[0], line_y), (0, 0, 255), 2)
        cv2.putText(
            annotated, "COUNTING LINE",
            (FRAME_SIZE[0] // 2 - 70, line_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
        )

        if results[0].boxes.id is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            ids     = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, tid, cls in zip(boxes, ids, classes):
                if cls not in VEHICLE_CLASSES:
                    continue
                _, y1, _, y2 = box
                center_y = int((y1 + y2) / 2)
                if abs(center_y - line_y) < tolerance and tid not in counted_ids:
                    counted_ids.add(tid)

        # Overlay total count
        count = len(counted_ids)
        cv2.rectangle(annotated, (8, 8), (220, 45), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            f"Vehicles crossed: {count}",
            (14, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

        cv2.imshow("Vehicle Counting  |  Q to quit", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[DONE]  Total Vehicles Counted: {len(counted_ids)}")
    return len(counted_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/traffic1.mp4")
    parser.add_argument("--model",  default="yolov8n.pt")
    parser.add_argument("--line",   type=float, default=0.65,
                        help="Counting line vertical position (0.0–1.0)")
    args = parser.parse_args()

    count_vehicles(args.source, args.model, args.line)