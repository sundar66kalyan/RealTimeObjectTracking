"""
track.py
--------
Real-time vehicle detection and tracking using YOLOv8 + ByteTrack.
Press Q to quit.

Usage:
    python src/track.py
    python src/track.py --source data/highway.mp4
    python src/track.py --source data/highway.mp4 --model yolov8s.pt
"""

import argparse
import sys
import cv2
from ultralytics import YOLO

# ── Constants ──────────────────────────────────────────────────────────────────

VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
WINDOW_NAME = "YOLOv8 + ByteTrack  |  Q to quit"
FRAME_SIZE  = (640, 360)


# ── Main ───────────────────────────────────────────────────────────────────────

def run_tracking(source: str, model_path: str) -> None:

    model = YOLO(model_path)
    cap   = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {source}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO]  Source : {source}")
    print(f"[INFO]  Frames : {total_frames}  |  FPS: {fps:.1f}")
    print(f"[INFO]  Model  : {model_path}")
    print("[INFO]  Starting tracking ... press Q to stop.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1000, 600)

    active_ids: set[int] = set()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, FRAME_SIZE)

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            classes=list(VEHICLE_CLASSES.keys()),
        )

        annotated = results[0].plot()

        # Track active vehicle IDs
        if results[0].boxes.id is not None:
            ids     = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            for tid, cls in zip(ids, classes):
                if cls in VEHICLE_CLASSES:
                    active_ids.add(tid)

        # Overlay info
        cv2.putText(
            annotated,
            f"Tracked: {len(active_ids)}  |  Frame: {frame_idx}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
        )

        cv2.imshow(WINDOW_NAME, annotated)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO]  Stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[DONE]  Frames processed : {frame_idx}")
    print(f"[DONE]  Unique IDs tracked: {len(active_ids)}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 + ByteTrack real-time tracking")
    parser.add_argument("--source", default="data/traffic1.mp4", help="Video file path")
    parser.add_argument("--model",  default="yolov8n.pt",        help="YOLOv8 model weights")
    args = parser.parse_args()

    run_tracking(args.source, args.model)