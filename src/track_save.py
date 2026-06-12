"""
track_save.py
-------------
Runs YOLOv8 + ByteTrack on a video and saves the annotated output.
Preserves original resolution. Press Q to quit early.

Usage:
    python src/track_save.py
    python src/track_save.py --source data/highway.mp4 --output outputs/tracked.mp4
"""

import argparse
import os
import sys
import cv2
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}


def save_tracked_video(source: str, output: str, model_path: str) -> None:

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    model = YOLO(model_path)
    cap   = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {source}")
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

    print(f"[INFO]  Source    : {source}")
    print(f"[INFO]  Output    : {output}")
    print(f"[INFO]  Resolution: {width}x{height}  |  FPS: {fps}  |  Frames: {total}")

    cv2.namedWindow("Saving tracked video  |  Q to stop", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Saving tracked video  |  Q to stop", 1000, 600)

    frame_idx  = 0
    unique_ids: set[int] = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results  = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            classes=list(VEHICLE_CLASSES.keys()),
        )
        annotated = results[0].plot()

        if results[0].boxes.id is not None:
            for tid, cls in zip(
                results[0].boxes.id.cpu().numpy().astype(int),
                results[0].boxes.cls.cpu().numpy().astype(int),
            ):
                if cls in VEHICLE_CLASSES:
                    unique_ids.add(tid)

        writer.write(annotated)
        cv2.imshow("Saving tracked video  |  Q to stop", annotated)
        frame_idx += 1

        # Progress every 30 frames
        if frame_idx % 30 == 0:
            pct = frame_idx / max(total, 1) * 100
            print(f"\r[INFO]  Progress: {pct:.1f}%  ({frame_idx}/{total})", end="", flush=True)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO]  Stopped by user.")
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\n[DONE]  Saved : {output}")
    print(f"[DONE]  Frames: {frame_idx}  |  Unique vehicles: {len(unique_ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/traffic1.mp4")
    parser.add_argument("--output", default="outputs/tracked.mp4")
    parser.add_argument("--model",  default="yolov8n.pt")
    args = parser.parse_args()

    save_tracked_video(args.source, args.output, args.model)