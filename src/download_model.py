"""
download_model.py
-----------------
Downloads YOLOv8 model weights and verifies the installation.
Run once before executing any other script.

Usage:
    python src/download_model.py
"""

import os
import sys


def download_model(model_name: str = "yolov8n.pt") -> None:
    """Download YOLOv8 model if not already present."""

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.")
        print("        Run: pip install ultralytics")
        sys.exit(1)

    if os.path.exists(model_name):
        print(f"[INFO]  Model already exists: {model_name}")
    else:
        print(f"[INFO]  Downloading {model_name} ...")

    model = YOLO(model_name)

    print(f"[OK]    Model ready: {model_name}")
    print(f"[INFO]  Task type  : {model.task}")
    print(f"[INFO]  Input size : 640 x 640 (default)")


if __name__ == "__main__":
    download_model()