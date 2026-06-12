from ultralytics import YOLO

print("Downloading YOLOv8 model...")

model = YOLO("yolov8n.pt")

print("YOLOv8 model downloaded successfully!")