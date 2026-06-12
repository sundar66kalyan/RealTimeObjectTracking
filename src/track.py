import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("data/traffic1.mp4")

cv2.namedWindow(
    "YOLOv8 Tracking",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "YOLOv8 Tracking",
    1000,
    600
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize 4K video for CPU processing
    frame = cv2.resize(
        frame,
        (640, 360)
    )

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    annotated = results[0].plot()

    cv2.imshow(
        "YOLOv8 Tracking",
        annotated
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()