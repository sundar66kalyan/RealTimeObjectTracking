import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("data/traffic1.mp4")

# Vehicle classes in COCO
vehicle_classes = [2, 3, 5, 7]

# Store counted IDs
counted_ids = set()

# Counting line position
line_y = 250

cv2.namedWindow(
    "Vehicle Counter",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Vehicle Counter",
    1000,
    600
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize for CPU performance
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

    # Draw counting line
    cv2.line(
        annotated,
        (0, line_y),
        (640, line_y),
        (0, 0, 255),
        3
    )

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()

        ids = results[0].boxes.id.cpu().numpy().astype(int)

        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(
            boxes,
            ids,
            classes
        ):

            if cls not in vehicle_classes:
                continue

            x1, y1, x2, y2 = box

            center_y = int((y1 + y2) / 2)

            if abs(center_y - line_y) < 10:

                counted_ids.add(track_id)

    total_count = len(counted_ids)

    cv2.putText(
        annotated,
        f"Vehicle Count: {total_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Vehicle Counter",
        annotated
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(
    f"Total Vehicles Counted: {len(counted_ids)}"
)