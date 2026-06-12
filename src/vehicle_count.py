import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("data/traffic1.mp4")

# Vehicle classes in COCO
vehicle_classes = [2, 3, 5, 7]

# Store counted vehicle IDs
counted_ids = set()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    annotated_frame = results[0].plot()

    if results[0].boxes.id is not None:

        ids = results[0].boxes.id.cpu().numpy().astype(int)

        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for track_id, cls in zip(ids, classes):

            if cls in vehicle_classes:

                counted_ids.add(track_id)

    vehicle_count = len(counted_ids)

    cv2.putText(
        annotated_frame,
        f"Vehicle Count: {vehicle_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Vehicle Counting", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Total Vehicles Counted: {len(counted_ids)}")