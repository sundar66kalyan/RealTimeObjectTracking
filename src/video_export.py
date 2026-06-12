import cv2
import os
from ultralytics import YOLO

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("data/traffic1.mp4")

# Output settings
frame_width = 640
frame_height = 360
fps = 30

output_path = "outputs/traffic_analytics_output.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (frame_width, frame_height)
)

vehicle_names = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

counted_ids = set()

vehicle_counts = {
    "Car": 0,
    "Motorcycle": 0,
    "Bus": 0,
    "Truck": 0
}

line_y = 250

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(
        frame,
        (frame_width, frame_height)
    )

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    annotated = results[0].plot()

    cv2.line(
        annotated,
        (0, line_y),
        (frame_width, line_y),
        (0, 0, 255),
        2
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

            if cls not in vehicle_names:
                continue

            x1, y1, x2, y2 = box

            center_y = int(
                (y1 + y2) / 2
            )

            if (
                abs(center_y - line_y) < 10
                and track_id not in counted_ids
            ):

                counted_ids.add(track_id)

                vehicle_counts[
                    vehicle_names[cls]
                ] += 1

    # Dashboard Overlay
    y = 30

    for vehicle, count in vehicle_counts.items():

        cv2.putText(
            annotated,
            f"{vehicle}: {count}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        y += 30

    cv2.putText(
        annotated,
        f"Total: {len(counted_ids)}",
        (10, y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    # Save frame
    out.write(annotated)

    cv2.imshow(
        "Traffic Analytics Export",
        annotated
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("\nVideo Export Complete")
print(output_path)