import cv2
import time
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("data/traffic1.mp4")

# Vehicle classes
vehicle_classes = [2, 3, 5, 7]

# Store previous positions and timestamps
previous_positions = {}
previous_times = {}

# Store average speeds
vehicle_speeds = {}

cv2.namedWindow(
    "Speed Estimation",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Speed Estimation",
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

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            current_time = time.time()

            if track_id in previous_positions:

                prev_x, prev_y = previous_positions[track_id]

                distance_pixels = (
                    ((center_x - prev_x) ** 2) +
                    ((center_y - prev_y) ** 2)
                ) ** 0.5

                elapsed_time = (
                    current_time -
                    previous_times[track_id]
                )

                if elapsed_time > 0:

                    pixel_speed = (
                        distance_pixels /
                        elapsed_time
                    )

                    # Approximate conversion
                    speed_kmh = pixel_speed * 0.05

                    vehicle_speeds[
                        track_id
                    ] = speed_kmh

                    cv2.putText(
                        annotated,
                        f"{speed_kmh:.1f} km/h",
                        (
                            center_x,
                            center_y - 15
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2
                    )

            previous_positions[track_id] = (
                center_x,
                center_y
            )

            previous_times[track_id] = (
                current_time
            )

    cv2.imshow(
        "Speed Estimation",
        annotated
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nVehicle Speed Summary")
print("-" * 35)

for vehicle_id, speed in vehicle_speeds.items():

    print(
        f"Vehicle ID {vehicle_id} : "
        f"{speed:.1f} km/h"
    )