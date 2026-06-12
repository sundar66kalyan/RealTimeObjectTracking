import cv2
import pandas as pd
import os
from datetime import datetime
from ultralytics import YOLO

# ====================================
# CREATE OUTPUT FOLDERS
# ====================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/history", exist_ok=True)

# ====================================
# LOAD MODEL
# ====================================

model = YOLO("yolov8n.pt")

# ====================================
# VIDEO INPUT
# ====================================

cap = cv2.VideoCapture("data/traffic1.mp4")

# ====================================
# VEHICLE CLASSES
# ====================================

vehicle_names = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# ====================================
# COUNTERS
# ====================================

counted_ids = set()

vehicle_counts = {
    "Car": 0,
    "Motorcycle": 0,
    "Bus": 0,
    "Truck": 0
}

# Counting line
line_y = 250

# ====================================
# WINDOW
# ====================================

cv2.namedWindow(
    "Vehicle Analytics",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Vehicle Analytics",
    1000,
    600
)

# ====================================
# MAIN LOOP
# ====================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

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

    # ====================================
    # DISPLAY COUNTS
    # ====================================

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

    cv2.imshow(
        "Vehicle Analytics",
        annotated
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ====================================
# CLEANUP
# ====================================

cap.release()
cv2.destroyAllWindows()

# ====================================
# SUMMARY
# ====================================

print("\nVehicle Summary")
print("-" * 30)

for vehicle, count in vehicle_counts.items():

    print(
        f"{vehicle}: {count}"
    )

print("-" * 30)
print(
    "Total Vehicles:",
    len(counted_ids)
)

# ====================================
# CSV EXPORT
# ====================================

df = pd.DataFrame(
    list(vehicle_counts.items()),
    columns=[
        "Vehicle Type",
        "Count"
    ]
)

total_row = pd.DataFrame(
    [["Total", len(counted_ids)]],
    columns=[
        "Vehicle Type",
        "Count"
    ]
)

df = pd.concat(
    [df, total_row],
    ignore_index=True
)

csv_file = (
    "outputs/vehicle_report.csv"
)

df.to_csv(
    csv_file,
    index=False
)

print("\nCSV Exported")
print(csv_file)

# ====================================
# HISTORICAL ANALYTICS
# ====================================

history_file = (
    "outputs/history/traffic_history.csv"
)

record = {
    "Timestamp": datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    ),
    "Car": vehicle_counts["Car"],
    "Motorcycle": vehicle_counts["Motorcycle"],
    "Bus": vehicle_counts["Bus"],
    "Truck": vehicle_counts["Truck"],
    "Total": len(counted_ids)
}

if os.path.exists(
    history_file
):

    history_df = pd.read_csv(
        history_file
    )

    history_df = pd.concat(
        [
            history_df,
            pd.DataFrame([record])
        ],
        ignore_index=True
    )

else:

    history_df = pd.DataFrame(
        [record]
    )

history_df.to_csv(
    history_file,
    index=False
)

print(
    "\nHistorical Analytics Updated"
)

print(
    f"Records Stored: {len(history_df)}"
)

print(
    f"History File: {history_file}"
)