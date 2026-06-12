import pandas as pd
from datetime import datetime
import os

# Current analytics result
vehicle_counts = {
    "Car": 6,
    "Motorcycle": 0,
    "Bus": 3,
    "Truck": 0
}

total_vehicles = sum(vehicle_counts.values())

# Create history record
record = {
    "Timestamp": datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    ),
    "Car": vehicle_counts["Car"],
    "Motorcycle": vehicle_counts["Motorcycle"],
    "Bus": vehicle_counts["Bus"],
    "Truck": vehicle_counts["Truck"],
    "Total": total_vehicles
}

history_file = (
    "outputs/history/traffic_history.csv"
)

# Append history
if os.path.exists(history_file):

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

# Save history
history_df.to_csv(
    history_file,
    index=False
)

print(
    "\nHistorical data saved!"
)

print(history_df.tail())