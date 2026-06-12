import pandas as pd
import os
from datetime import datetime

# Read latest history
history_file = "outputs/history/traffic_history.csv"

if not os.path.exists(history_file):
    print("History file not found.")
    exit()

history_df = pd.read_csv(history_file)

latest = history_df.iloc[-1]

total_vehicles = latest["Total"]

# Alert thresholds
if total_vehicles >= 20:
    alert_level = "HIGH TRAFFIC"
elif total_vehicles >= 10:
    alert_level = "MEDIUM TRAFFIC"
else:
    alert_level = "LOW TRAFFIC"

alert_record = {
    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Total Vehicles": total_vehicles,
    "Alert": alert_level
}

alert_file = "outputs/traffic_alerts.csv"

if os.path.exists(alert_file):

    alert_df = pd.read_csv(alert_file)

    alert_df = pd.concat(
        [alert_df, pd.DataFrame([alert_record])],
        ignore_index=True
    )

else:

    alert_df = pd.DataFrame([alert_record])

alert_df.to_csv(
    alert_file,
    index=False
)

print("\nTraffic Alert Generated")
print("-" * 30)
print("Vehicles :", total_vehicles)
print("Status   :", alert_level)
print("-" * 30)