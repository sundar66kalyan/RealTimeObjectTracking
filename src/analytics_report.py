import pandas as pd

vehicle_counts = {
    "Car": 6,
    "Motorcycle": 0,
    "Bus": 3,
    "Truck": 0
}

df = pd.DataFrame(
    list(vehicle_counts.items()),
    columns=["Vehicle Type", "Count"]
)

df.to_csv(
    "outputs/vehicle_report.csv",
    index=False
)

print(df)

print(
    "\nReport saved to outputs/vehicle_report.csv"
)