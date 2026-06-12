import pandas as pd

# Vehicle counts from your analytics
vehicle_counts = {
    "Car": 6,
    "Motorcycle": 0,
    "Bus": 3,
    "Truck": 0
}

# Convert to DataFrame
df = pd.DataFrame(
    list(vehicle_counts.items()),
    columns=["Vehicle Type", "Count"]
)

# Add total row
total_count = df["Count"].sum()

total_row = pd.DataFrame(
    [["Total", total_count]],
    columns=["Vehicle Type", "Count"]
)

df = pd.concat(
    [df, total_row],
    ignore_index=True
)

# Save CSV
output_file = "outputs/vehicle_report.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nVehicle Analytics Report")
print(df)

print(
    f"\nCSV file saved successfully: {output_file}"
)