from pathlib import Path
import pandas as pd

# 1. Locate input and output files
project_folder = Path(__file__).resolve().parent

input_file = (
    project_folder
    / "results"
    / "task2_delay_recovery"
    / "delay_recovery_data.csv"
)

output_folder = (
    project_folder / "results" / "task3_delay_recovery"
)
output_folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file)

# 2. Check that all three months are included
df["FlightDate"] = pd.to_datetime(
    df["FlightDate"], errors="coerce"
)

if df["FlightDate"].isna().any():
    raise ValueError("Invalid FlightDate values. Check Task 1.")

months = df["FlightDate"].dt.to_period("M").astype(str)

print("\nEligible flights by month:")
print(months.value_counts().sort_index())

expected_months = {"2025-01", "2025-02", "2025-12"}

if set(months) != expected_months:
    raise ValueError(
        "Your Task 2 output does not contain all three required "
        "months. Rebuild Task 1 using January, February and "
        "December, then rerun Task 2."
    )

# 3. Create scheduled departure and arrival hour features
# Examples: 0730 -> 7, 1545 -> 15, 2400 -> 0.
# Invalid times become missing values.
def extract_hour(series):
    times = pd.to_numeric(series, errors="coerce")

    valid = (
        (
            times.between(0, 2359)
            & (times % 100 < 60)
            & (times % 1 == 0)
        )
        | (times == 2400)
    )

    return (times.where(valid) // 100) % 24


df["Scheduled_Departure_Hour"] = extract_hour(df["CRSDepTime"])
df["Scheduled_Arrival_Hour"] = extract_hour(df["CRSArrTime"])

# 4. Select the final nine input features
feature_columns = [
    "DepDelay",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "Distance",
    "CRSElapsedTime",
    "DayOfWeek",
    "Scheduled_Departure_Hour",
    "Scheduled_Arrival_Hour"
]

target_column = "Delay_Recovery_Status"

# Verify that these are eligible Model 2 records
df["DepDelay"] = pd.to_numeric(df["DepDelay"], errors="coerce")

if not df["DepDelay"].ge(15).all():
    raise ValueError("Task 2 must contain only valid DepDelay >= 15.")

if not df[target_column].isin([0, 1, 2]).all():
    raise ValueError("Target must contain only classes 0, 1 and 2.")

X = df[feature_columns].copy()
y = df[target_column].copy()

# 5. Inspect the features and target
print("\nInput features:")
print(X.head().to_string(index=False))

print("\nTarget:")
print(y.head().to_string(index=False))

print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nMissing values in features:")
print(X.isna().sum())

# 6. Save features, target and date together
# FlightDate is retained for chronological splitting in Task 4.
# It is NOT included in X.
task3_df = df[
    ["FlightDate"] + feature_columns + [target_column]
].copy()

output_file = output_folder / "delay_recovery_features.csv"
task3_df.to_csv(output_file, index=False)

print("\nSaved:", output_file)