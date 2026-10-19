from pathlib import Path

import numpy as np
import pandas as pd


# 1. Locate the input and output folders
project_folder = Path(__file__).resolve().parent

input_file = (
    project_folder / "US Airline Data" / "clean_data.csv"
)

output_folder = (
    project_folder / "results" / "task2_delay_recovery"
)

output_folder.mkdir(parents=True, exist_ok=True)

if not input_file.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {input_file}\n"
        "Run Task 1 first."
    )

print("Reading:", input_file)


# 2. Load the cleaned dataset from Task 1
clean_df = pd.read_csv(input_file, low_memory=False)

# Remove any previously exported index columns.
clean_df = clean_df.loc[
    :, ~clean_df.columns.str.startswith("Unnamed:")
].copy()

required_columns = ["FlightDate", "DepDelay", "ArrDelay"]

missing_columns = [
    column
    for column in required_columns
    if column not in clean_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print(f"\nRows loaded: {len(clean_df):,}")


# 3. Validate dates and confirm all three months are present
clean_df["FlightDate"] = pd.to_datetime(
    clean_df["FlightDate"], errors="coerce"
)

invalid_dates = clean_df["FlightDate"].isna().sum()

if invalid_dates > 0:
    raise ValueError(
        f"Found {invalid_dates:,} invalid flight dates. "
        "Check Task 1 before continuing."
    )

months = (
    clean_df["FlightDate"]
    .dt.to_period("M")
    .astype(str)
)

print("\nCleaned flights by month:")
print(months.value_counts().sort_index())

expected_months = {"2025-01", "2025-02", "2025-12"}

if set(months.unique()) != expected_months:
    raise ValueError(
        "The cleaned dataset must contain January, February "
        "and December 2025. Rebuild Task 1 first."
    )


# 4. Convert and validate delay values
delay_columns = ["DepDelay", "ArrDelay"]

for column in delay_columns:
    clean_df[column] = pd.to_numeric(
        clean_df[column], errors="coerce"
    )

# Infinite values are invalid, just like missing values.
clean_df[delay_columns] = (
    clean_df[delay_columns]
    .replace([np.inf, -np.inf], np.nan)
)

valid_df = clean_df.dropna(
    subset=delay_columns
).copy()


# 5. Select flights eligible for delay recovery analysis
# Task 1 has already excluded cancelled and diverted flights.
# Model 2 includes only departures delayed at least 15 minutes.
recovery_df = valid_df.loc[
    valid_df["DepDelay"] >= 15
].copy()

eligibility_summary = pd.DataFrame({
    "Step": [
        "Loaded cleaned dataset",
        "Excluded: missing or invalid delay values",
        "Excluded: departure delay below 15 minutes",
        "Eligible flights"
    ],
    "Rows": [
        len(clean_df),
        len(clean_df) - len(valid_df),
        int(valid_df["DepDelay"].lt(15).sum()),
        len(recovery_df)
    ]
})

print("\nEligibility summary:")
print(eligibility_summary.to_string(index=False))

if recovery_df.empty:
    raise ValueError(
        "No eligible flights found. Check the input data."
    )


# 6. Create the target variable: Delay_Recovery_Status
# Apply this function only to eligible flights.
def classify_recovery(frame):
    return np.select(
        [
            # Class 0: arrival delay is below 15 minutes.
            frame["ArrDelay"] < 15,

            # Class 1: still delayed, but less than at departure.
            (
                (frame["ArrDelay"] >= 15)
                & (frame["ArrDelay"] < frame["DepDelay"])
            )
        ],
        [0, 1],

        # Class 2: arrival delay equals or exceeds departure delay.
        default=2
    ).astype(int)


recovery_df["Delay_Recovery_Status"] = classify_recovery(
    recovery_df
)

class_names = {
    0: "Recovered",
    1: "Partially Recovered",
    2: "Not Recovered or Worsened"
}

print("\nSample target values:")
print(
    recovery_df[
        ["DepDelay", "ArrDelay", "Delay_Recovery_Status"]
    ].head(10).to_string(index=False)
)


# 7. Verify classification boundaries and eligibility
boundary_checks = pd.DataFrame({
    "DepDelay": [15, 20, 15, 40, 40, 40, 40],
    "ArrDelay": [14, 15, 15, 39, 40, 41, -5],
    "Expected": [0, 1, 2, 1, 2, 2, 0]
})

boundary_checks["Actual"] = classify_recovery(
    boundary_checks
)

boundary_checks["Passed"] = (
    boundary_checks["Expected"]
    == boundary_checks["Actual"]
)

assert boundary_checks["Passed"].all(), (
    "Target boundary check failed."
)

assert recovery_df["DepDelay"].ge(15).all(), (
    "An ineligible departure was included."
)

assert np.isfinite(
    recovery_df[delay_columns].to_numpy()
).all(), "Invalid delay values remain."

assert recovery_df["Delay_Recovery_Status"].isin(
    [0, 1, 2]
).all(), "Unexpected target class."

assert len(clean_df) == (
    (len(clean_df) - len(valid_df))
    + int(valid_df["DepDelay"].lt(15).sum())
    + len(recovery_df)
), "Eligibility row counts do not reconcile."

print("\nBoundary checks:")
print(boundary_checks.to_string(index=False))

print("\nAll target checks passed.")


# 8. Report class counts and percentages
counts = (
    recovery_df["Delay_Recovery_Status"]
    .value_counts()
    .reindex([0, 1, 2], fill_value=0)
)

class_summary = counts.rename("Count").to_frame()
class_summary.index.name = "Class"

class_summary["Status"] = (
    class_summary.index.map(class_names)
)

class_summary["Percentage"] = (
    class_summary["Count"] / len(recovery_df) * 100
).round(2)

class_summary = class_summary[
    ["Status", "Count", "Percentage"]
]

print("\nDelay Recovery Class Distribution:")
print(class_summary.to_string())


# 9. Summarise class imbalance
largest_class = counts.idxmax()
smallest_class = counts.idxmin()

print(
    f"\nLargest class: {class_names[largest_class]} "
    f"({counts[largest_class] / len(recovery_df):.2%})"
)

print(
    f"Smallest class: {class_names[smallest_class]} "
    f"({counts[smallest_class] / len(recovery_df):.2%})"
)

if counts.min() > 0:
    imbalance_ratio = counts.max() / counts.min()

    print(
        "Largest-to-smallest class ratio: "
        f"{imbalance_ratio:.2f}:1"
    )
else:
    print("At least one target class has no observations.")

print(
    "\nEvaluation implication: accuracy alone may hide poor "
    "performance on less common classes. In Task 4, include "
    "macro F1, per-class recall and confusion matrices."
)


# 10. Report eligible flights and target counts by month
recovery_months = (
    recovery_df["FlightDate"]
    .dt.to_period("M")
    .astype(str)
)

monthly_counts = (
    recovery_months
    .value_counts()
    .sort_index()
    .rename_axis("Month")
    .rename("Eligible_Flights")
)

monthly_class_counts = pd.crosstab(
    recovery_months,
    recovery_df["Delay_Recovery_Status"]
).reindex(columns=[0, 1, 2], fill_value=0)

monthly_class_counts.index.name = "Month"

print("\nEligible flights by month:")
print(monthly_counts.to_string())

print("\nRecovery classes by month:")
print(monthly_class_counts.to_string())


# 11. Save the rebuilt Task 2 dataset and evidence
output_file = output_folder / "delay_recovery_data.csv"

recovery_df.to_csv(output_file, index=False)

eligibility_summary.to_csv(
    output_folder / "eligibility_summary.csv",
    index=False
)

class_summary.to_csv(
    output_folder / "class_distribution.csv",
    index=True
)

boundary_checks.to_csv(
    output_folder / "boundary_checks.csv",
    index=False
)

monthly_counts.to_csv(
    output_folder / "eligible_flights_by_month.csv"
)

monthly_class_counts.to_csv(
    output_folder / "class_counts_by_month.csv"
)


# 12. Read back the saved file to verify its row counts
saved_df = pd.read_csv(
    output_file,
    usecols=["FlightDate", "Delay_Recovery_Status"]
)

saved_dates = pd.to_datetime(
    saved_df["FlightDate"], errors="coerce"
)

assert saved_dates.notna().all(), (
    "Invalid dates in the saved file."
)

assert len(saved_df) == len(recovery_df), (
    "Saved row count does not match the prepared dataset."
)

assert np.array_equal(
    saved_df["Delay_Recovery_Status"].to_numpy(),
    recovery_df["Delay_Recovery_Status"].to_numpy()
), "Saved target values do not match."

saved_monthly_counts = (
    saved_dates.dt.to_period("M")
    .astype(str)
    .value_counts()
    .sort_index()
    .rename_axis("Month")
    .rename("Eligible_Flights")
)

pd.testing.assert_series_equal(
    saved_monthly_counts,
    monthly_counts
)

print("\nVerified saved Task 2 flights by month:")
print(saved_monthly_counts.to_string())

print(f"\nTotal saved recovery flights: {len(saved_df):,}")
print("\nTask 2 completed successfully.")
print("Saved:", output_file)