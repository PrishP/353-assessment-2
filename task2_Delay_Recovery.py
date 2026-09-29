import pandas as pd
import numpy as np
from pathlib import Path

 
# 1. Load the cleaned dataset produced by Task 1
 

input_file = Path("US Airline Data/clean_data.csv")
output_folder = Path("results/task2_delay_recovery")
output_folder.mkdir(parents=True, exist_ok=True)

clean_df = pd.read_csv(input_file)

# Remove an exported index column if Task 1 saved one.
clean_df = clean_df.loc[
    :, ~clean_df.columns.str.startswith("Unnamed:")
].copy()

required_columns = ["DepDelay", "ArrDelay"]

missing_columns = [
    col for col in required_columns
    if col not in clean_df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print("Rows loaded from cleaned dataset:", len(clean_df))


 
# 2. Select flights eligible for recovery analysis
 

for col in required_columns:
    clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

# Treat infinite values as invalid delay values.
clean_df[required_columns] = clean_df[required_columns].replace(
    [np.inf, -np.inf], np.nan
)

valid_df = clean_df.dropna(subset=required_columns).copy()

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
        int((valid_df["DepDelay"] < 15).sum()),
        len(recovery_df)
    ]
})

print("\nEligibility summary:")
print(eligibility_summary.to_string(index=False))

if recovery_df.empty:
    raise ValueError("No eligible flights found. Check the input data.")


 
# 3. Construct Delay_Recovery_Status
 

# This function is applied only to eligible flights:
# valid delay values and DepDelay >= 15.
def classify_recovery(frame):
    return np.select(
        [
            frame["ArrDelay"] < 15,
            (
                (frame["ArrDelay"] >= 15)
                & (frame["ArrDelay"] < frame["DepDelay"])
            )
        ],
        [0, 1],
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


 
# 4. Verify target boundaries and eligibility
 

boundary_checks = pd.DataFrame({
    "DepDelay": [15, 20, 15, 40, 40, 40, 40],
    "ArrDelay": [14, 15, 15, 39, 40, 41, -5],
    "Expected": [0, 1, 2, 1, 2, 2, 0]
})

boundary_checks["Actual"] = classify_recovery(boundary_checks)

boundary_checks["Passed"] = (
    boundary_checks["Expected"] == boundary_checks["Actual"]
)

assert boundary_checks["Passed"].all(), "Boundary check failed."
assert recovery_df["DepDelay"].ge(15).all()
assert recovery_df[required_columns].notna().all().all()
assert recovery_df["Delay_Recovery_Status"].isin([0, 1, 2]).all()

print("\nBoundary checks:")
print(boundary_checks.to_string(index=False))
print("\nAll target checks passed.")


 
# 5. Report class counts and percentages
 

counts = (
    recovery_df["Delay_Recovery_Status"]
    .value_counts()
    .reindex([0, 1, 2], fill_value=0)
)

class_summary = counts.rename("Count").to_frame()
class_summary.index.name = "Class"

class_summary["Status"] = class_summary.index.map(class_names)

class_summary["Percentage"] = (
    class_summary["Count"] / len(recovery_df) * 100
).round(2)

class_summary = class_summary[
    ["Status", "Count", "Percentage"]
]

print("\nDelay Recovery Class Distribution:")
print(class_summary.to_string())


 
# 6. Show information for the imbalance discussion
 

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
    print(f"Largest-to-smallest class ratio: {imbalance_ratio:.2f}:1")
else:
    print("At least one target class has no observations.")

print(
    "\nEvaluation implication: accuracy alone may hide poor "
    "performance on less common classes. Later evaluation should "
    "include macro F1, per-class recall and a confusion matrix."
)

# 7. Save the Task 2 dataset and evidence
recovery_df.to_csv(
    output_folder / "delay_recovery_data.csv",
    index=False
)

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

print("\nTask 2 outputs saved to:", output_folder.resolve())