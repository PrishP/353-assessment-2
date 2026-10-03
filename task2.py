import pandas as pd
import numpy as np
 
# 1. Load Cleaned Data
cleaned_df = pd.read_csv("US Airline Data/clean_data.csv")
print(cleaned_df.shape)

# 2. Create New Variable
cleaned_df['Arrival_Disruption_Severity'] = [0 if x < 15 else 1 if 15 >= x <30 else 2 if 30 >= x < 90 else 3 if 90 >= x else 99 for x in cleaned_df['ArrDelay']]
print(cleaned_df.loc[50])

class_names = {
    0: "No Delay",
    1: "Minor Delay",
    2: "Severe Delay"
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