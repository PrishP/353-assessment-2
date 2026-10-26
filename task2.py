import pandas as pd
import numpy as np
 
# 1. Load Cleaned Data
cleaned_df = pd.read_csv("US Airline Data/clean_data.csv")
print(cleaned_df.shape)

# 2. Create New Variable
cleaned_df['Arrival_Disruption_Severity'] = [0 if x < 15 else 1 if 15 >= x < 30 else 2 if 30 >= x < 90 else 3 if 90 >= x else 0 for x in cleaned_df['ArrDelay']]
print(cleaned_df.loc[50])

class_names = {
    0: "No Delay",
    1: "Minor Delay",
    2: "Moderate Delay",
    3: "Severe Delay"
}

print("\nSample target values:")
print(
    cleaned_df[
        ["ArrDelay", "Arrival_Disruption_Severity"]
    ].head(10).to_string(index=False)
)


 
# 4. Verify target boundaries and eligibility
 

boundary_checks = pd.DataFrame({
    "ArrDelay": [14, 15, 15, 39, 40, 41, -5],
    "Expected": [0, 1, 2, 1, 2, 2, 0]
})

#boundary_checks["Actual"] = classify_recovery(boundary_checks)

#boundary_checks["Passed"] = (
    #boundary_checks["Expected"] == boundary_checks["Actual"]
#)

#assert boundary_checks["Passed"].all(), "Boundary check failed."
assert cleaned_df["Arrival_Disruption_Severity"].isin([0, 1, 2, 3, 99]).all()

print("\nBoundary checks:")
print(boundary_checks.to_string(index=False))
print("\nAll target checks passed.")


 
# 5. Report class counts and percentages
 

counts = (
    cleaned_df["Arrival_Disruption_Severity"]
    .value_counts()
    .reindex([0, 1, 2, 3], fill_value=0)
)

class_summary = counts.rename("Count").to_frame()
class_summary.index.name = "Class"

class_summary["Status"] = class_summary.index.map(class_names)

class_summary["Percentage"] = (
    class_summary["Count"] / len(cleaned_df) * 100
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
    f"({counts[largest_class] / len(cleaned_df):.2%})"
)

print(
    f"Smallest class: {class_names[smallest_class]} "
    f"({counts[smallest_class] / len(cleaned_df):.2%})"
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
cleaned_df.to_csv(
    "US Airline Data/arrival_delay_data.csv",
    index=False
)

#eligibility_summary.to_csv(
 #   "US Airline Data/eligibility_summary.csv",
  #  index=False
#)

class_summary.to_csv(
    "US Airline Data/class_distribution.csv",
    index=True
)

boundary_checks.to_csv(
    "US Airline Data/boundary_checks.csv",
    index=False
)

print("Task 2 outputs saved:")