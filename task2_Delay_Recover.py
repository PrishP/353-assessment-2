from pathlib import Path
import pandas as pd
import numpy as np

# 1. Load cleaned data from Task 1
folder = Path(__file__).resolve().parent
input_file = folder / 'clean_data.csv'
if not input_file.exists():
    input_file = folder / 'US Airline Data' / 'clean_data.csv'
cleaned_df = pd.read_csv(input_file, low_memory=False)
print('Cleaned dataset shape:', cleaned_df.shape)

# 2. Keep flights with valid delays and departure delay >= 15 minutes
for column in ['DepDelay', 'ArrDelay']:
    cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce')
    cleaned_df[column] = cleaned_df[column].replace([np.inf, -np.inf], np.nan)

model2_df = cleaned_df.dropna(subset=['DepDelay', 'ArrDelay']).copy()
model2_df = model2_df[model2_df['DepDelay'] >= 15].copy()
print('Eligible Model 2 flights:', len(model2_df))

# 3. Create Delay Recovery target variable (three classes)
def classify_recovery(df):
    return np.select(
        [
            df['ArrDelay'] < 15,
            (df['ArrDelay'] >= 15) & (df['ArrDelay'] < df['DepDelay']),
            df['ArrDelay'] >= df['DepDelay']
        ],
        [0, 1, 2],
        default=99
    ).astype(int)

model2_df['Delay_Recovery_Status'] = classify_recovery(model2_df)
class_names = {
    0: 'Recovered',
    1: 'Partially Recovered',
    2: 'Not Recovered or Worsened'
}
print('\nSample target values:')
print(model2_df[['DepDelay', 'ArrDelay', 'Delay_Recovery_Status']].head(10).to_string(index=False))

# 4. Test the boundary conditions
boundary_checks = pd.DataFrame({
    'DepDelay': [15, 20, 15, 40, 40, 41, 40],
    'ArrDelay': [14, 15, 15, 39, 40, 41, -5],
    'Expected': [0, 1, 2, 1, 2, 2, 0]
})
boundary_checks['Actual'] = classify_recovery(boundary_checks)
boundary_checks['Passed'] = boundary_checks['Expected'] == boundary_checks['Actual']
assert boundary_checks['Passed'].all(), 'Boundary check failed.'
assert model2_df['Delay_Recovery_Status'].isin([0, 1, 2]).all(), 'Invalid class detected.'
assert (model2_df['DepDelay'] >= 15).all(), 'Ineligible flight detected.'
print('\nBoundary checks:')
print(boundary_checks.to_string(index=False))
print('\nAll target checks passed.')

# 5. Class distribution and class imbalance
counts = model2_df['Delay_Recovery_Status'].value_counts().reindex([0, 1, 2], fill_value=0)
class_summary = counts.rename('Count').to_frame()
class_summary.index.name = 'Class'
class_summary['Status'] = class_summary.index.map(class_names)
class_summary['Percentage'] = (class_summary['Count'] / len(model2_df) * 100).round(2)
class_summary = class_summary[['Status', 'Count', 'Percentage']]
print('\nDelay Recovery Class Distribution:')
print(class_summary.to_string())

largest = counts.idxmax()
smallest = counts.idxmin()
print(f'\nLargest class: {class_names[largest]} ({counts[largest] / len(model2_df):.2%})')
print(f'Smallest class: {class_names[smallest]} ({counts[smallest] / len(model2_df):.2%})')
if counts.min() > 0:
    print(f'Imbalance ratio: {counts.max() / counts.min():.2f}:1')
print('For Task 4, evaluate macro F1, per-class recall and confusion matrix as well as accuracy.')

# 6. Save results
output_folder = folder / 'results' / 'task2_delay_recovery'
output_folder.mkdir(parents=True, exist_ok=True)
model2_df.to_csv(output_folder / 'delay_recovery_data.csv', index=False)
class_summary.to_csv(output_folder / 'class_distribution.csv')
boundary_checks.to_csv(output_folder / 'boundary_checks.csv', index=False)
print('\nTask 2 outputs saved to:', output_folder)
