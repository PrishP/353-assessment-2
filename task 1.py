import pandas as pd
import numpy as np

   
# 1. Load Monthly Datasets
Jan = pd.read_csv("data/US Airline Data/flight_delays_2025_01.csv")
Feb = pd.read_csv("data/US Airline Data/flight_delays_2025_02.csv")
Dec = pd.read_csv("data/US Airline Data/flight_delays_2025_12.csv")

dataset = {
    "January": Jan,
    "February": Feb,
    "December": Dec
}

   
# 2. Inspect Each Dataset
for name, df in dataset.items():
    print("\n" + "="*50)
    print(name)
    print("="*50)

    print("\nShape:")
    print(df.shape)

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values (Top 10):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

print("\nJanuary Columns:", Jan.columns.tolist())
print("February Columns:", Feb.columns.tolist())
print("December Columns:", Dec.columns.tolist())

   
# 3. Combine All Months
combined_df = pd.concat([Jan, Feb, Dec], ignore_index=True)

print("\nCombined Shape:", combined_df.shape)
print("Total Rows:", len(combined_df))
print("Columns:", combined_df.columns.tolist())

   
# 4. Convert Data Types
combined_df['FlightDate'] = pd.to_datetime(combined_df['FlightDate'], errors='coerce')

numeric_cols = ['ArrDelay', 'DepDelay', 'Distance']
for col in numeric_cols:
    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

# 5. Remove Cancelled & Diverted Flights
clean_df = combined_df[
    (combined_df['Cancelled'] == 0) &
    (combined_df['Diverted'] == 0)
].copy()

print("\nRows after removing cancelled/diverted:", len(clean_df))

# 6. Remove Missing ArrDelay
clean_df = clean_df.dropna(subset=['ArrDelay'])
print("Rows after removing missing ArrDelay:", len(clean_df))


print("\nFinal Cleaned Dataset Shape:", clean_df.shape)
