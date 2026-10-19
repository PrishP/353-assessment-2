import pandas as pd 

Jan = pd.read_csv("data/US Airline Data/flight_delays_2025_01.csv")
Feb = pd.read_csv("data/US Airline Data/flight_delays_2025_02.csv")
Dec = pd.read_csv("data/US Airline Data/flight_delays_2025_12.csv")

dataset = {
    "January": Jan,
    "February": Feb,
    "December": Dec
}

for name, df in dataset.items():
    print("\n"+"="* 50)
    print(name)
    print("=" * 50 )

    print("\nShape:")
    print(df.shape)

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum(). sort_values(ascending=False).head(10))
    print("\nDupplicate Rows:")
    print(df.duplicated().sum())
print(df.duplicated().sum())
print("Jan Shape:", Jan.shape)
print("Feb Shape:", Feb.shape)
print("Dec Shape:", Dec.shape)

print("\nJanuary Columns")
print(Jan.columns.tolist())

print("\nFebuary Columns")
print(Feb.columns.tolist())

print("\nDecember Columns")
print(Dec.columns.tolist())

