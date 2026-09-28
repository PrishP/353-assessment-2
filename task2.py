import pandas as pd
import numpy as np
 
# 1. Load Cleaned Data
cleaned_df = pd.read_csv("US Airline Data/clean_data.csv")
print(cleaned_df.shape)

# 2. Create New Variable
cleaned_df['Arrival_Disruption_Severity'] = [0 if x < 15 else 1 if 15 >= x <30 else 2 if 30 >= x < 90 else 3 if 90 >= x else 99 for x in cleaned_df['ArrDelay']]
print(cleaned_df.loc[50])
