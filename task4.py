import scipy
from sklearn import linear_model
import pandas as pd

traindf = pd.read_csv("US Airline Data/clean_data.csv")

x = traindf[['DepDelay', 'Distance']]
y = traindf['ArrDelay']

regr = linear_model.LinearRegression()
regr.fit(x, y)

