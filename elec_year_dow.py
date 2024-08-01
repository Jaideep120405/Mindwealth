import pandas as pd
import yfinance as yf

# Function to get weekly data for a specific year
def get_weekly_data(year):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    # Fetch historical data using yfinance
    spx_data = yf.download('^DJI', start=start_date, end=end_date, interval='1wk')

    return spx_data

# List of election years from 1992 to 2020
election_years = list(range(1992, 2021, 4))

# Create 8 data frames
data_frames = {}

# Loop through election years and create data frames
for i, year in enumerate(election_years):
    data_frames[f'df_{i+1}'] = get_weekly_data(year)

# Print information about the data frames
for key, df in data_frames.items():
    print(f"{key}: {df.shape}")

# Example: Access data for the first data frame
print("\nSample data for df_1:")
print(data_frames['df_1'].head())

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt



df2 = pd.read_excel('elec_years_close_dji.xlsx')
dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(1, 100))

df2['AVG'] = scaler.fit_transform(df2['AVG'].values.reshape(-1, 1)).flatten()

values = df2['AVG']

# Plotting
plt.plot( values.index,values,label = 'DJI')
plt.show()
