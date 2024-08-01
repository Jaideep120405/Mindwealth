# sigma(price) indicators.

''' check for the daily returns... and arrange in increasing order. and plot the distribution and check for the sigma moves. 
    plot the price and below the weekly returns. 
    highlight the points where there are weekly sigma moves(>1 sigma).

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import percentileofscore


# Define the stock symbol and start/end dates
stock_symbol = "NVDA"  # Change this to your desired stock symbol
print(stock_symbol)
end_date = datetime.today().date() + timedelta(days=1)
start_date = end_date - timedelta(days=30)  # 1 month ago

# Fetch historical stock data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Extract the closing prices
closing_prices = stock_data['Close']

# If end_date is not a trading day, consider the previous trading day
if end_date not in closing_prices.index:
    end_date = closing_prices.index[-1]

# If start_date is not a trading day, consider the previous tradingday
if start_date not in closing_prices.index:
    start_date = closing_prices.index[0]

# Get the closing prices for start_date and end_date
start_price = closing_prices.loc[start_date]
end_price = closing_prices.loc[end_date]

# Calculate the percentage change
live_percentage_change = ((end_price - start_price) / start_price) * 100

# Print the result
print(f"1-month return from {start_date} to {end_date}: {live_percentage_change:.2f}%")

df = pd.read_csv("NVDA.csv")
df['monthly_returns'] = (df['Close'] - df['Open'])
df['monthly_returns_perc'] = df['monthly_returns'] * 100 / df['Close']
jd =  df['monthly_returns_perc'].sort_values()
plt.hist(jd, bins = 30)
plt.plot()
# Specify the value of x you want to indicate
value_of_x = 0

average_monthly_returns_perc = np.mean(df['monthly_returns_perc'])
std_dev_monthly_returns_perc = np.std(df['monthly_returns_perc'])
sigma_level = (live_percentage_change-average_monthly_returns_perc)/std_dev_monthly_returns_perc
print("sigma_level: ",sigma_level)

df['sigma_level'] = (df['monthly_returns'] - average_monthly_returns_perc)/std_dev_monthly_returns_perc
#print(df['sigma_level'].describe())

# Plot a vertical line at the specified x value
plt.axvline(x=value_of_x, color='red', linestyle='-', linewidth=1.5)


# Annotate the line with a small triangle pointer
plt.annotate(
    f' x = {value_of_x}',  # Text to display
    xy=(value_of_x, 0),  # Coordinate to annotate
    xytext=(value_of_x, 30),  # Coordinate to place the text
    textcoords='offset points',
    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', lw=1),
    color='black',
    fontsize=10,
)

value_of_x = live_percentage_change

# Plot a vertical line at the specified x value
plt.axvline(x=value_of_x, color='blue', linestyle='-', linewidth=0.5)

# Annotate the line with a small triangle pointer
plt.annotate(
    f' x = {value_of_x}',  # Text to display
    xy=(value_of_x, 0),  # Coordinate to annotate
    xytext=(value_of_x, 30),  # Coordinate to place the text
    textcoords='offset points',
    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', lw=1),
    color='blue',
    fontsize=10,
)

plt.show()

if live_percentage_change >=0:
    history = df[df['monthly_returns_perc'].between(live_percentage_change,1000)].index
else:
    history = df[df['monthly_returns_perc'].between(-1000,live_percentage_change)].index

zero_sigma = df[df['monthly_returns_perc'].between(average_monthly_returns_perc-0.1*std_dev_monthly_returns_perc,
                                                   average_monthly_returns_perc+0.1*std_dev_monthly_returns_perc)].index

plt.plot(df['Date'],df['Close'])

for i in history:
    value_of_x = df['Date'][i]

    # Plot a vertical line at the specified x value
    plt.axvline(x=value_of_x, color='blue', linestyle='-', linewidth=0.5)

    # # Annotate the line with a small triangle pointer
    # plt.annotate(
    #     f' x = {value_of_x}',  # Text to display
    #     xy=(value_of_x, 0),  # Coordinate to annotate
    #     xytext=(value_of_x, 30),  # Coordinate to place the text
    #     textcoords='offset points',
    #     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', lw=1),
    #     color='blue',
    #     fontsize=10,
    # )

plt.show()
one_month = []
onee_month = []
three_month = []
six_month = []
twelve_month = []
date = []
sigma_level = []


for i in history:
    value_of_x = df['Date'][i]
    date.append(df['Date'][i])

    try:
        sigma_level.append(((100*(df['Close'][i]-df['Close'][i-1])/df['Close'][i-1])-average_monthly_returns_perc)/std_dev_monthly_returns_perc)
    except Exception as e:
        sigma_level.append("NA")

    try:
        one_month.append(100*(df['Close'][i]-df['Close'][i-1])/df['Close'][i-1])
    except Exception as e:
        one_month.append("NA")

    try:
        onee_month.append(100*(df['Close'][i+1]-df['Close'][i])/df['Close'][i])
    except Exception as e:
        onee_month.append("NA")
    
    try:
        three_month.append(100*(df['Close'][i+3]-df['Close'][i])/df['Close'][i])
    except Exception as e:
        three_month.append("NA")

    try:
        six_month.append(100*(df['Close'][i+6]-df['Close'][i])/df['Close'][i])
    except Exception as e:
        six_month.append("NA")

    try:
        twelve_month.append(100*(df['Close'][i+12]-df['Close'][i])/df['Close'][i])
    except Exception as e:
        twelve_month.append("NA")


data = {'past-month':one_month,'1-month': onee_month,'3-month': three_month, '6-month': six_month, '12-month': twelve_month, 'date' : date,'sigma': sigma_level}
df2 = pd.DataFrame(data)
df2.to_csv('history.csv')
df3 = pd.read_csv('history.csv')
df4 = df3.dropna(subset=['past-month'])
print(df4.drop('Unnamed: 0',axis=1))
print(df4['3-month'].describe())
ans = df4['3-month'].mean()

#percentile:
sorted_returns = np.sort(three_month)

# Find the index positions of the largest negative value and smallest positive value
max_neg_index = np.searchsorted(sorted_returns, 0, side='right')
min_pos_index = np.searchsorted(sorted_returns, 0, side='left')

# Interpolate the percentile based on the positions of the values around zero
if max_neg_index == 0:
    percentile = 0
elif min_pos_index == len(sorted_returns):
    percentile = 100
else:
    percentile = (max_neg_index / len(sorted_returns)) * 100

print("Percentile of zero:", f'{percentile:.1f}')
print('average 1 month returns: ',f'{average_monthly_returns_perc:.2f}')
print('Normalized_average_returns_in_3_months: ', f'{ans:.2f}',' - 3*',f'{average_monthly_returns_perc:.2f}',' = ',f'{(ans-3*average_monthly_returns_perc):.2f}')

one_month = []
onee_month = []
three_month = []
six_month = []
twelve_month = []
date = []
sigma_level = []


for i in zero_sigma:
    value_of_x = df['Date'][i]
    date.append(df['Date'][i])

    try:
        sigma_level.append(((100*(df['Close'][i]-df['Close'][i-1])/df['Close'][i-1])-average_monthly_returns_perc)/std_dev_monthly_returns_perc)
    except Exception as e:
        sigma_level.append("NA")

    try:
        one_month.append(100*(df['Close'][i]-df['Close'][i-1])/df['Close'][i-1])
    except Exception as e:
        one_month.append("NA")

    try:
        onee_month.append(100*(df['Close'][i+1]-df['Close'][i])/df['Close'][i])
    except Exception as e:
        onee_month.append("NA")
    
    try:
        three_month.append(100*(df['Close'][i+3]-df['Close'][i])/df['Close'][i])
    except Exception as e:
        three_month.append("NA")

    try:
        six_month.append(100*(df['Close'][i+6]-df['Close'][i])/df['Close'][i])
    except Exception as e:
        six_month.append("NA")

    try:
        twelve_month.append(100*(df['Close'][i+12]-df['Close'][i])/df['Close'][i])
    except Exception as e:
        twelve_month.append("NA")


data = {'past-month':one_month,'1-month': onee_month,'3-month': three_month, '6-month': six_month, '12-month': twelve_month, 'date' : date,'sigma': sigma_level}
df2 = pd.DataFrame(data)
df2.to_csv('zero_sigma.csv')
df3 = pd.read_csv('zero_sigma.csv')
df4 = df3.dropna(subset=['past-month'])
#print(df4.drop('Unnamed: 0',axis=1))
#print(df4['3-month'].describe())
print('mean of 3-month returns after a zero sigma instance: ', f'{np.mean(three_month):.2f}')
print('Normalized_average_returns_in_3_months: ', f'{ans:.2f}',' - ',f'{np.mean(three_month)}',' = ',f'{(ans-np.mean(three_month)):.2f}')