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

# # Define the stock symbol and start/end dates
# stock_symbol = "HDFCBANK.NS"  # Change this to your desired stock symbol
# end_date = datetime(2024, 1, 25)
# start_date = end_date - timedelta(days=30)  # 1 month ago

# # Fetch historical stock data
# stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# # Extract the closing prices
# closing_prices = stock_data['Close']

# # If end_date is not a trading day, consider the previous trading day
# if end_date not in closing_prices.index:
#     end_date = closing_prices.index[-1]

# # If start_date is not a trading day, consider the previous trading day
# if start_date not in closing_prices.index:
#     start_date = closing_prices.index[0]

# # Get the closing prices for start_date and end_date
# start_price = closing_prices.loc[start_date]
# end_price = closing_prices.loc[end_date]

# # Calculate the percentage change
# live_percentage_change = ((end_price - start_price) / start_price) * 100

# # Print the result
# print(f"1-month return from {start_date} to {end_date}: {live_percentage_change:.2f}%")


df = pd.read_csv("HDFCBANK.NS.csv")
df['monthly_returns'] = (df['Close'] - df['Open'])
df['monthly_returns_perc'] = df['monthly_returns'] * 100 / df['Close']
jd =  df['monthly_returns_perc'].sort_values()
plt.hist(jd, bins = 30)
plt.plot()

# Specify the value of x you want to indicate
value_of_x = 0

average_monthly_returns_perc = np.mean(df['monthly_returns_perc'])
std_dev_monthly_returns_perc = np.std(df['monthly_returns_perc'])
sigma_level = (df['monthly_returns_perc'].iloc[-1]-average_monthly_returns_perc)/std_dev_monthly_returns_perc
print("sigma_level: ",sigma_level)

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

value_of_x = df['monthly_returns_perc'].iloc[-1]

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

history = df[df['monthly_returns_perc'].between(df['monthly_returns_perc'].iloc[-1]-1, 
                                                df['monthly_returns_perc'].iloc[-1]+1)].index

plt.plot(df['Date'],df['Close'])

value_of_x = 0
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


for i in history:
    value_of_x = df['Date'][i]
    try:
        one_month.append(100*(df['Close'][i]-df['Close'][i-1])/df['Close'][i-1])
    except Exception as e:
        one_month.append("NA")

    try:
        onee_month.append(100*(df['Close'][i+1]-df['Close'][i])/df['Close'][i])
    except Exception as e:
        onee_month.append("NA")
    
    try:
        six_month.append(100*(df['Close'][i+6]-df['Close'][i])/df['Close'][i])
    except Exception as e:
        six_month.append("NA")

    try:
        twelve_month.append(100*(df['Close'][i+12]-df['Close'][i])/df['Close'][i])
    except Exception as e:
        twelve_month.append("NA")

data = {'past-month':one_month,'1-month': onee_month, '6-month': six_month, '12-month': twelve_month, 'date' : value_of_x}
df2 = pd.DataFrame(data)
print(df2)
df2.to_csv('history.csv')
df3 = pd.read_csv('history.csv')
print(df3['6-month'].describe())
print(average_monthly_returns_perc)
print(jd[0])