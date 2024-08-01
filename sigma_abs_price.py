import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_ipo_date(ticker_symbol):
    stock_data = yf.download(ticker_symbol, period='max')
    ipo_date = stock_data.index[0]
    
    return ipo_date

def fetch_stock_data_from_ipo(ticker_symbol):
    ipo_date = get_ipo_date(ticker_symbol)
    stock_data = yf.download(ticker_symbol, start=ipo_date, interval='1d')
    
    return stock_data

def calculate_sigma(stock_data, lookback_time_period):
    stock_data.reset_index(inplace=True)
    stock_data['sigma_value'] = np.nan
    
    start_date = stock_data['Date'].min()
    end_date = stock_data['Date'].max()
    
    lookback_period_days = int(lookback_time_period * 365)
    
    for index, row in stock_data.iterrows():
        current_date = row['Date']
        lookback_date = current_date - pd.Timedelta(days=lookback_period_days)
        
        if lookback_date < start_date:
            continue
        
        lookback_data = stock_data[(stock_data['Date'] >= lookback_date) & (stock_data['Date'] <= current_date)]
        
        if len(lookback_data) > 0:
            mean_value = lookback_data['Close'].mean()
            std_value = lookback_data['Close'].std()
            sigma_value = (row['Close'] - mean_value) / std_value
            stock_data.at[index, 'sigma_value'] = sigma_value
    
    return stock_data

ticker_symbol = 'TSLA'  ###########################################################################.

ipo_date = get_ipo_date(ticker_symbol)
print(f"IPO Date for {ticker_symbol}: {ipo_date}")

stock_data = fetch_stock_data_from_ipo(ticker_symbol)

lookback_time_period = 1  #########################################################################.

stock_data_with_sigma = calculate_sigma(stock_data, lookback_time_period)

print(stock_data_with_sigma)

last_sigma_value = stock_data_with_sigma['sigma_value'].iloc[-1]
print(f"Last sigma value: {last_sigma_value}")

closing_prices = stock_data_with_sigma['Close'].iloc[-int(lookback_time_period * 365):]
average_abs_price = closing_prices.mean()
end_price = closing_prices.iloc[-1]
sigma_level = stock_data_with_sigma['sigma_value'].iloc[-1]

# histogram
plt.hist(closing_prices.sort_values(), bins=20)
plt.plot()

value_of_x = average_abs_price

plt.axvline(x=value_of_x, color='red', linestyle='-', linewidth=1.5)

plt.annotate(
    f' avg = {value_of_x:.2f}',  
    xy=(value_of_x, 0),  
    xytext=(value_of_x, 1),  
    textcoords='data',
    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=1', lw=1),
    color='black',
    fontsize=8,
)

value_of_x = end_price

plt.axvline(x=value_of_x, color='blue', linestyle='-', linewidth=0.5)
plt.annotate(
    f' current = {value_of_x:.2f}',  
    xy=(value_of_x, 0),  
    xytext=(value_of_x, 1),  
    textcoords='data',
    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=1', lw=1),
    color='blue',
    fontsize=8,
)

plt.title(f"Histogram of Absolute Prices of {ticker_symbol}. Current sigma at: {sigma_level:.2f}.")
plt.show()