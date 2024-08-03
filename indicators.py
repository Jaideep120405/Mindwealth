### This code downloads historical stock data for 2 series, here XLK (Technology Select Sector SPDR Fund), SPY (SPDR S&P 500 ETF Trust), 
### and AAPL (Apple Inc., used as a proxy for SPX) using the yfinance library. It then calculates the ratio of XLK to SPY (inverted and negated), 
### computes the correlation between this ratio and the AAPL (proxy for SPX) closing prices, and creates a dual-axis plot showing both the XLK/SPY 
### ratio and the AAPL closing prices over time.




import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download data for XLK, SPY, and SPX
XLK_data = yf.download('XLK', start='2008-12-22')
spy_data = yf.download('SPY', start='2008-12-22')
SPX_data = yf.download('AAPL', start='2008-12-22')  # SPX is the ticker symbol for SPX

# Extract 'Date' column from each dataset
XLK_dates = XLK_data.index
spy_dates = spy_data.index
SPX_dates = SPX_data.index

# Calculate the ratio of XLK to SPY
XLK_spy_ratio =  -1*spy_data['Close'] / XLK_data['Close']

# Calculate correlation between XLK/SPY ratio and SPX closing prices
correlation = XLK_spy_ratio.corr(SPX_data['Close'])

# Create a new DataFrame with 'Date', 'XLK/SPY', and 'SPX Close' columns
combined_df = pd.DataFrame({'XLK/SPY': XLK_spy_ratio, 'SPX Close': SPX_data['Close']}, index=XLK_dates)

# Plot both graphs on a single plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot XLK/SPY ratio on the first y-axis
ax1.plot(combined_df.index, combined_df['XLK/SPY'], label='-ve XLK/SPY Ratio', color='blue')
ax1.set_xlabel('Date')
ax1.set_ylabel('XLK/SPY Ratio', color='blue')
ax1.tick_params('y', colors='blue')
ax1.legend(loc='upper left')

# Create a second y-axis to plot SPX closing prices on a log scale
ax2 = ax1.twinx()
ax2.plot(combined_df.index, combined_df['SPX Close'], label='SPX Close', color='red')
ax2.set_yscale('log')  # Set y-axis to log scale
ax2.set_ylabel('SPX Close (Log Scale)', color='red')
ax2.tick_params('y', colors='red')
ax2.legend(loc='lower left')

plt.title(' XLK/SPY Ratio and SPX Close Over 15 years'f' [[Correlation: {correlation:.2f}]]')
plt.show()

# Print the correlation value
print(f'Correlation between XLK/SPY Ratio and SPX Close: {correlation:.2f}')

# Potential enhancements:

# Date range flexibility: Allow users to input custom date ranges for analysis.
# Moving averages: Add moving averages to smooth out short-term fluctuations in the ratio and closing prices.
# Correlation window: Calculate rolling correlation over different time windows to show how the relationship changes over time.
# Performance metrics: Include additional performance metrics like returns, volatility, or Sharpe ratio.
# Relative strength: Calculate and plot the relative strength of XLK compared to SPY.
# Automated insights: Add functionality to automatically identify significant events or trends in the data.
# Ratio analysis: Include additional statistical analysis of the XLK/SPY ratio, such as its historical range, standard deviation, etc.
# Predictive modeling: Implement simple predictive models to forecast future ratio values or SPX prices based on historical data.
