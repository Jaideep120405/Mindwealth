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
