import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

def plot_stock_prices(stock_prices):
    # Calculate statistics
    min_price = np.min(stock_prices)
    max_price = np.max(stock_prices)
    average_price = np.mean(stock_prices)
    median_price = np.median(stock_prices)
    std_dev_price = np.std()
    # Plot the data
    plt.plot(stock_prices)
    plt.title('Time-Series')
    plt.xlabel('Time')
    plt.ylabel('Series')

    # Create a box with statistics
    stats_box = f"percentile_of_0: {percentileofscore(stock_prices,0):.0f}\nMax: {max_price:.2f}\nAverage: {average_price:.2f}\nMin: {min_price:.2f}\nMedian: {median_price:.2f}"

    plt.text(1, 0.5, stats_box, transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Show the plot
    plt.show()



df = pd.read_excel("aaii_data.xls")
df.dropna(inplace=True)
df2 = df.iloc[:1906]

# Set the second row as column names
df2.columns = df2.iloc[0]

# Drop the second row from the DataFrame using the index
df2 = df2.drop(df2.index[0])

# Reset the index
df2 = df2.reset_index(drop=True)

plt.plot(df2['Spread'])

# Find the bottom 10 percentile values
top_10_percentile = df2['Spread'].quantile(0.05)

# Identify and plot the bottom 10 percentile elements with red dots
top_10_indices = df2[df2['Spread'] < top_10_percentile].index
plt.scatter(top_10_indices, df2.loc[top_10_indices, 'Spread'], color='red', label='Top 5 Percentile')


plt.title("AAII Bull-Bear Spread")
plt.show()

# Extract the last entry in the 'Spread' column
last_entry = df2['Spread'].iloc[-1]

# Calculate the percentile using percentileofscore
percentile = percentileofscore(df2['Spread'], last_entry)

print(f"Last value in 'Spread' column: {last_entry}")
print(f"The percentile of the last entry ({last_entry}) in the 'Spread' column is: {percentile}%")

print(df2.columns)

plt.plot(df2['Close'])
for index in top_10_indices:
    plt.axvline(index, color='r', linestyle='--', linewidth = 0.5)
plt.show()

new1 = []
new3 = []
new6 = []
new12 = []

for i in top_10_indices:
    try:
        new1.append((df2['Close'][i+4]-df2['Close'][i])*100 / df2['Close'][i])
    except KeyError:
        break

for i in top_10_indices:
    try:
        new3.append((df2['Close'][i+13]-df2['Close'][i])*100 / df2['Close'][i])
    except KeyError:
        break

for i in top_10_indices:
    try:
        new6.append((df2['Close'][i+26]-df2['Close'][i])*100 / df2['Close'][i])
    except KeyError:
        break

for i in top_10_indices:
    try:
        new12.append((df2['Close'][i+52]-df2['Close'][i])*100 / df2['Close'][i])
    except KeyError:
        break

np.sort(new1)
np.sort(new3)
np.sort(new6)
np.sort(new12)

plot_stock_prices(new12)
print(percentileofscore(new1, 0))
