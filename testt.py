import numpy as np

def simulate_trade(low_bid, high_bid):

    total_profit = 0

    start = 900
    end = 1000
    dataset_size = 100000
    probabilities = np.linspace(0,1, end-start+1)
    threshold_prices = np.random.choice(np.arange(start, end+1), size = dataset_size, p = probabilities/np.sum(probabilities))

    for price in threshold_prices:
        if low_bid < price <= high_bid:
            total_profit += 1000-high_bid
        if price <= low_bid:
            total_profit += 1000-low_bid
    
    return total_profit

def find_optimal_bids(max_iterations = 10000):

    best_low_bid = 900
    best_high_bid = 1000
    best_total_profit = simulate_trade(best_low_bid, best_high_bid)

    for x in range(max_iterations):
        low_bid = np.random.uniform(900,1000)
        high_bid = np.random.uniform(low_bid, 1000)
        total_profit = simulate_trade(low_bid, high_bid)

        if total_profit > best_total_profit:
            best_low_bid = low_bid
            best_high_bid = high_bid
            best_total_profit = total_profit

    return best_low_bid, best_high_bid, best_total_profit

low_bid, high_bid, total_profit = find_optimal_bids()

print(f"Lowest bid = {low_bid:.2f}")
print(f"Highest bid = {high_bid:.2f}")
print(f"Total profit = {total_profit:.2f}")   