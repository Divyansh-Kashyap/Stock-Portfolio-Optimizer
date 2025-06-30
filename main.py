import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fetch_data(tickers, start='2020-01-01', end='2024-12-31'):
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)
    if len(tickers) == 1:
        return data[['Adj Close']].rename(columns={'Adj Close': tickers[0]})
    return data['Adj Close']

def calculate_returns(data):
    daily_returns = data.pct_change().dropna()
    annual_returns = daily_returns.mean() * 252
    return daily_returns, annual_returns, daily_returns.cov() * 252

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std_dev

def simulate_portfolios(n_portfolios, mean_returns, cov_matrix, risk_free_rate=0.02, max_allocation=1.0):
    results = {'Returns': [], 'Volatility': [], 'Sharpe': [], 'Weights': []}
    num_assets = len(mean_returns)

    for _ in range(n_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        if any(weights > max_allocation):
            continue
        ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe = (ret - risk_free_rate) / vol
        results['Returns'].append(ret)
        results['Volatility'].append(vol)
        results['Sharpe'].append(sharpe)
        results['Weights'].append(weights)
    
    return pd.DataFrame(results)

def plot_efficient_frontier(df):
    max_sharpe = df.loc[df['Sharpe'].idxmax()]
    plt.figure(figsize=(10, 7))
    plt.scatter(df['Volatility'], df['Returns'], c=df['Sharpe'], cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(max_sharpe['Volatility'], max_sharpe['Returns'], c='red', marker='*', s=200)
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Expected Return')
    plt.grid(True)
    plt.savefig('efficient_frontier.png')
    plt.show()

def main():
    tickers = input("Enter stock tickers (comma-separated): ").upper().split(',')
    investment = float(input("Enter total investment amount: "))
    risk_free_rate = float(input("Enter risk-free rate (e.g. 0.02 for 2%): "))
    max_alloc = float(input("Enter max allocation per stock (e.g. 0.4 for 40%): "))

    print("\nFetching data and simulating portfolios...\n")
    data = fetch_data(tickers)
    daily_returns, annual_returns, cov_matrix = calculate_returns(data)
    df = simulate_portfolios(10000, annual_returns, cov_matrix, risk_free_rate, max_alloc)

    max_sharpe = df.loc[df['Sharpe'].idxmax()]
    print("✅ Optimal Portfolio Allocation:")
    for ticker, weight in zip(tickers, max_sharpe['Weights']):
        print(f"{ticker}: {round(weight * 100, 2)}%")

    print(f"\n📊 Expected Return: {round(max_sharpe['Returns'] * 100, 2)}%")
    print(f"📉 Volatility: {round(max_sharpe['Volatility'] * 100, 2)}%")
    print(f"📈 Sharpe Ratio: {round(max_sharpe['Sharpe'], 2)}")

    # Save to CSV
    portfolio_df = pd.DataFrame({
        'Ticker': tickers,
        'Allocation (%)': [round(w * 100, 2) for w in max_sharpe['Weights']]
    })
    portfolio_df.to_csv("optimal_portfolio.csv", index=False)
    print("\n📁 Saved optimal portfolio to 'optimal_portfolio.csv'")

    # Plot
    plot_efficient_frontier(df)

if __name__ == "__main__":
    main()
