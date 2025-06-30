import gradio as gr
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
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Volatility'], df['Returns'], c=df['Sharpe'], cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(max_sharpe['Volatility'], max_sharpe['Returns'], c='red', marker='*', s=200)
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Expected Return')
    plt.grid(True)
    plt.tight_layout()
    plot_path = 'efficient_frontier_gradio.png'
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def run_optimizer(ticker_input, investment, risk_free_rate, max_alloc):
    tickers = [t.strip().upper() for t in ticker_input.split(',')]
    data = fetch_data(tickers)
    daily_returns, annual_returns, cov_matrix = calculate_returns(data)
    df = simulate_portfolios(5000, annual_returns, cov_matrix, risk_free_rate, max_alloc)

    max_sharpe = df.loc[df['Sharpe'].idxmax()]
    portfolio = pd.DataFrame({
        'Ticker': tickers,
        'Allocation (%)': [round(w * 100, 2) for w in max_sharpe['Weights']]
    })

    plot_path = plot_efficient_frontier(df)
    summary = f"""📊 Optimal Portfolio:
{portfolio.to_string(index=False)}

💰 Expected Return: {round(max_sharpe['Returns']*100, 2)}%
📉 Volatility: {round(max_sharpe['Volatility']*100, 2)}%
📈 Sharpe Ratio: {round(max_sharpe['Sharpe'], 2)}
"""
    return summary, plot_path

demo = gr.Interface(
    fn=run_optimizer,
    inputs=[
        gr.Textbox(label="Enter stock tickers (comma-separated)", placeholder="e.g. AAPL, MSFT, GOOGL"),
        gr.Number(label="Total Investment (Not used, for display only)", value=100000),
        gr.Slider(0.0, 0.1, value=0.02, label="Risk-Free Rate (e.g. 0.02 for 2%)"),
        gr.Slider(0.1, 1.0, value=0.4, label="Max Allocation Per Stock")
    ],
    outputs=[
        gr.Textbox(label="Optimal Portfolio Summary"),
        gr.Image(type="filepath", label="Efficient Frontier Plot")
    ],
    title="📈 Stock Portfolio Optimizer",
    description="Optimize your portfolio using Modern Portfolio Theory and Monte Carlo Simulations."
)

if __name__ == "__main__":
    demo.launch()