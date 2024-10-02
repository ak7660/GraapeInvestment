import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from statsmodels.api import OLS
import yfinance as yf

import matplotlib.pyplot as plt
import statsmodels.api as sm

import logging

# ----- Setup Logging for Traditional Assets Strategy -----
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to capture essential information
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("traditional_assets_backtest.log"),  # Log to a file
        logging.StreamHandler()  # Also log to console
    ]
)

# ----- Function to Load Crypto Price and Market Cap Data -----
def load_price_and_market_cap_data(folder_path, coins):
    price_data = {}
    market_cap_data = {}

    for coin in coins:
        file_path = os.path.join(folder_path, f"{coin}.csv")
        if not os.path.exists(file_path):
            logging.warning(f"CSV file for {coin} not found at path: {file_path}. Skipping this coin.")
            continue
        df = pd.read_csv(file_path, parse_dates=['Start'], index_col='Start')
        price_data[coin] = df['Close']
        market_cap_data[coin] = df[['Market Cap', 'Volume']]

    price_data = pd.DataFrame(price_data)
    price_data.ffill(inplace=True)
    price_data.dropna(inplace=True)
    market_cap_data = {coin: df for coin, df in market_cap_data.items()}
    logging.info("Loaded Crypto price and market cap data successfully.")
    return price_data, market_cap_data

# ----- Function to Calculate Factors -----
def calculate_factors(price_data, cap_data, rolling_window=7):
    logging.info("Calculating factors (Momentum, Size, Value) for Crypto Strategy.")
    # Momentum: past week return
    momentum = price_data.pct_change(rolling_window).shift(1)
    logging.debug(f"Momentum head:\n{momentum.head()}")

    # Size: market cap of the coins
    size = pd.DataFrame({coin: cap_data[coin]['Market Cap'] for coin in cap_data})
    logging.debug(f"Size head:\n{size.head()}")

    # Value: inverse of NVT ratio (Market Cap / Volume)
    value = pd.DataFrame({coin: cap_data[coin]['Market Cap'] / cap_data[coin]['Volume'] for coin in cap_data})
    logging.debug(f"Value (NVT) head before handling infinities:\n{value.head()}")

    # Handle potential divide-by-zero or inf values in NVT ratio
    value.replace([np.inf, -np.inf], np.nan, inplace=True)
    value.fillna(0, inplace=True)
    value = 1 / value
    value = value.clip(lower=-1e9, upper=1e9)
    logging.debug(f"Value (after inversion and clipping) head:\n{value.head()}")

    return momentum, size, value

# ----- Function to Backtest Crypto Strategy (Clean, No Logging) -----
def backtest_crypto_strategy(start_date, end_date, initial_investment, price_data, cap_data, rebalance_time='00:00'):
    portfolio_value = initial_investment
    portfolio_history = []
    returns_history = []

    dates = pd.date_range(start=start_date, end=end_date, freq='W-SUN')
    momentum, size, value = calculate_factors(price_data, cap_data)
    scaler = StandardScaler()
    momentum_std = pd.DataFrame(scaler.fit_transform(momentum.fillna(0)), index=momentum.index, columns=momentum.columns)
    size_std = pd.DataFrame(scaler.fit_transform(size.fillna(0)), index=size.index, columns=size.columns)
    value_std = pd.DataFrame(scaler.fit_transform(value.fillna(0)), index=value.index, columns=value.columns)
    combined_signal = (momentum_std + size_std + value_std) / 3
    combined_signal = combined_signal.loc[start_date:end_date]

    for date in dates:
        portfolio_date = date

        if portfolio_date in combined_signal.index:
            signals = combined_signal.loc[portfolio_date]

            if signals.isna().all():
                portfolio_history.append(portfolio_value)
                returns_history.append(0.0)
                continue

            long_threshold = 0.5
            short_threshold = -0.5
            long_coins = signals[signals > long_threshold].index
            short_coins = signals[signals < short_threshold].index

            if len(long_coins) == 0 and len(short_coins) == 0:
                portfolio_history.append(portfolio_value)
                returns_history.append(0.0)
                continue

            total_market_cap = size.loc[portfolio_date, long_coins].sum() if len(long_coins) > 0 else 1
            weights = size.loc[portfolio_date, long_coins] / total_market_cap if len(long_coins) > 0 else pd.Series(0, index=long_coins)

            try:
                future_dates = price_data.index[price_data.index > portfolio_date]
                if future_dates.empty:
                    portfolio_history.append(portfolio_value)
                    returns_history.append(0.0)
                    continue
                next_date = future_dates[0]

                if next_date not in price_data.index:
                    portfolio_history.append(portfolio_value)
                    returns_history.append(0.0)
                    continue

                long_price_current = price_data.loc[portfolio_date, long_coins]
                long_price_next = price_data.loc[next_date, long_coins]
                long_returns = (long_price_next / long_price_current - 1) * weights
                long_return_total = long_returns.sum()

                short_return_total = 0.0
                if len(short_coins) > 0:
                    short_price_current = price_data.loc[portfolio_date, short_coins]
                    short_price_next = price_data.loc[next_date, short_coins]
                    short_returns = (short_price_next / short_price_current - 1)
                    short_return_total = short_returns.mean() if not short_returns.empty else 0.0

            except Exception as e:
                portfolio_history.append(portfolio_value)
                returns_history.append(0.0)
                continue

            weekly_return = long_return_total - short_return_total
            portfolio_value *= (1 + weekly_return)
            portfolio_history.append(portfolio_value)
            returns_history.append(weekly_return)
        else:
            portfolio_history.append(portfolio_value)
            returns_history.append(0.0)

    if portfolio_history:
        portfolio_history_df = pd.DataFrame(portfolio_history, index=dates, columns=['Portfolio Value'])
    else:
        portfolio_history_df = pd.DataFrame(columns=['Portfolio Value'], index=dates)

    return portfolio_history_df, returns_history

# ----- Function to Backtest Traditional Assets Strategy with Logging -----
def backtest_traditional_strategy(start_date, end_date, initial_investment, traditional_tickers, rebalance_freq='W-MON'):
    logging.info("Starting Traditional Assets backtest.")

    # Fetch historical price data for traditional assets using yfinance
    logging.info("Fetching historical price data for Traditional Assets from yfinance.")
    try:
        data = yf.download(traditional_tickers, start=start_date, end=end_date, progress=False)['Adj Close']
        logging.info(f"Fetched data for tickers: {traditional_tickers}")
    except Exception as e:
        logging.error(f"Error fetching data for Traditional Assets tickers: {e}")
        return pd.DataFrame(), []

    if data.empty:
        logging.error("No data fetched for Traditional Assets. Please check the tickers and date range.")
        return pd.DataFrame(), []

    # Handle missing data
    data.ffill(inplace=True)
    data.dropna(inplace=True)
    logging.info(f"Data after forward fill and dropna: {data.shape[0]} rows and {data.shape[1]} columns.")

    # Initialize portfolio metrics
    portfolio_value = initial_investment
    portfolio_history = []
    returns_history = []

    # Determine rebalance dates
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_freq)
    logging.info(f"Rebalance dates: {rebalance_dates}")

    # Determine the number of assets
    num_assets = len(traditional_tickers)
    if num_assets == 0:
        logging.error("No Traditional Tickers provided for backtesting.")
        return pd.DataFrame(), []

    # Initial equal weights
    weights = pd.Series(1/num_assets, index=traditional_tickers)
    logging.debug(f"Initial weights: \n{weights}")

    for i, rebalance_date in enumerate(rebalance_dates):
        logging.info(f"Rebalancing portfolio on {rebalance_date.date()}")

        if rebalance_date not in data.index:
            logging.warning(f"Rebalance date {rebalance_date.date()} not found in data index. Skipping this date.")
            portfolio_history.append(portfolio_value)
            returns_history.append(0.0)
            continue

        # Current equal weights
        weights = pd.Series(1/num_assets, index=traditional_tickers)
        logging.debug(f"Weights before rebalancing: \n{weights}")

        # Find the next rebalance date
        future_dates = data.index[data.index > rebalance_date]
        if not future_dates.empty:
            next_rebalance_date = future_dates[0]
            logging.debug(f"Next rebalance date: {next_rebalance_date.date()}")
        else:
            logging.warning(f"No future rebalance date available after {rebalance_date.date()}. Ending backtest.")
            portfolio_history.append(portfolio_value)
            returns_history.append(0.0)
            break

        # Current and future prices
        try:
            current_prices = data.loc[rebalance_date, traditional_tickers]
            future_prices = data.loc[next_rebalance_date, traditional_tickers]
            logging.debug(f"Current prices on {rebalance_date.date()}:\n{current_prices}")
            logging.debug(f"Future prices on {next_rebalance_date.date()}:\n{future_prices}")
        except KeyError as e:
            logging.error(f"Price data missing for date: {e}. Skipping this period.")
            portfolio_history.append(portfolio_value)
            returns_history.append(0.0)
            continue

        # Calculate returns
        try:
            returns = (future_prices / current_prices - 1).replace([np.inf, -np.inf], np.nan).fillna(0)
            logging.debug(f"Returns from {rebalance_date.date()} to {next_rebalance_date.date()}:\n{returns}")
        except Exception as e:
            logging.error(f"Error calculating returns: {e}. Skipping this period.")
            portfolio_history.append(portfolio_value)
            returns_history.append(0.0)
            continue

        # Calculate portfolio return
        portfolio_return = (returns * weights).sum()
        logging.info(f"Portfolio return for this period: {portfolio_return:.6f}")

        # Update portfolio value
        portfolio_value *= (1 + portfolio_return)
        logging.info(f"Updated Traditional Portfolio value: {portfolio_value:.2f}")

        # Record portfolio history
        portfolio_history.append(portfolio_value)
        returns_history.append(portfolio_return)

    # Convert history to DataFrame
    if portfolio_history:
        # Ensure the length of rebalance_dates matches the portfolio_history
        rebalance_dates_trimmed = rebalance_dates[:len(portfolio_history)]
        portfolio_history_df = pd.DataFrame(portfolio_history, index=rebalance_dates_trimmed, columns=['Traditional Portfolio Value'])
        logging.info("Completed backtest for Traditional Assets Strategy.")
    else:
        logging.warning("No portfolio history recorded for Traditional Assets Strategy.")
        portfolio_history_df = pd.DataFrame(columns=['Traditional Portfolio Value'], index=rebalance_dates)

    return portfolio_history_df, returns_history

# ----- Function to Calculate Performance Metrics -----
def calculate_metrics(portfolio_history, returns_history, strategy_name, start_date, end_date):
    logging.info(f"Calculating performance metrics for {strategy_name} Strategy.")
    # Convert returns_history to a pandas Series with backtest dates as index
    backtest_dates = portfolio_history.index
    returns_series = pd.Series(returns_history, index=backtest_dates)
    logging.debug(f"Returns Series for {strategy_name}: \n{returns_series}")

    # Calculate Sharpe Ratio
    returns_cleaned = returns_series.dropna()
    logging.debug(f"Cleaned returns history for {strategy_name}: \n{returns_cleaned}")

    if len(returns_cleaned) == 0:
        logging.warning(f"No returns available to calculate metrics for {strategy_name} Strategy.")
        return np.nan, np.nan, np.nan

    sharpe_ratio = (np.mean(returns_cleaned) / np.std(returns_cleaned)) * np.sqrt(52)  # Annualized Sharpe Ratio
    logging.info(f"Calculated Sharpe Ratio for {strategy_name} Strategy: {sharpe_ratio:.4f}")

    # Fetch S&P 500 data using yfinance
    sp500_ticker = '^GSPC'
    logging.info(f"Fetching S&P 500 data from {start_date} to {end_date} using yfinance for {strategy_name} Strategy.")

    try:
        sp500_data = yf.download(sp500_ticker, start=start_date, end=end_date, progress=False)
        if sp500_data.empty:
            logging.error("Fetched S&P 500 data is empty.")
            return sharpe_ratio, np.nan, np.nan
    except Exception as e:
        logging.error(f"Error fetching S&P 500 data: {e}")
        return sharpe_ratio, np.nan, np.nan

    # Ensure the data is sorted in ascending order
    sp500_data.sort_index(inplace=True)
    logging.debug("Sorted S&P 500 data in ascending order.")

    # Calculate daily returns for S&P 500
    market_returns_daily = sp500_data['Close'].pct_change().dropna()
    logging.debug(f"Calculated daily market returns for {strategy_name} Strategy.")

    # Resample daily market returns to weekly returns (W-SUN) by compounding
    market_returns_weekly = market_returns_daily.resample('W-SUN').agg(lambda x: (1 + x).prod() - 1)
    logging.debug(f"Weekly market returns:\n{market_returns_weekly.head()}")

    # Align the indices of returns_history and market_returns_weekly using common dates
    common_dates = returns_series.index.intersection(market_returns_weekly.index)
    aligned_returns_history = returns_series.loc[common_dates]
    aligned_market_returns = market_returns_weekly.loc[common_dates]

    logging.debug(f"Common dates after intersection: {common_dates}")
    logging.debug(f"Aligned Returns History:\n{aligned_returns_history}")
    logging.debug(f"Aligned Market Returns:\n{aligned_market_returns}")

    # Ensure both series have no NaN values
    aligned_returns_history = aligned_returns_history.dropna()
    aligned_market_returns = aligned_market_returns.dropna()

    # Re-align indices to ensure they match exactly
    common_dates_final = aligned_returns_history.index.intersection(aligned_market_returns.index)
    aligned_returns_history = aligned_returns_history.loc[common_dates_final]
    aligned_market_returns = aligned_market_returns.loc[common_dates_final]

    logging.debug(f"Final common dates after re-intersection: {common_dates_final}")

    # Check if we have enough data to run the regression
    if len(aligned_returns_history) < 2:
        logging.warning(f"Not enough data points to perform regression for {strategy_name} Strategy.")
        return sharpe_ratio, np.nan, np.nan

    # Convert data to float to avoid dtype issues
    aligned_returns_history = aligned_returns_history.astype(float)
    aligned_market_returns = aligned_market_returns.astype(float)

    # Calculate Beta and Alpha using OLS regression
    X = sm.add_constant(aligned_market_returns)  # Adds a constant term to the predictor
    model = OLS(aligned_returns_history, X).fit()
    beta = model.params[aligned_market_returns.name]
    alpha = model.params['const']

    logging.info(f"Calculated Beta for {strategy_name} Strategy: {beta:.4f}")
    logging.info(f"Calculated Alpha for {strategy_name} Strategy: {alpha:.4f}")

    return sharpe_ratio, beta, alpha

# ----- Plotting Functions -----
def plot_portfolio(portfolio_history, title='Portfolio Value Over Time', label='Portfolio Value'):
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_history.index, portfolio_history.iloc[:,0], label=label)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_equity_drawdown(portfolio_history, title='Equity Drawdown Over Time', label='Drawdown'):
    drawdown = portfolio_history.copy()
    drawdown['Max Value'] = drawdown.iloc[:,0].cummax()
    drawdown['Drawdown'] = (drawdown.iloc[:,0] - drawdown['Max Value']) / drawdown['Max Value']

    plt.figure(figsize=(10, 6))
    plt.plot(drawdown.index, drawdown['Drawdown'], label=label, color='red')
    plt.fill_between(drawdown.index, drawdown['Drawdown'], color='red', alpha=0.3)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_daily_returns_on_assets(price_data, title='Daily Returns on Assets'):
    daily_returns = price_data.pct_change().dropna()
    plt.figure(figsize=(12, 8))
    for column in daily_returns.columns:
        plt.plot(daily_returns.index, daily_returns[column], label=column)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Daily Return (%)')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()

def plot_both_portfolios(crypto_history, traditional_history):
    plt.figure(figsize=(12, 6))
    plt.plot(crypto_history.index, crypto_history['Portfolio Value'], label='Crypto Portfolio')
    plt.plot(traditional_history.index, traditional_history['Traditional Portfolio Value'], label='Traditional Portfolio')
    plt.title('Portfolio Values Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_both_drawdowns(crypto_history, traditional_history):
    # Crypto Drawdown
    crypto_drawdown = crypto_history.copy()
    crypto_drawdown['Max Value'] = crypto_drawdown['Portfolio Value'].cummax()
    crypto_drawdown['Drawdown'] = (crypto_drawdown['Portfolio Value'] - crypto_drawdown['Max Value']) / crypto_drawdown['Max Value']

    # Traditional Drawdown
    traditional_drawdown = traditional_history.copy()
    traditional_drawdown['Max Value'] = traditional_drawdown['Traditional Portfolio Value'].cummax()
    traditional_drawdown['Drawdown'] = (traditional_drawdown['Traditional Portfolio Value'] - traditional_drawdown['Max Value']) / traditional_drawdown['Max Value']

    plt.figure(figsize=(12, 6))
    plt.plot(crypto_drawdown.index, crypto_drawdown['Drawdown'], label='Crypto Drawdown', color='red')
    plt.plot(traditional_drawdown.index, traditional_drawdown['Drawdown'], label='Traditional Drawdown', color='blue')
    plt.title('Equity Drawdowns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()

# ----- Main Execution Block -----
if __name__ == "__main__":
    # Parameters
    start_date = "2023-10-01"
    end_date = "2024-09-29"    # Ensure yfinance has data up to this date
    initial_investment = 100_000_000  # 100M USD

    # ----- Crypto Strategy -----
    # Define top 10 cryptocurrencies (excluding USDT and USDC)
    top_10_coins = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'SOL', 'TON', 'TRX', 'AVAX']
    folder_path = 'Data'  # Path to CSV files containing market cap data

    # Load price and market cap data from CSV files
    price_data, market_cap_data = load_price_and_market_cap_data(folder_path, top_10_coins)

    # Backtest the crypto strategy
    portfolio_history_crypto, returns_history_crypto = backtest_crypto_strategy(
        start_date, end_date, initial_investment, price_data, market_cap_data
    )

    # ----- Traditional Assets Strategy -----
    # Define traditional asset tickers
    traditional_tickers = [
        'QQQ',    # Invesco QQQ Trust (Technology)
        'VGT',    # Vanguard Information Technology ETF
        'XLK',    # Technology Select Sector SPDR Fund
        'GLD',    # SPDR Gold Shares
        'SUSA',   # iShares MSCI USA ESG Select ETF
        'ESGU',   # iShares ESG Aware MSCI USA ETF
        'IYT',    # iShares Transportation Average ETF (example as a sector)
        'VHT',    # Vanguard Health Care ETF (example mutual fund)
        'VPU'     # Vanguard Utilities ETF (additional ETF for diversification)
    ]
    # Remove duplicates if any
    traditional_tickers = list(set(traditional_tickers))

    # Backtest the Traditional Assets strategy with logging and adjusted rebalance frequency ('W-MON')
    traditional_portfolio_history, traditional_returns_history = backtest_traditional_strategy(
        start_date, end_date, initial_investment, traditional_tickers, rebalance_freq='W-MON'
    )

    # ----- Calculate Performance Metrics -----
    # Crypto Metrics
    sharpe_ratio_crypto, beta_crypto, alpha_crypto = calculate_metrics(
        portfolio_history_crypto, returns_history_crypto, "Crypto", start_date, end_date
    )

    # Traditional Metrics
    sharpe_ratio_traditional, beta_traditional, alpha_traditional = calculate_metrics(
        traditional_portfolio_history, traditional_returns_history, "Traditional Assets", start_date, end_date
    )

    # ----- Print Performance Metrics -----
    print("=== Performance Metrics ===")
    print("Crypto Strategy:")
    print(f"Sharpe Ratio: {sharpe_ratio_crypto:.4f}")
    print(f"Beta: {beta_crypto:.4f}")
    print(f"Alpha: {alpha_crypto:.4f}\n")

    print("Traditional Assets Strategy:")
    print(f"Sharpe Ratio: {sharpe_ratio_traditional:.4f}")
    print(f"Beta: {beta_traditional:.4f}")
    print(f"Alpha: {alpha_traditional:.4f}\n")

    # ----- Plotting -----
    # Plot both portfolio values
    if not portfolio_history_crypto.empty and not traditional_portfolio_history.empty:
        plot_both_portfolios(portfolio_history_crypto, traditional_portfolio_history)

    # Plot both equity drawdowns
    if not portfolio_history_crypto.empty and not traditional_portfolio_history.empty:
        plot_both_drawdowns(portfolio_history_crypto, traditional_portfolio_history)

    # Plot daily returns on assets for Traditional Assets
    # This provides insights into the volatility and performance of individual assets
    logging.info("Fetching historical price data for Traditional Assets for plotting.")
    try:
        traditional_price_data = yf.download(traditional_tickers, start=start_date, end=end_date, progress=False)['Adj Close']
        traditional_price_data.ffill(inplace=True)
        traditional_price_data.dropna(inplace=True)
        logging.info("Plotting daily returns for Traditional Assets.")
        plot_daily_returns_on_assets(traditional_price_data, title='Daily Returns on Traditional Assets')
    except Exception as e:
        logging.error(f"Error fetching Traditional Assets price data for plotting: {e}")

    # Optional: Plot daily returns on assets for Crypto (if desired)
    # Uncomment the following lines if you wish to visualize Crypto daily returns
    # print("Plotting daily returns for Crypto Assets.")
    # plot_daily_returns_on_assets(price_data, title='Daily Returns on Crypto Assets')