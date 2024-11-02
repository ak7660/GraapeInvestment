# Cryptocurrency Investment Strategy with Factor Investing and Portfolio Insurance
#### By GrapeInvestmentGroup (team project)
This repository contains an implementation of a cryptocurrency-focused investment strategy that combines **Factor Investing** with a **Portfolio Insurance** strategy. The goal is to optimize returns while managing downside risk dynamically.

## Strategy Overview

The strategy employs **Factor Investing** principles with a dynamic **Synthetic Put (SP)** portfolio insurance approach. The strategy is divided into two main components:

1. **Factor-Based Cryptocurrency Selection**:
   - **Momentum**: Selects cryptocurrencies based on past price performance.
   - **Size**: Uses market capitalization to target smaller-cap cryptocurrencies, which historically show higher growth potential.
   - **Value**: Employs the Network Value to Transactions (NVT) ratio to identify undervalued cryptocurrencies.

2. **Portfolio Insurance using Synthetic Put (SP)**:
   - The **SP strategy** dynamically adjusts the portfolio’s hedge ratio based on the relative performance of long and short positions.
   - This helps mitigate potential losses during market downturns while allowing for upside potential.

## Key Components of the Code

### 1. **Setup & Data Loading**
   - The strategy loads historical price and market cap data for the top 10 cryptocurrencies and traditional assets (ETFs, gold) using CSV files and `yfinance`.
   - Data is cleaned and prepared for backtesting.
   
### 2. **Factor Calculation**
   - **Momentum** is calculated as the weekly percentage change.
   - **Size** is based on market capitalization.
   - **Value** is computed as the inverse of the NVT ratio.

### 3. **Backtesting**
   - The backtest function simulates portfolio performance over time, adjusting the portfolio based on factor signals and rebalancing every three months.
   - **Portfolio Insurance (SP)** is applied by adjusting the hedge ratio when long positions underperform relative to short positions.
   
### 4. **Performance Metrics**
   - The strategy calculates key metrics such as **CAGR**, **Sharpe Ratio**, **Sortino Ratio**, **Max Drawdown**, and **Calmar Ratio** to assess the strategy’s performance.
   
### 5. **Visualizations**
   - Several plots are generated to visualize:
     - Portfolio value over time.
     - Equity drawdowns.
     - Daily returns on assets.
     - Portfolio holdings breakdown.

## How to Run the Strategy

1. Clone this repository.
2. Ensure all dependencies are installed (see `requirements.txt`).
```bash
pip install -r requirements.txt
```
3. Load data files or ensure access to `yfinance` API for traditional assets.
4. Run the Jupyter notebook or Python script to execute the strategy, backtest, and visualize results.

## Improvements & Future Work

- **Additional Factors**: Incorporate volatility, liquidity, or sentiment data to enhance factor-based selection.
- **Risk Management**: Explore advanced techniques like Value at Risk (VaR) or Conditional Value at Risk (CVaR).
- **Derivatives**: Consider adding crypto derivatives (options, futures) to hedge or leverage positions.
- **Broader Asset Selection**: Expand the strategy to include more cryptocurrencies and traditional assets for better diversification.

![Grape Investment GIF](wowgrape.gif)
