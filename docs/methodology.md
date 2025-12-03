# Strategy Methodology

## 1. Introduction
This project implements a quantitative investment strategy using multiple valuation factors to select stocks from the CSI 300 index.

## 2. Factor Selection
The strategy uses five factors:
1. P/E Ratio (TTM) - Valuation factor
2. 30-Day Volatility - Risk factor  
3. Market Capitalization - Size factor
4. Dividend Yield - Income factor
5. Profit Growth - Quality factor

## 3. Data Processing
- Data source: Baostock API
- Stock universe: CSI 300 constituents
- Period: 2020-01-01 to 2024-12-31
- Data cleaning: Remove stocks with insufficient data

## 4. Portfolio Construction
- Rebalancing: Every 2 months
- Stock selection: Top 10% within each industry
- Minimum holdings: 25 stocks
- Industry neutralization: Yes

## 5. Risk Management
- Stop loss: -5%
- Maximum drawdown protection: -15%
- Transaction costs: 0.2% per trade