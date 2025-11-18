Valuation-Based Multi-Factor Equity Strategy for CSI 300

A full quantitative research pipeline: data → factors → IC analysis → portfolio construction → backtesting.


Overview

This project implements a valuation-driven multi-factor equity selection strategy for the CSI 300 universe.
It includes the complete quantitative research workflow:

Data acquisition via Baostock

Factor construction & cleaning

Z-score standardization

Industry-neutral stock selection

Portfolio backtesting with trading costs

Performance evaluation

Visualization & reporting


The entire strategy is implemented in one fully engineered Python script, suitable for use in quantitative finance graduate applications (MFE / MFin / Financial Engineering).


Strategy Design

1. Universe

CSI 300 constituents

Daily frequency

Backtest period: 2020–2024


2. Factors Included

All factors undergo winsorization, rolling Z-score normalization, and sign-direction adjustment.

Factor	Description	Direction

PE_TTM	Valuation (low PE preferred)	Lower is better
Vol_30D	30-day realized volatility	Lower is better
Size	Log market-cap proxy	Smaller is better
Dividend Yield	dvRatioTTM	Higher is better
Profit Growth	YoY net profit growth	Higher is better


Factor Weights

PE_TTM:         30%
Vol_30D:        25%
Size:           20%
Dividend Yield: 15%
Profit Growth:  10%


Methodology

1. Data Collection

Using Baostock APIs:

OHLCV

PE / PB / PS

Dividend payout

Profit growth

Industry classification


2. Factor Engineering

Winsorize each factor at 5% / 95%

Rolling-window mean & standard deviation normalization

Direction adjustment (+1 or –1)

Weighted linear combination into a composite multi-factor score


3. Portfolio Construction

Rebalance frequency: every 2 months

Industry-neutral selection: choose top 10% within each industry

Minimum holdings: 25 stocks

Transaction cost applied based on turnover

Stop-loss & drawdown control to prevent extreme downside


4. Backtesting Framework

The backtest includes:

Forward-looking returns

Per-period factor IC (Information Coefficient)

Portfolio NAV path

Extreme-loss filtering

Drawdown-based protection

Performance evaluation


Outputs include:

NAV curve

Drawdown curve

Factor IC heatmap

Monthly return distribution


Performance Metrics (Automatically Generated)

Total return

Annualized return

Annualized volatility

Sharpe ratio

Calmar ratio

Maximum drawdown

Win rate

Profit/loss ratio

Excess return vs CSI 300

Information ratio


Repository Structure

This version uses a single-file pipeline for clarity:

valuation_strategy.py   # End-to-end strategy pipeline
README.md               # Project documentation
results/                # Output charts and metrics (generated after running)


How to Run

1. Install required packages

pip install baostock pandas numpy matplotlib scipy

2. Run the full pipeline

python valuation_strategy.py

3. Outputs (saved automatically)

*_nav.csv – NAV time series

*_returns.csv – Period returns

*_ic.csv – Factor IC table

*_metrics.csv – Performance summary

*_holdings.csv – Holdings per rebalance

equity_curve.png – Net asset value curve


Skills Demonstrated

This project showcases core quantitative finance competencies:

Multi-factor model design

Equity cross-sectional research

Python for financial modeling

Time-series processing

Factor standardization & normalization

Industry-neutral stock selection

Backtesting methodology

Risk management (stop-loss, drawdown control)

Data engineering and reproducibility

Visualization and performance reporting


Ideal for MFE / MFin / Data Science graduate program applications.


Contact

Feel free to reach out for collaboration or discussion on quantitative strategies.
