Valuation-Based Multi-Factor Equity Strategy for CSI300

A complete quantitative investment research project using Python, Baostock, and multi-factor modeling.

Overview

This project implements a valuation-driven multi-factor stock selection strategy for the CSI 300 universe.
The entire workflow—from data acquisition to factor engineering, IC analysis, industry-neutral portfolio construction, and backtesting—is completed in a single, fully-engineered Python pipeline.

This project demonstrates:

Quantitative research workflow
Factor construction and standardization
Industry-neutral stock selection
Portfolio backtesting
Performance evaluation
Data engineering and reproducibility

This repository contains the exact implementation used for my MFE/MFin applications.

Strategy Design

1. Universe

CSI 300 constituents
Daily frequency
Backtest period: 2020–2024

2. Factors Included

All factors are cleaned by winsorization + Z-score standardization + direction normalization.

Factor	        Description	                   Direction

PE_TTM	        Valuation factor	           Lower is better
Vol_30D	        30-day rolling volatility	   Lower is better
Size	        Log-market-cap proxy	       Smaller is better
Dividend Yield	dvRatioTTM	                   Higher is better
Profit Growth	YoY net profit growth	       Higher is better


Factor weights:

PE_TTM:        30%
Vol_30D:       25%
Size:          20%
Dividend Yield:15%
Profit Growth: 10%


Methodology

1. Data Collection

Using Baostock APIs:

Daily OHLCV

PE / PB / PS

Dividend ratio

Profit growth

Industry classification

2. Factor Engineering

Outlier removal (winsorize at 5%/95%)

Rolling historical mean/std standardization

Direction adjustment (+/-)

Factor combination using weighted linear aggregation

3. Portfolio Construction

Rebalance frequency: every 2 months

Industry-neutral selection

Select top 10% within each industry

Minimum 25 positions

Turnover-based transaction cost included

Stop-loss & drawdown control mechanism

4. Backtesting Framework

Forward-looking period returns

IC computation for every factor

Portfolio NAV path

Trading cost deduction

Extreme-loss control

Drawdown protection


Performance Metrics Reported

The script automatically computes:

Annualized return

Annualized volatility

Sharpe ratio

Calmar ratio

Maximum drawdown

Win rate

Profit/loss ratio

Excess return vs CSI300

Information ratio


And outputs:

NAV curve

Drawdown curve

Factor IC heatmap

Monthly returns distribution


Repository Structure（single-file version）

Your repository currently uses a single-file design:

valuation_strategy.py        # Full strategy pipeline (data, factors, backtest, visualization)
README.md       # Project description

The full pipeline is contained in one complete script for clarity and ease of execution.


How to Run

1. Install required packages:



pip install baostock pandas numpy matplotlib scipy

2. Run the strategy:



python valuation_strategy.py

3. Outputs appear as:



NAV curve PNG

IC table CSV

Holdings CSV

Returns CSV

Strategy metrics CSV


Skills Demonstrated

Multi-factor equity modeling

Python for quant research

Time-series data processing

Factor standardization methods

Industry-neutral portfolio construction

Backtesting & risk analysis

Visualization & reporting

Large-scale data cleaning

Reproducible research workflow


This project showcases the exact type of quantitative research skills required for MFE / MFin / Data Science graduate programs.


Contact

If you have any questions about this project, feel free to reach out.
