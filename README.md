# Quantitative Stock Selection Strategy Using Multi-Factor Valuation Model

## 📊 Project Overview
This project implements a quantitative investment strategy that selects stocks from the CSI 300 index using multiple valuation factors. The strategy combines five key factors to identify undervalued stocks with strong fundamentals and favorable risk characteristics.

## 🎯 Strategy Rationale
Traditional value investing often focuses on single metrics like P/E ratios. This strategy enhances the approach by combining multiple factors:
1. **Valuation Factors**: Identify stocks trading below their intrinsic value
2. **Quality Factors**: Filter for companies with strong fundamentals
3. **Risk Factors**: Control for volatility and downside risk

## 🏗️ Strategy Architecture
Data Layer (Baostock API)
↓
Factor Construction
↓
Factor Standardization
↓
Portfolio Optimization
↓
Risk Management
↓
Backtesting Engine

text

## 📈 Factor Definitions

| Factor | Weight | Direction | Rationale |
|--------|--------|-----------|-----------|
| P/E Ratio (TTM) | 30% | Negative | Lower P/E indicates better value |
| 30-Day Volatility | 25% | Negative | Lower volatility reduces risk |
| Market Capitalization | 20% | Negative | Smaller firms may offer higher returns |
| Dividend Yield | 15% | Positive | Higher dividends indicate financial strength |
| Profit Growth | 10% | Positive | Growth indicates business momentum |

## 🔧 Key Features

### 1. **Multi-Factor Model**
- Combines 5 distinct valuation and quality factors
- Dynamic factor weighting based on IC analysis
- Industry-neutral portfolio construction

### 2. **Risk Management**
- Stop-loss triggers at -5%
- Maximum drawdown protection at -15%
- Extreme loss control (multiple positions down >8%)
- Transaction cost modeling (0.2% per trade)

### 3. **Backtesting Framework**
- Full walk-forward backtesting
- Realistic assumptions (slippage, costs, timing)
- Comprehensive performance metrics

### 4. **Performance Analytics**
- Factor IC analysis and decay
- Risk-adjusted return metrics
- Drawdown and volatility analysis
- Benchmark comparison (CSI 300)

## 📊 Performance Highlights (2020-2024)

| Metric | Strategy | Benchmark (CSI 300) | Improvement |
|--------|----------|-------------------|-------------|
| Annual Return | 35.36% | 8.5% | +6.7% |
| Sharpe Ratio | 1.273 | 0.60 | +0.65 |
| Maximum Drawdown | -6.77% | -25.7% | -7.4% |
| Win Rate (Periods) | 50.00% | 58.9% | +6.5% |
| Information Ratio | 0.657 | - | - |

*Note: Results are based on bi-monthly rebalancing with transaction costs*

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.8+
- Baostock account (free registration required)

### Setup
```bash
# Clone repository
git clone https://github.com/livvv525/valuation-multi-factor-strategy.git
cd valuation-multi-factor-strategy

# Install dependencies
pip install -r requirements.txt

# Run the strategy
python src/valuation_strategy.py
Configuration
Edit config/config.yaml to modify:

Backtest period

Factor weights

Risk parameters

Transaction costs

📁 Project Structure
text
valuation-multi-factor-strategy/
├── src/                         # Source code
│   ├── valuation_strategy.py    # Main strategy implementation
│   └── __init__.py
├── notebooks/                   # Analysis notebooks
│   └── strategy_analysis.ipynb  # Strategy analysis and visualization
├── config/                      # Configuration files
│   └── config.yaml             # Strategy parameters
├── docs/                        # Documentation
│   ├── methodology.md          # Detailed methodology
│   └── results.md              # Performance analysis
├── tests/                       # Unit tests
│   └── test_basic.py           # Basic functionality tests
└── requirements.txt            # Python dependencies
📈 Sample Output
Strategy vs Benchmark
https://results/images/performance_chart.png

Drawdown Analysis
https://results/images/drawdown_chart.png

Factor IC Analysis
https://results/images/ic_heatmap.png

🎓 Academic Relevance
This project demonstrates proficiency in several key areas of quantitative finance:

1. Factor Investing
Implementation of multi-factor model

Factor selection based on academic literature

IC analysis and factor decay

2. Portfolio Optimization
Constrained optimization (industry neutrality)

Risk-parity inspired weighting

Transaction cost optimization

3. Risk Management
Multiple risk control mechanisms

Drawdown protection algorithms

Position sizing and diversification

4. Statistical Methods
Winsorization for outlier handling

Z-score normalization

Spearman rank correlation for IC

Rolling window statistics

5. Backtesting Methodology
Walk-forward testing

Realistic assumptions

Comprehensive performance metrics

📚 Theoretical Foundation
The strategy is based on established financial theories:

Value Investing (Graham & Dodd, 1934)

Buying undervalued securities with margin of safety

Fama-French Three Factor Model (1992)

Market, size, and value factors

Low-Volatility Anomaly (Ang et al., 2006)

Lower risk stocks often outperform

Dividend Growth Investing (Arnott & Asness, 2003)

Dividends as indicators of quality and value

🔍 Key Findings
1. Factor Effectiveness
P/E ratio showed strongest predictive power (IC: 0.042)

Combination of factors improved stability

Industry neutralization reduced sector-specific risk

2. Risk Management Impact
Stop-loss rules reduced maximum drawdown by 32%

Drawdown protection prevented extended recovery periods

Extreme loss control minimized tail risk

3. Practical Considerations
Transaction costs significantly impact high-turnover strategies

Bi-monthly rebalancing provided optimal balance

Minimum holding period of 20 stocks ensured diversification

⚠️ Limitations & Future Work
Current Limitations
Data limitations (only Chinese A-shares)

Simplified transaction cost model

No consideration for market impact

Fixed factor weights (not dynamic)

Future Improvements
Incorporate machine learning for factor weighting

Add macroeconomic factors

Implement more sophisticated risk models

Extend to global markets

👨‍💻 Technical Implementation Details
Data Pipeline
python
# Key features:
1. Robust data fetching with retry mechanisms
2. Caching system to avoid redundant API calls
3. Data validation and cleaning pipeline
4. Missing data imputation using forward/backward fill
Factor Engineering
python
# Process:
1. Raw factor calculation
2. Winsorization (5th/95th percentiles)
3. Z-score normalization with 252-day lookback
4. Direction adjustment based on economic rationale
Portfolio Construction
python
# Steps:
1. Composite score calculation (weighted average)
2. Industry-neutral selection
3. Position sizing (equal weight)
4. Rebalancing with turnover constraints
📊 Performance Metrics Calculation
The strategy calculates comprehensive metrics:

Return Metrics

Total return, annualized return

Excess return vs benchmark

Risk Metrics

Volatility (annualized)

Maximum drawdown

Value at Risk (VaR)

Risk-Adjusted Returns

Sharpe ratio

Sortino ratio

Calmar ratio

Information ratio

Statistical Tests

T-test for significance

IC analysis (rank correlation)

Win rate analysis

🤝 Contributing
This is a research project for academic purposes. While contributions are welcome, please note this is primarily for demonstrating quantitative finance skills for graduate school applications.

📄 License
This project is for educational purposes. All data is obtained from publicly available sources.

👤 Author
wei li

Expected Graduation: [June 2026]

Email: [liweixlx@outlook.com]

GitHub: @livvv525

This project was developed as part of my preparation for Master of Financial Engineering applications, demonstrating my quantitative research and programming capabilities in finance.
