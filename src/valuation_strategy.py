import baostock as bs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, norm
import pickle
import os
from datetime import datetime
import time
import logging
import seaborn as sns

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ValuationStrategy:
    """估值多因子选股策略"""
    
    def __init__(self):
        self.start_date = '2020-01-01'
        self.end_date = '2024-12-31'
        self.top_pct = 0.10
        self.transaction_cost = 0.002
        self.min_stocks = 25
        self.winsor_lo = 0.05
        self.winsor_hi = 0.95
        self.adjust_freq = '2ME'
        self.cache_path = 'valuation_strategy_cache.pkl'
        
        self.factor_config = {
            'pe_ttm': {'weight': 0.30, 'direction': -1},
            'vol_30d': {'weight': 0.25, 'direction': -1},
            'size': {'weight': 0.20, 'direction': -1},
            'div_yield': {'weight': 0.15, 'direction': 1},
            'profit_grow': {'weight': 0.10, 'direction': 1},
        }
        
        self.stop_loss_threshold = -0.05
        self.max_drawdown_threshold = -0.15
        
        self.lg = None
        self.prices = {}
        self.fundamentals = {}
        self.industry_map = {}

    def initialize(self):
        """初始化数据"""
        cache_data = self._load_cache()
        
        if cache_data:
            self.prices, self.fundamentals, self.industry_map = cache_data
            self.lg = self._login_baostock()
            benchmark = self._get_benchmark()
            self._logout_baostock()
        else:
            self.lg = self._login_baostock()
            stocks = self._get_hs300_constituents()
            logger.info(f"HS300 constituents: {len(stocks)}")
            
            self.prices, self.fundamentals = self._batch_fetch_data(stocks)
            self.industry_map = self._get_industry_info(list(self.prices.keys()))
            benchmark = self._get_benchmark()
            
            self._save_cache((self.prices, self.fundamentals, self.industry_map))
            self._logout_baostock()
        
        return benchmark

    def _login_baostock(self):
        """登录Baostock"""
        for attempt in range(3):
            lg = bs.login()
            if lg.error_code == '0':
                logger.info("Baostock login successful")
                return lg
            time.sleep(2)
        raise ConnectionError("Failed to connect to Baostock")

    def _logout_baostock(self):
        """退出登录"""
        if self.lg:
            bs.logout()
            logger.info("Baostock connection closed")

    def _get_hs300_constituents(self, date=None):
        """获取沪深300成分股"""
        if date is None:
            date = self.start_date
        rs = bs.query_hs300_stocks(date=date)
        if rs.error_code == '0':
            df = rs.get_data()
            return df['code'].tolist()
        return []

    def _fetch_price_data(self, code):
        """获取价格数据"""
        fields = "date,close,volume,turn"
        rs = bs.query_history_k_data_plus(
            code, fields,
            start_date=self.start_date, 
            end_date=self.end_date,
            frequency="d", 
            adjustflag="3"
        )
        
        if rs.error_code != '0':
            return None
            
        df = rs.get_data()
        if df.empty:
            return None
            
        df['date'] = pd.to_datetime(df['date'])
        numeric_cols = ['close', 'volume', 'turn']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=['date', 'close'])
        
        return df.set_index('date').sort_index()

    def _fetch_fundamental_data(self, code):
        """获取基本面数据"""
        # 获取估值数据
        rs = bs.query_history_k_data_plus(
            code, "date,peTTM,pbMRQ,psTTM",
            start_date=self.start_date,
            end_date=self.end_date,
            frequency="d", 
            adjustflag="3"
        )
        
        if rs.error_code != '0':
            return None
            
        df = rs.get_data()
        if df.empty:
            return None
            
        df['date'] = pd.to_datetime(df['date'])
        for col in ['peTTM', 'pbMRQ', 'psTTM']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        fund_df = df.set_index('date')[['peTTM', 'pbMRQ', 'psTTM']]
        
        # 获取股息率数据
        div_data = self._fetch_dividend_data(code)
        if div_data is not None:
            fund_df = fund_df.join(div_data, how='left')
            
        # 获取盈利增长数据
        growth_data = self._fetch_growth_data(code)
        if growth_data is not None:
            fund_df = fund_df.join(growth_data, how='left')
            
        fund_df = fund_df.ffill(limit=30).bfill(limit=5)
        return fund_df.sort_index()

    def _fetch_dividend_data(self, code):
        """获取股息数据"""
        try:
            current_year = datetime.now().year
            div_records = []
            
            for year in range(current_year-2, current_year+1):
                rs = bs.query_dividend_data(code=code, year=str(year), yearType="report")
                if rs.error_code == '0':
                    df = rs.get_data()
                    if not df.empty and 'dividendRatio' in df.columns:
                        df['report_date'] = pd.to_datetime(df['reportDate'])
                        df['div_yield'] = pd.to_numeric(df['dividendRatio'], errors='coerce')
                        div_records.append(df[['report_date', 'div_yield']])
            
            if div_records:
                div_df = pd.concat(div_records, ignore_index=True)
                return div_df.set_index('report_date')['div_yield'].rename('dvRatioTTM')
        except Exception as e:
            logger.warning(f"Failed to fetch dividend data for {code}: {e}")
            
        return None

    def _fetch_growth_data(self, code):
        """获取增长数据"""
        try:
            current_year = datetime.now().year
            growth_records = []
            
            for year in range(current_year-2, current_year+1):
                for quarter in [1, 2, 3, 4]:
                    rs = bs.query_growth_data(code=code, year=str(year), quarter=str(quarter))
                    if rs.error_code == '0':
                        df = rs.get_data()
                        if not df.empty and 'netProfitGrowRate' in df.columns:
                            df['report_date'] = pd.to_datetime(df['statDate'])
                            df['profit_grow'] = pd.to_numeric(df['netProfitGrowRate'], errors='coerce')
                            growth_records.append(df[['report_date', 'profit_grow']])
            
            if growth_records:
                growth_df = pd.concat(growth_records, ignore_index=True)
                return growth_df.set_index('report_date')['profit_grow'].rename('netProfitGrowthRate')
        except Exception as e:
            logger.warning(f"Failed to fetch growth data for {code}: {e}")
            
        return None

    def _batch_fetch_data(self, stocks):
        """批量获取数据"""
        prices = {}
        fundamentals = {}
        
        for i in range(0, len(stocks), 15):
            batch = stocks[i:i+15]
            logger.info(f"Fetching batch {i//15 + 1}, size: {len(batch)}")
            
            for code in batch:
                try:
                    price_data = self._fetch_price_data(code)
                    if price_data is not None and len(price_data) > 100:
                        prices[code] = price_data
                        
                    fund_data = self._fetch_fundamental_data(code)
                    if fund_data is not None and len(fund_data) > 50:
                        fundamentals[code] = fund_data
                        
                except Exception as e:
                    logger.error(f"Error fetching data for {code}: {e}")
                    
                time.sleep(0.5)
                
            if i + 15 < len(stocks):
                time.sleep(6)
                
        return prices, fundamentals

    def _get_industry_info(self, stocks):
        """获取行业信息"""
        industry_map = {}
        for code in stocks:
            try:
                rs = bs.query_stock_basic(code=code)
                if rs.error_code == '0':
                    df = rs.get_data()
                    if not df.empty and 'industry' in df.columns:
                        industry_map[code] = df['industry'].iloc[0]
                    else:
                        industry_map[code] = 'Unknown'
            except Exception as e:
                logger.warning(f"Failed to get industry for {code}: {e}")
                industry_map[code] = 'Unknown'
                
            time.sleep(0.2)
            
        return industry_map

    def _get_benchmark(self):
        """获取基准数据"""
        try:
            rs = bs.query_history_k_data_plus(
                "sh.000300", "date,close",
                start_date=self.start_date,
                end_date=self.end_date,
                frequency="d",
                adjustflag="3"
            )
            
            if rs.error_code != '0':
                return None
                
            df = rs.get_data()
            if df.empty:
                return None
                
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['date'] = pd.to_datetime(df['date'])
            df = df.dropna().set_index('date')
            
            bench_daily = df['close'].pct_change()
            bench_period = bench_daily.resample(self.adjust_freq).apply(
                lambda x: (1 + x).prod() - 1
            )
            
            return bench_period.iloc[1:]
            
        except Exception as e:
            logger.error(f"Failed to get benchmark data: {e}")
            return None

    def build_factors(self):
        """构建因子"""
        valid_codes = [
            code for code in self.prices 
            if code in self.fundamentals and len(self.prices[code]) > 150
        ]
        
        if len(valid_codes) < self.min_stocks:
            raise ValueError(f"Insufficient valid stocks: {len(valid_codes)}")
            
        logger.info(f"Building factors for {len(valid_codes)} stocks")
        
        # 价格面板
        price_panel = pd.concat(
            [self.prices[code]['close'].rename(code) for code in valid_codes], 
            axis=1
        )
        returns_daily = price_panel.pct_change(fill_method=None)
        
        factors = {}
        
        # PE因子
        pe_data = []
        for code in valid_codes:
            if 'peTTM' in self.fundamentals[code].columns:
                pe_series = pd.to_numeric(
                    self.fundamentals[code]['peTTM'], errors='coerce'
                )
                pe_series = pe_series[(pe_series > 0) & (pe_series < 100)]
                pe_data.append(pe_series.rename(code))
                
        if pe_data:
            factors['pe_ttm'] = pd.concat(pe_data, axis=1).reindex(price_panel.index)
            factors['pe_ttm'] = factors['pe_ttm'].ffill(limit=10)
        else:
            factors['pe_ttm'] = pd.DataFrame(index=price_panel.index, columns=valid_codes)
        
        # 波动率因子
        factors['vol_30d'] = -returns_daily.rolling(30, min_periods=20).std()
        
        # 市值因子
        factors['size'] = -np.log(price_panel.rolling(30, min_periods=20).mean())
        
        # 股息率因子
        div_data = []
        for code in valid_codes:
            if 'dvRatioTTM' in self.fundamentals[code].columns:
                div_series = pd.to_numeric(
                    self.fundamentals[code]['dvRatioTTM'], errors='coerce'
                )
                div_series = div_series[div_series >= 0]
                div_data.append(div_series.rename(code))
                
        if div_data:
            factors['div_yield'] = pd.concat(div_data, axis=1).reindex(price_panel.index)
            factors['div_yield'] = factors['div_yield'].ffill(limit=10)
        else:
            factors['div_yield'] = pd.DataFrame(index=price_panel.index, columns=valid_codes)
        
        # 盈利增长因子
        profit_data = []
        for code in valid_codes:
            if 'netProfitGrowthRate' in self.fundamentals[code].columns:
                profit_series = pd.to_numeric(
                    self.fundamentals[code]['netProfitGrowthRate'], errors='coerce'
                )
                profit_data.append(profit_series.rename(code))
                
        if profit_data:
            factors['profit_grow'] = pd.concat(profit_data, axis=1).reindex(price_panel.index)
            factors['profit_grow'] = factors['profit_grow'].ffill(limit=10)
        else:
            factors['profit_grow'] = pd.DataFrame(index=price_panel.index, columns=valid_codes)
        
        return factors, price_panel, returns_daily

    def standardize_factors(self, factors):
        """因子标准化"""
        factor_z = {}
        
        for factor_name, factor_data in factors.items():
            if factor_name not in self.factor_config or factor_data.empty:
                continue
                
            # 清理数据
            df_clean = factor_data.apply(pd.to_numeric, errors='coerce')
            
            # 异常值处理
            for date in df_clean.index:
                values = df_clean.loc[date].dropna()
                if len(values) < 15:
                    continue
                    
                low_quantile = values.quantile(self.winsor_lo)
                high_quantile = values.quantile(self.winsor_hi)
                df_clean.loc[date] = values.clip(lower=low_quantile, upper=high_quantile)
            
            # 标准化
            standardized_data = []
            for date in df_clean.index:
                historical_data = df_clean.loc[:date].tail(252)
                if len(historical_data) < 20:
                    continue
                    
                current_values = df_clean.loc[date].dropna()
                if current_values.empty:
                    continue
                    
                hist_mean = historical_data.mean()
                hist_std = historical_data.std()
                
                common_codes = current_values.index.intersection(hist_mean.index)
                if len(common_codes) < 10:
                    continue
                    
                z_scores = (current_values[common_codes] - hist_mean[common_codes]) / (hist_std[common_codes] + 1e-8)
                standardized_data.append(pd.Series(z_scores, name=date))
            
            if standardized_data:
                z_df = pd.concat(standardized_data, axis=1).T.sort_index()
                direction = self.factor_config[factor_name]['direction']
                factor_z[factor_name] = z_df.shift(2) * direction
                
        return factor_z

    def _group_by_industry(self, codes):
        """按行业分组"""
        groups = {}
        for code in codes:
            industry = self.industry_map.get(code, 'Unknown')
            groups.setdefault(industry, []).append(code)
        return groups

    def calculate_factor_ic(self, factor_z, period_returns, period_dates):
        """计算因子IC"""
        ic_table = pd.DataFrame(index=period_dates, columns=list(factor_z.keys()))
        
        for factor_name, factor_data in factor_z.items():
            for date in period_dates:
                if date not in factor_data.index or date not in period_returns.index:
                    continue
                    
                factor_values = factor_data.loc[date].dropna()
                return_values = period_returns.loc[date].dropna()
                
                common_codes = factor_values.index.intersection(return_values.index)
                if len(common_codes) < 15:
                    continue
                    
                try:
                    ic_value, _ = spearmanr(factor_values[common_codes], return_values[common_codes])
                    ic_table.loc[date, factor_name] = ic_value
                except:
                    ic_table.loc[date, factor_name] = np.nan
                    
        return ic_table

    def backtest(self, factor_z, price_panel):
        """策略回测"""
        period_prices = price_panel.resample(self.adjust_freq).last()
        period_returns = period_prices.pct_change(fill_method=None).shift(-1)
        period_dates = period_returns.index[:-1]
        
        # 因子IC分析
        ic_table = self.calculate_factor_ic(factor_z, period_returns, period_dates)
        ic_table = ic_table.apply(pd.to_numeric, errors='coerce')
        
        if not ic_table.empty:
            logger.info("Factor IC analysis:")
            ic_stats = pd.DataFrame({
                "Mean_IC": ic_table.mean().round(4),
                "IC_Win_Rate": ((ic_table > 0).sum() / ic_table.notna().sum()).round(4),
            })
            for factor, stats in ic_stats.iterrows():
                status = "Good" if stats['Mean_IC'] > 0.01 and stats['IC_Win_Rate'] > 0.55 else "Weak"
                logger.info(f"{factor}: IC={stats['Mean_IC']}, WinRate={stats['IC_Win_Rate']} [{status}]")
        
        # 回测主循环
        portfolio_returns = []
        portfolio_dates = []
        portfolio_holdings = []
        previous_holdings = set()
        cumulative_nav = 1.0
        max_nav = 1.0
        in_drawdown_protection = False
        
        for i, current_date in enumerate(period_dates[:-1]):
            # 检查回撤状态
            current_drawdown = (cumulative_nav - max_nav) / max_nav
            if current_drawdown < self.max_drawdown_threshold and not in_drawdown_protection:
                in_drawdown_protection = True
                logger.warning(f"Entering drawdown protection: {current_drawdown:.2%}")
                
            if in_drawdown_protection and cumulative_nav >= max_nav * (1 - self.max_drawdown_threshold / 2):
                in_drawdown_protection = False
                logger.info("Exiting drawdown protection")
            
            # 投资组合构建
            if in_drawdown_protection:
                current_holdings = list(previous_holdings)[:max(5, len(previous_holdings)//2)] if previous_holdings else []
            else:
                # 计算综合得分
                composite_score = pd.Series(0.0, index=price_panel.columns)
                total_weight = 0
                
                for factor_name, config in self.factor_config.items():
                    if factor_name in factor_z and current_date in factor_z[factor_name].index:
                        factor_scores = factor_z[factor_name].loc[current_date].dropna()
                        if not factor_scores.empty:
                            weight = config['weight']
                            composite_score = composite_score.add(factor_scores * weight, fill_value=0)
                            total_weight += weight
                
                if total_weight > 0:
                    composite_score /= total_weight
                    
                composite_score = composite_score.dropna()
                
                # 行业中性选股
                current_holdings = []
                industry_groups = self._group_by_industry(composite_score.index)
                
                for industry, codes in industry_groups.items():
                    industry_scores = composite_score.loc[codes].dropna()
                    if len(industry_scores) < 5:
                        continue
                        
                    n_select = max(1, int(len(industry_scores) * self.top_pct))
                    industry_mean = industry_scores.mean()
                    industry_std = industry_scores.std()
                    
                    # 选择显著高于平均的股票
                    qualified_stocks = industry_scores[industry_scores > industry_mean + 0.2 * industry_std]
                    if len(qualified_stocks) > 0:
                        selected_stocks = qualified_stocks.nlargest(min(n_select, len(qualified_stocks))).index.tolist()
                        current_holdings.extend(selected_stocks)
                
                current_holdings = list(set(current_holdings))
                
                # 确保最小持仓数量
                if len(current_holdings) < self.min_stocks:
                    all_scores = composite_score.sort_values(ascending=False)
                    needed = self.min_stocks - len(current_holdings)
                    candidate_stocks = all_scores[~all_scores.index.isin(current_holdings)].head(needed * 2)
                    
                    if not candidate_stocks.empty:
                        candidate_mean = candidate_stocks.mean()
                        good_candidates = candidate_stocks[candidate_stocks > candidate_mean].head(needed)
                        current_holdings.extend(good_candidates.index.tolist())
            
            # 计算期间收益
            next_period_date = period_returns.index[i+1]
            try:
                holding_returns = period_returns.loc[next_period_date, current_holdings].dropna()
                if len(holding_returns) < max(8, self.min_stocks//3):
                    if previous_holdings:
                        current_holdings = list(previous_holdings)
                        holding_returns = period_returns.loc[next_period_date, current_holdings].dropna()
                    else:
                        continue
            except:
                continue
            
            # 交易成本
            if not previous_holdings:
                trading_cost = self.transaction_cost * 2
            else:
                turnover = len(set(current_holdings) ^ previous_holdings) / max(len(current_holdings), 1)
                trading_cost = turnover * self.transaction_cost
            
            # 组合收益
            portfolio_return = holding_returns.mean()
            
            # 止损逻辑
            if cumulative_nav * (1 + portfolio_return) < max_nav * (1 + self.stop_loss_threshold):
                portfolio_return = max(portfolio_return, self.stop_loss_threshold)
            
            # 极端损失控制
            extreme_loss_count = (holding_returns < -0.08).sum()
            if extreme_loss_count > len(holding_returns) * 0.2:
                portfolio_return = max(portfolio_return, -0.03)
            
            net_return = portfolio_return - trading_cost
            
            portfolio_returns.append(net_return)
            portfolio_dates.append(next_period_date)
            portfolio_holdings.append(current_holdings)
            previous_holdings = set(current_holdings)
            
            # 更新净值
            cumulative_nav *= (1 + net_return)
            max_nav = max(max_nav, cumulative_nav)
        
        return pd.Series(portfolio_returns, index=portfolio_dates), ic_table, portfolio_holdings

    def calculate_performance_metrics(self, returns, benchmark_returns=None):
        """计算绩效指标"""
        if self.adjust_freq == '2M':
            periods_per_year = 6
        else:
            periods_per_year = 12
            
        total_periods = len(returns)
        years = total_periods / periods_per_year
        
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        annual_volatility = returns.std() * np.sqrt(periods_per_year) if len(returns) > 1 else 0
        
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 1e-8 else 0
        
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() if not drawdowns.empty else 0
        
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        win_rate = (returns > 0).mean() if len(returns) > 0 else 0
        
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
        avg_loss = losing_returns.mean() if len(losing_returns) > 0 else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        metrics = {
            'Total_Return': total_return,
            'Annual_Return': annual_return,
            'Annual_Volatility': annual_volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Calmar_Ratio': calmar_ratio,
            'Win_Rate': win_rate,
            'Profit_Loss_Ratio': profit_loss_ratio,
        }
        
        if benchmark_returns is not None:
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
            if len(aligned_returns) > 0:
                excess_returns = aligned_returns - aligned_benchmark
                annual_excess = excess_returns.mean() * periods_per_year
                tracking_error = excess_returns.std() * np.sqrt(periods_per_year) if len(excess_returns) > 1 else 0
                information_ratio = annual_excess / tracking_error if tracking_error > 1e-8 else 0
                
                metrics['Annual_Excess_Return'] = annual_excess
                metrics['Information_Ratio'] = information_ratio
                metrics['Excess_Win_Rate'] = (excess_returns > 0).mean()
        
        return metrics

    def plot_results(self, cumulative_nav, ic_table=None, benchmark_nav=None):
        """绘制结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 净值曲线
        axes[0, 0].plot(cumulative_nav.index, cumulative_nav, label='Strategy', linewidth=2)
        if benchmark_nav is not None:
            axes[0, 0].plot(benchmark_nav.index, benchmark_nav, label='HS300', alpha=0.8)
        axes[0, 0].set_title('Cumulative NAV')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 回撤曲线
        rolling_max = cumulative_nav.expanding().max()
        drawdown = (cumulative_nav - rolling_max) / rolling_max
        axes[0, 1].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].plot(drawdown.index, drawdown, color='red')
        axes[0, 1].axhline(self.max_drawdown_threshold, color='orange', linestyle='--', 
                          label=f'Drawdown Threshold ({self.max_drawdown_threshold:.1%})')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 因子IC热力图
        if ic_table is not None and not ic_table.empty:
            yearly_ic = ic_table.resample('YE').mean()
            if not yearly_ic.empty and not yearly_ic.isnull().all().all():
                im = axes[1, 0].imshow(yearly_ic.T, cmap='RdYlBu', aspect='auto', 
                                      vmin=-0.05, vmax=0.05)
                axes[1, 0].set_yticks(range(len(yearly_ic.columns)))
                axes[1, 0].set_yticklabels(yearly_ic.columns)
                axes[1, 0].set_xticks(range(len(yearly_ic.index)))
                axes[1, 0].set_xticklabels([d.year for d in yearly_ic.index])
                plt.colorbar(im, ax=axes[1, 0])
                axes[1, 0].set_title('Yearly Factor IC Heatmap')
        
        # 收益分布
        monthly_returns = cumulative_nav.resample('ME').last().ffill().pct_change(fill_method=None).dropna()
        axes[1, 1].hist(monthly_returns, bins=15, alpha=0.7, color='skyblue', 
                       edgecolor='black', density=True)
        
        x = np.linspace(monthly_returns.min(), monthly_returns.max(), 100)
        mu, std = norm.fit(monthly_returns)
        axes[1, 1].plot(x, norm.pdf(x, mu, std), 'r-', 
                       label=f'Normal (μ={mu:.2%}, σ={std:.2%})')
        axes[1, 1].axvline(monthly_returns.mean(), color='green', linestyle='--',
                          label=f'Mean: {monthly_returns.mean():.2%}')
        axes[1, 1].set_title('Monthly Return Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs('results/images', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        plt.savefig(f'results/images/performance_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'results/images/performance_{timestamp}.pdf', bbox_inches='tight')
        
        plt.show()
        
        print(f"图表已保存到 results/images/ 文件夹")

    def _load_cache(self):
        """加载缓存数据"""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None

    def _save_cache(self, data):
        """保存缓存数据"""
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info("Data cached successfully")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def run(self):
        """运行策略"""
        logger.info("Starting valuation multi-factor strategy")
        
        # 初始化数据
        benchmark_returns = self.initialize()
        
        # 筛选有效股票
        valid_codes = [
            code for code in self.prices 
            if len(self.prices[code]) > 150 and code in self.fundamentals
        ]
        
        if len(valid_codes) < self.min_stocks:
            raise ValueError(f"Insufficient valid stocks: {len(valid_codes)}")
            
        logger.info(f"Valid stocks: {len(valid_codes)}")
        
        # 构建因子
        filtered_prices = {code: self.prices[code] for code in valid_codes}
        filtered_fundamentals = {code: self.fundamentals[code] for code in valid_codes}
        
        factors, price_panel, _ = self.build_factors()
        
        # 因子标准化
        factor_z = self.standardize_factors(factors)
        if not factor_z:
            raise ValueError("No valid factors generated")
            
        # 回测
        strategy_returns, ic_table, holdings = self.backtest(factor_z, price_panel)
        
        if strategy_returns.empty:
            raise ValueError("Backtest produced no results")
            
        cumulative_nav = (1 + strategy_returns).cumprod()
        
        # 计算绩效指标
        metrics = self.calculate_performance_metrics(strategy_returns, benchmark_returns)
        
        # 输出结果
        logger.info("=" * 60)
        logger.info("STRATEGY PERFORMANCE REPORT")
        logger.info("=" * 60)
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                if any(x in metric.lower() for x in ['return', 'rate', 'drawdown']):
                    logger.info(f"{metric}: {value:.2%}")
                else:
                    logger.info(f"{metric}: {value:.3f}")
            else:
                logger.info(f"{metric}: {value}")
                
        logger.info("=" * 60)
        
        # 可视化
        benchmark_nav = None
        if benchmark_returns is not None:
            benchmark_nav = (1 + benchmark_returns).cumprod()
            benchmark_nav = benchmark_nav / benchmark_nav.iloc[0]
            
        self.plot_results(cumulative_nav, ic_table, benchmark_nav)
        
        # 保存结果
        self._save_results(strategy_returns, cumulative_nav, ic_table, holdings, metrics)
        
        logger.info("Strategy execution completed")

    def _save_results(self, returns, nav, ic_table, holdings, metrics):
        """保存结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        prefix = f"valuation_strategy_{timestamp}"
        
        returns.to_csv(f"{prefix}_returns.csv", header=['return'])
        nav.to_csv(f"{prefix}_nav.csv", header=['nav'])
        
        if ic_table is not None and not ic_table.empty:
            ic_table.to_csv(f"{prefix}_ic.csv")
            
        pd.DataFrame({
            'date': [pd.Timestamp(date) for date in returns.index],
            'holdings': holdings
        }).to_csv(f"{prefix}_holdings.csv", index=False)
        
        pd.DataFrame(list(metrics.items()), columns=['metric', 'value']).to_csv(
            f"{prefix}_metrics.csv", index=False
        )
        
        logger.info(f"Results saved with prefix: {prefix}")


if __name__ == "__main__":
    strategy = ValuationStrategy()
    strategy.run()