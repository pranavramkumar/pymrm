"""
Portfolio Metrics: Comprehensive Portfolio Analysis and Performance Measurement Framework

This module provides extensive portfolio analysis functionality including:
- Performance metrics: Sharpe ratio, Sortino ratio, coefficient of variation
- Portfolio construction: weighted returns, variance, standard deviation calculations
- CAPM analysis: expected returns, beta calculations, security market line
- Beta estimation: variance-covariance method, correlation method, rolling beta
- Portfolio optimization: efficient frontier, risk-return analysis
- Performance attribution and risk decomposition
- Multi-factor models and advanced portfolio analytics

Author: Claude AI
Created: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from enum import Enum
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RebalancingFrequency(Enum):
    """Portfolio rebalancing frequency enumeration."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"


class RiskMeasure(Enum):
    """Risk measure type enumeration."""
    STANDARD_DEVIATION = "standard_deviation"
    DOWNSIDE_DEVIATION = "downside_deviation"
    VALUE_AT_RISK = "value_at_risk"
    CONDITIONAL_VAR = "conditional_var"
    MAX_DRAWDOWN = "max_drawdown"


class BenchmarkType(Enum):
    """Benchmark type enumeration."""
    MARKET_INDEX = "market_index"
    RISK_FREE_RATE = "risk_free_rate"
    CUSTOM_PORTFOLIO = "custom_portfolio"
    PEER_GROUP = "peer_group"


@dataclass
class PortfolioConfig:
    """Configuration class for portfolio calculations."""

    # General settings
    precision: int = 6
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])

    # Data settings
    data_frequency: str = 'daily'  # daily, weekly, monthly
    lookback_periods: int = 252    # trading days
    min_periods: int = 30          # minimum periods for calculations

    # Risk-free rate settings
    risk_free_rate: float = 0.02   # annual risk-free rate
    risk_free_proxy: str = '^TNX'  # 10-year Treasury yield

    # CAPM settings
    market_proxy: str = '^GSPC'    # S&P 500 as market proxy
    beta_lookback: int = 252       # periods for beta calculation
    rolling_window: int = 63       # rolling window for time-varying beta

    # Portfolio optimization settings
    max_weight: float = 0.4        # maximum position size
    min_weight: float = 0.01       # minimum position size
    target_return: float = 0.12    # target annual return
    rebalancing_threshold: float = 0.05  # rebalancing threshold

    # Performance settings
    benchmark_symbol: str = '^GSPC'
    downside_threshold: float = 0.0  # threshold for downside deviation

    # Plotting settings
    plot_style: str = 'seaborn-v0_8'
    figure_size: Tuple[int, int] = (15, 10)
    color_palette: str = 'viridis'


class PerformanceMetrics:
    """
    Portfolio performance metrics calculations.
    """

    def __init__(self, config: PortfolioConfig = None):
        self.config = config or PortfolioConfig()

    def coefficient_of_variation(self, returns: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Calculate coefficient of variation for risk-adjusted return analysis.

        Parameters:
        -----------
        returns : Series or array
            Return series

        Returns:
        --------
        Dict with coefficient of variation analysis
        """
        try:
            # Convert to numpy array
            if isinstance(returns, pd.Series):
                returns_array = returns.dropna().values
            else:
                returns_array = np.array(returns)
                returns_array = returns_array[~np.isnan(returns_array)]

            if len(returns_array) == 0:
                return {'error': 'No valid returns data'}

            # Calculate basic statistics
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array, ddof=1)

            # Coefficient of variation
            if mean_return != 0:
                cv = std_return / abs(mean_return)
            else:
                cv = float('inf') if std_return > 0 else 0

            # Annualized values
            periods_per_year = self._get_periods_per_year()
            annualized_return = mean_return * periods_per_year
            annualized_volatility = std_return * np.sqrt(periods_per_year)
            annualized_cv = annualized_volatility / abs(annualized_return) if annualized_return != 0 else float('inf')

            # Risk-return efficiency interpretation
            efficiency_rating = self._interpret_cv(cv)

            return {
                'coefficient_of_variation': round(cv, self.config.precision),
                'mean_return': round(mean_return, self.config.precision),
                'standard_deviation': round(std_return, self.config.precision),
                'annualized_return': round(annualized_return, self.config.precision),
                'annualized_volatility': round(annualized_volatility, self.config.precision),
                'annualized_cv': round(annualized_cv, self.config.precision),
                'efficiency_rating': efficiency_rating,
                'sample_size': len(returns_array),
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def sharpe_ratio(self, returns: Union[pd.Series, np.ndarray],
                    risk_free_rate: float = None) -> Dict[str, Any]:
        """
        Calculate Sharpe ratio for risk-adjusted performance.

        Parameters:
        -----------
        returns : Series or array
            Return series
        risk_free_rate : float
            Risk-free rate (annual)

        Returns:
        --------
        Dict with Sharpe ratio analysis
        """
        try:
            if risk_free_rate is None:
                risk_free_rate = self.config.risk_free_rate

            # Convert to numpy array
            if isinstance(returns, pd.Series):
                returns_array = returns.dropna().values
            else:
                returns_array = np.array(returns)
                returns_array = returns_array[~np.isnan(returns_array)]

            if len(returns_array) == 0:
                return {'error': 'No valid returns data'}

            # Calculate excess returns
            periods_per_year = self._get_periods_per_year()
            period_risk_free = risk_free_rate / periods_per_year
            excess_returns = returns_array - period_risk_free

            # Calculate Sharpe ratio
            mean_excess_return = np.mean(excess_returns)
            std_excess_return = np.std(excess_returns, ddof=1)

            if std_excess_return > 0:
                sharpe_ratio = mean_excess_return / std_excess_return
                annualized_sharpe = sharpe_ratio * np.sqrt(periods_per_year)
            else:
                sharpe_ratio = 0
                annualized_sharpe = 0

            # Additional statistics
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array, ddof=1)

            # Confidence interval for Sharpe ratio
            sharpe_ci = self._calculate_sharpe_confidence_interval(
                returns_array, risk_free_rate, 0.95
            )

            # Performance interpretation
            performance_rating = self._interpret_sharpe_ratio(annualized_sharpe)

            return {
                'sharpe_ratio': round(sharpe_ratio, self.config.precision),
                'annualized_sharpe_ratio': round(annualized_sharpe, self.config.precision),
                'mean_return': round(mean_return, self.config.precision),
                'mean_excess_return': round(mean_excess_return, self.config.precision),
                'standard_deviation': round(std_return, self.config.precision),
                'risk_free_rate': round(risk_free_rate, self.config.precision),
                'annualized_return': round(mean_return * periods_per_year, self.config.precision),
                'annualized_volatility': round(std_return * np.sqrt(periods_per_year), self.config.precision),
                'sharpe_confidence_interval': {
                    'lower': round(sharpe_ci[0], self.config.precision),
                    'upper': round(sharpe_ci[1], self.config.precision)
                },
                'performance_rating': performance_rating,
                'sample_size': len(returns_array),
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def sortino_ratio(self, returns: Union[pd.Series, np.ndarray],
                     risk_free_rate: float = None,
                     target_return: float = None) -> Dict[str, Any]:
        """
        Calculate Sortino ratio focusing on downside risk.

        Parameters:
        -----------
        returns : Series or array
            Return series
        risk_free_rate : float
            Risk-free rate (annual)
        target_return : float
            Target return threshold

        Returns:
        --------
        Dict with Sortino ratio analysis
        """
        try:
            if risk_free_rate is None:
                risk_free_rate = self.config.risk_free_rate

            if target_return is None:
                target_return = self.config.downside_threshold

            # Convert to numpy array
            if isinstance(returns, pd.Series):
                returns_array = returns.dropna().values
            else:
                returns_array = np.array(returns)
                returns_array = returns_array[~np.isnan(returns_array)]

            if len(returns_array) == 0:
                return {'error': 'No valid returns data'}

            # Calculate excess returns over target
            periods_per_year = self._get_periods_per_year()
            period_risk_free = risk_free_rate / periods_per_year
            period_target = target_return / periods_per_year

            excess_returns = returns_array - period_risk_free

            # Calculate downside deviation
            downside_returns = returns_array - period_target
            downside_only = downside_returns[downside_returns < 0]

            if len(downside_only) > 0:
                downside_deviation = np.sqrt(np.mean(downside_only ** 2))
            else:
                downside_deviation = 0

            # Calculate Sortino ratio
            mean_excess_return = np.mean(excess_returns)

            if downside_deviation > 0:
                sortino_ratio = mean_excess_return / downside_deviation
                annualized_sortino = sortino_ratio * np.sqrt(periods_per_year)
            else:
                sortino_ratio = float('inf') if mean_excess_return > 0 else 0
                annualized_sortino = float('inf') if mean_excess_return > 0 else 0

            # Additional downside risk metrics
            downside_frequency = len(downside_only) / len(returns_array)
            max_drawdown = self._calculate_max_drawdown(returns_array)

            # Performance interpretation
            performance_rating = self._interpret_sortino_ratio(annualized_sortino)

            return {
                'sortino_ratio': round(sortino_ratio, self.config.precision),
                'annualized_sortino_ratio': round(annualized_sortino if annualized_sortino != float('inf') else 999, self.config.precision),
                'downside_deviation': round(downside_deviation, self.config.precision),
                'annualized_downside_deviation': round(downside_deviation * np.sqrt(periods_per_year), self.config.precision),
                'mean_excess_return': round(mean_excess_return, self.config.precision),
                'target_return': round(target_return, self.config.precision),
                'downside_frequency': round(downside_frequency, self.config.precision),
                'max_drawdown': round(max_drawdown, self.config.precision),
                'performance_rating': performance_rating,
                'sample_size': len(returns_array),
                'downside_observations': len(downside_only),
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def comprehensive_performance_metrics(self, returns: Union[pd.Series, np.ndarray],
                                        benchmark_returns: Union[pd.Series, np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive set of performance metrics.

        Parameters:
        -----------
        returns : Series or array
            Portfolio return series
        benchmark_returns : Series or array
            Benchmark return series

        Returns:
        --------
        Dict with comprehensive performance analysis
        """
        try:
            # Basic performance metrics
            cv_result = self.coefficient_of_variation(returns)
            sharpe_result = self.sharpe_ratio(returns)
            sortino_result = self.sortino_ratio(returns)

            # Additional metrics
            additional_metrics = self._calculate_additional_metrics(returns)

            # Benchmark comparison if provided
            benchmark_comparison = {}
            if benchmark_returns is not None:
                benchmark_comparison = self._compare_to_benchmark(returns, benchmark_returns)

            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(returns)

            # Performance attribution
            attribution = self._calculate_performance_attribution(returns, benchmark_returns)

            return {
                'coefficient_of_variation': cv_result,
                'sharpe_ratio': sharpe_result,
                'sortino_ratio': sortino_result,
                'additional_metrics': additional_metrics,
                'risk_metrics': risk_metrics,
                'benchmark_comparison': benchmark_comparison,
                'performance_attribution': attribution,
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def _get_periods_per_year(self) -> int:
        """Get number of periods per year based on data frequency."""
        frequency_map = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4,
            'annually': 1
        }
        return frequency_map.get(self.config.data_frequency, 252)

    def _interpret_cv(self, cv: float) -> str:
        """Interpret coefficient of variation."""
        if cv < 0.5:
            return 'Excellent'
        elif cv < 1.0:
            return 'Good'
        elif cv < 1.5:
            return 'Fair'
        else:
            return 'Poor'

    def _interpret_sharpe_ratio(self, sharpe: float) -> str:
        """Interpret Sharpe ratio."""
        if sharpe > 2.0:
            return 'Excellent'
        elif sharpe > 1.0:
            return 'Good'
        elif sharpe > 0.5:
            return 'Fair'
        elif sharpe > 0:
            return 'Poor'
        else:
            return 'Very Poor'

    def _interpret_sortino_ratio(self, sortino: float) -> str:
        """Interpret Sortino ratio."""
        if sortino == float('inf') or sortino > 3.0:
            return 'Excellent'
        elif sortino > 2.0:
            return 'Very Good'
        elif sortino > 1.0:
            return 'Good'
        elif sortino > 0.5:
            return 'Fair'
        else:
            return 'Poor'

    def _calculate_sharpe_confidence_interval(self, returns: np.ndarray,
                                            risk_free_rate: float, confidence: float) -> Tuple[float, float]:
        """Calculate confidence interval for Sharpe ratio."""
        try:
            n = len(returns)
            periods_per_year = self._get_periods_per_year()

            # Calculate Sharpe ratio
            period_rf = risk_free_rate / periods_per_year
            excess_returns = returns - period_rf
            sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1)
            annualized_sharpe = sharpe * np.sqrt(periods_per_year)

            # Standard error of Sharpe ratio
            se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / n)

            # Confidence interval
            z_score = stats.norm.ppf((1 + confidence) / 2)
            margin_error = z_score * se_sharpe * np.sqrt(periods_per_year)

            lower_bound = annualized_sharpe - margin_error
            upper_bound = annualized_sharpe + margin_error

            return lower_bound, upper_bound

        except Exception:
            return 0, 0

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative_returns = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            return np.min(drawdown)
        except Exception:
            return 0

    def _calculate_additional_metrics(self, returns: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Calculate additional performance metrics."""
        try:
            if isinstance(returns, pd.Series):
                returns_array = returns.dropna().values
            else:
                returns_array = np.array(returns)

            periods_per_year = self._get_periods_per_year()

            # Calmar ratio
            annualized_return = np.mean(returns_array) * periods_per_year
            max_dd = abs(self._calculate_max_drawdown(returns_array))
            calmar_ratio = annualized_return / max_dd if max_dd > 0 else 0

            # Information ratio (assuming benchmark is risk-free rate)
            excess_returns = returns_array - (self.config.risk_free_rate / periods_per_year)
            tracking_error = np.std(excess_returns, ddof=1) * np.sqrt(periods_per_year)
            information_ratio = (np.mean(excess_returns) * periods_per_year) / tracking_error if tracking_error > 0 else 0

            # Skewness and kurtosis
            skewness = stats.skew(returns_array)
            kurtosis = stats.kurtosis(returns_array)

            # Value at Risk (95%)
            var_95 = np.percentile(returns_array, 5)

            return {
                'calmar_ratio': round(calmar_ratio, self.config.precision),
                'information_ratio': round(information_ratio, self.config.precision),
                'tracking_error': round(tracking_error, self.config.precision),
                'skewness': round(skewness, self.config.precision),
                'kurtosis': round(kurtosis, self.config.precision),
                'value_at_risk_95': round(var_95, self.config.precision),
                'max_drawdown': round(max_dd, self.config.precision)
            }

        except Exception as e:
            return {'error': str(e)}

    def _calculate_risk_metrics(self, returns: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics."""
        try:
            if isinstance(returns, pd.Series):
                returns_array = returns.dropna().values
            else:
                returns_array = np.array(returns)

            periods_per_year = self._get_periods_per_year()

            # Volatility
            volatility = np.std(returns_array, ddof=1) * np.sqrt(periods_per_year)

            # Downside risk metrics
            negative_returns = returns_array[returns_array < 0]
            downside_volatility = np.std(negative_returns, ddof=1) * np.sqrt(periods_per_year) if len(negative_returns) > 0 else 0

            # Semi-variance
            mean_return = np.mean(returns_array)
            below_mean = returns_array[returns_array < mean_return]
            semi_variance = np.var(below_mean, ddof=1) if len(below_mean) > 0 else 0

            # Tail risk
            var_99 = np.percentile(returns_array, 1)
            cvar_99 = np.mean(returns_array[returns_array <= var_99]) if np.sum(returns_array <= var_99) > 0 else var_99

            return {
                'volatility': round(volatility, self.config.precision),
                'downside_volatility': round(downside_volatility, self.config.precision),
                'semi_variance': round(semi_variance, self.config.precision),
                'var_99': round(var_99, self.config.precision),
                'cvar_99': round(cvar_99, self.config.precision),
                'upside_capture': round(len(returns_array[returns_array > 0]) / len(returns_array), self.config.precision),
                'downside_capture': round(len(returns_array[returns_array < 0]) / len(returns_array), self.config.precision)
            }

        except Exception as e:
            return {'error': str(e)}

    def _compare_to_benchmark(self, returns: Union[pd.Series, np.ndarray],
                             benchmark_returns: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Compare portfolio performance to benchmark."""
        try:
            # Align data
            if isinstance(returns, pd.Series) and isinstance(benchmark_returns, pd.Series):
                common_index = returns.index.intersection(benchmark_returns.index)
                port_returns = returns.loc[common_index].values
                bench_returns = benchmark_returns.loc[common_index].values
            else:
                min_length = min(len(returns), len(benchmark_returns))
                port_returns = np.array(returns)[:min_length]
                bench_returns = np.array(benchmark_returns)[:min_length]

            periods_per_year = self._get_periods_per_year()

            # Excess returns
            excess_returns = port_returns - bench_returns
            mean_excess = np.mean(excess_returns)
            excess_volatility = np.std(excess_returns, ddof=1)

            # Tracking error
            tracking_error = excess_volatility * np.sqrt(periods_per_year)

            # Information ratio
            information_ratio = (mean_excess * periods_per_year) / tracking_error if tracking_error > 0 else 0

            # Up/down capture ratios
            up_periods = bench_returns > 0
            down_periods = bench_returns < 0

            upside_capture = (np.mean(port_returns[up_periods]) / np.mean(bench_returns[up_periods])) if np.sum(up_periods) > 0 else 1
            downside_capture = (np.mean(port_returns[down_periods]) / np.mean(bench_returns[down_periods])) if np.sum(down_periods) > 0 else 1

            # Correlation
            correlation = np.corrcoef(port_returns, bench_returns)[0, 1]

            return {
                'excess_return': round(mean_excess * periods_per_year, self.config.precision),
                'tracking_error': round(tracking_error, self.config.precision),
                'information_ratio': round(information_ratio, self.config.precision),
                'correlation': round(correlation, self.config.precision),
                'upside_capture': round(upside_capture, self.config.precision),
                'downside_capture': round(downside_capture, self.config.precision),
                'active_return': round(mean_excess * periods_per_year, self.config.precision)
            }

        except Exception as e:
            return {'error': str(e)}

    def _calculate_performance_attribution(self, returns: Union[pd.Series, np.ndarray],
                                         benchmark_returns: Union[pd.Series, np.ndarray] = None) -> Dict[str, Any]:
        """Calculate performance attribution."""
        try:
            if isinstance(returns, pd.Series):
                returns_array = returns.dropna().values
            else:
                returns_array = np.array(returns)

            periods_per_year = self._get_periods_per_year()

            # Time-based attribution
            monthly_returns = []
            if len(returns_array) >= 21:  # Approximate monthly periods
                for i in range(0, len(returns_array), 21):
                    month_returns = returns_array[i:i+21]
                    if len(month_returns) > 0:
                        monthly_returns.append(np.prod(1 + month_returns) - 1)

            # Performance consistency
            consistency_ratio = len([r for r in monthly_returns if r > 0]) / len(monthly_returns) if monthly_returns else 0

            return {
                'consistency_ratio': round(consistency_ratio, self.config.precision),
                'monthly_volatility': round(np.std(monthly_returns) if monthly_returns else 0, self.config.precision),
                'best_month': round(max(monthly_returns) if monthly_returns else 0, self.config.precision),
                'worst_month': round(min(monthly_returns) if monthly_returns else 0, self.config.precision)
            }

        except Exception as e:
            return {'error': str(e)}


class PortfolioCalculations:
    """
    Portfolio return, risk, and construction calculations.
    """

    def __init__(self, config: PortfolioConfig = None):
        self.config = config or PortfolioConfig()

    def weighted_portfolio_returns(self, returns: pd.DataFrame, weights: Union[List[float], np.ndarray],
                                 rebalancing: str = 'monthly') -> Dict[str, Any]:
        """
        Calculate weighted portfolio returns with rebalancing.

        Parameters:
        -----------
        returns : DataFrame
            Asset returns with assets as columns and dates as index
        weights : List or array
            Portfolio weights (must sum to 1)
        rebalancing : str
            Rebalancing frequency

        Returns:
        --------
        Dict with portfolio return calculations
        """
        try:
            if len(weights) != len(returns.columns):
                return {'error': 'Number of weights must match number of assets'}

            weights = np.array(weights)
            if abs(np.sum(weights) - 1.0) > 1e-6:
                return {'error': 'Weights must sum to 1.0'}

            # Calculate portfolio returns
            if rebalancing == 'buy_and_hold':
                # Buy and hold strategy
                portfolio_returns = self._calculate_buy_and_hold_returns(returns, weights)
            else:
                # Periodic rebalancing
                portfolio_returns = self._calculate_rebalanced_returns(returns, weights, rebalancing)

            # Calculate cumulative returns
            cumulative_returns = (1 + portfolio_returns).cumprod()

            # Performance metrics
            periods_per_year = self._get_periods_per_year()
            mean_return = portfolio_returns.mean()
            std_return = portfolio_returns.std(ddof=1)

            annualized_return = mean_return * periods_per_year
            annualized_volatility = std_return * np.sqrt(periods_per_year)

            # Total return
            total_return = cumulative_returns.iloc[-1] - 1

            # Maximum drawdown
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()

            # Calculate contribution to return by asset
            asset_contributions = self._calculate_asset_contributions(returns, weights, rebalancing)

            return {
                'portfolio_returns': portfolio_returns.round(self.config.precision),
                'cumulative_returns': cumulative_returns.round(self.config.precision),
                'weights': [round(w, self.config.precision) for w in weights],
                'mean_return': round(mean_return, self.config.precision),
                'annualized_return': round(annualized_return, self.config.precision),
                'volatility': round(std_return, self.config.precision),
                'annualized_volatility': round(annualized_volatility, self.config.precision),
                'total_return': round(total_return, self.config.precision),
                'max_drawdown': round(max_drawdown, self.config.precision),
                'sharpe_ratio': round((annualized_return - self.config.risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0, self.config.precision),
                'asset_contributions': asset_contributions,
                'rebalancing_frequency': rebalancing,
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def portfolio_variance_standard_deviation(self, returns: pd.DataFrame,
                                            weights: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """
        Calculate portfolio variance and standard deviation.

        Parameters:
        -----------
        returns : DataFrame
            Asset returns
        weights : List or array
            Portfolio weights

        Returns:
        --------
        Dict with variance and standard deviation calculations
        """
        try:
            if len(weights) != len(returns.columns):
                return {'error': 'Number of weights must match number of assets'}

            weights = np.array(weights)

            # Calculate covariance matrix
            cov_matrix = returns.cov()

            # Portfolio variance: w^T * Σ * w
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_std = np.sqrt(portfolio_variance)

            # Annualized values
            periods_per_year = self._get_periods_per_year()
            annualized_variance = portfolio_variance * periods_per_year
            annualized_std = portfolio_std * np.sqrt(periods_per_year)

            # Risk decomposition
            risk_contributions = self._calculate_risk_contributions(weights, cov_matrix)

            # Marginal risk contributions
            marginal_contributions = np.dot(cov_matrix, weights) / portfolio_std

            # Diversification ratio
            individual_volatilities = np.sqrt(np.diag(cov_matrix))
            weighted_avg_vol = np.dot(weights, individual_volatilities)
            diversification_ratio = weighted_avg_vol / portfolio_std

            return {
                'portfolio_variance': round(portfolio_variance, self.config.precision),
                'portfolio_standard_deviation': round(portfolio_std, self.config.precision),
                'annualized_variance': round(annualized_variance, self.config.precision),
                'annualized_standard_deviation': round(annualized_std, self.config.precision),
                'weights': [round(w, self.config.precision) for w in weights],
                'risk_contributions': {
                    asset: round(contrib, self.config.precision)
                    for asset, contrib in zip(returns.columns, risk_contributions)
                },
                'marginal_risk_contributions': {
                    asset: round(contrib, self.config.precision)
                    for asset, contrib in zip(returns.columns, marginal_contributions)
                },
                'diversification_ratio': round(diversification_ratio, self.config.precision),
                'covariance_matrix': cov_matrix.round(self.config.precision),
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def efficient_frontier(self, returns: pd.DataFrame, n_portfolios: int = 100,
                          include_risk_free: bool = True) -> Dict[str, Any]:
        """
        Generate efficient frontier for portfolio optimization.

        Parameters:
        -----------
        returns : DataFrame
            Asset returns
        n_portfolios : int
            Number of portfolios to generate
        include_risk_free : bool
            Include risk-free asset

        Returns:
        --------
        Dict with efficient frontier data
        """
        try:
            n_assets = len(returns.columns)

            # Calculate expected returns and covariance matrix
            mean_returns = returns.mean()
            cov_matrix = returns.cov()

            # Generate target returns
            min_return = mean_returns.min()
            max_return = mean_returns.max()
            target_returns = np.linspace(min_return, max_return, n_portfolios)

            efficient_portfolios = []

            for target_return in target_returns:
                # Optimize portfolio for given target return
                portfolio = self._optimize_portfolio(mean_returns, cov_matrix, target_return)

                if portfolio is not None:
                    efficient_portfolios.append(portfolio)

            # Extract returns and risks
            portfolio_returns = [p['return'] for p in efficient_portfolios]
            portfolio_risks = [p['risk'] for p in efficient_portfolios]
            portfolio_weights = [p['weights'] for p in efficient_portfolios]

            # Calculate tangency portfolio (maximum Sharpe ratio)
            tangency_portfolio = self._find_tangency_portfolio(mean_returns, cov_matrix)

            # Minimum variance portfolio
            min_var_portfolio = self._find_minimum_variance_portfolio(mean_returns, cov_matrix)

            return {
                'efficient_portfolios': efficient_portfolios,
                'portfolio_returns': [round(r, self.config.precision) for r in portfolio_returns],
                'portfolio_risks': [round(r, self.config.precision) for r in portfolio_risks],
                'portfolio_weights': [[round(w, self.config.precision) for w in weights] for weights in portfolio_weights],
                'tangency_portfolio': tangency_portfolio,
                'minimum_variance_portfolio': min_var_portfolio,
                'asset_names': list(returns.columns),
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def _calculate_buy_and_hold_returns(self, returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
        """Calculate buy-and-hold portfolio returns."""
        # Initial portfolio value allocation
        cumulative_asset_returns = (1 + returns).cumprod()

        # Calculate portfolio value at each time point
        portfolio_values = np.dot(cumulative_asset_returns, weights)

        # Calculate portfolio returns
        portfolio_returns = portfolio_values.pct_change().dropna()

        return portfolio_returns

    def _calculate_rebalanced_returns(self, returns: pd.DataFrame, weights: np.ndarray,
                                    rebalancing: str) -> pd.Series:
        """Calculate portfolio returns with periodic rebalancing."""
        # Determine rebalancing frequency
        freq_map = {
            'daily': 1,
            'weekly': 5,
            'monthly': 21,
            'quarterly': 63,
            'annually': 252
        }

        rebal_freq = freq_map.get(rebalancing, 21)

        portfolio_returns = []

        for i in range(0, len(returns), rebal_freq):
            period_returns = returns.iloc[i:i+rebal_freq]
            if len(period_returns) > 0:
                # Calculate period portfolio returns
                period_portfolio_returns = np.dot(period_returns, weights)
                portfolio_returns.extend(period_portfolio_returns.tolist())

        return pd.Series(portfolio_returns, index=returns.index[:len(portfolio_returns)])

    def _calculate_asset_contributions(self, returns: pd.DataFrame, weights: np.ndarray,
                                     rebalancing: str) -> Dict[str, float]:
        """Calculate each asset's contribution to portfolio return."""
        try:
            mean_returns = returns.mean()
            contributions = weights * mean_returns

            return {
                asset: round(contrib, self.config.precision)
                for asset, contrib in zip(returns.columns, contributions)
            }
        except Exception:
            return {}

    def _calculate_risk_contributions(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Calculate risk contributions of each asset."""
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / portfolio_variance
        return risk_contrib

    def _optimize_portfolio(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                          target_return: float) -> Dict[str, Any]:
        """Optimize portfolio for given target return."""
        try:
            n_assets = len(mean_returns)

            # Objective function (minimize variance)
            def objective(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))

            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # weights sum to 1
                {'type': 'eq', 'fun': lambda weights: np.dot(weights, mean_returns) - target_return}  # target return
            ]

            # Bounds (0 <= weight <= max_weight)
            bounds = tuple((0, self.config.max_weight) for _ in range(n_assets))

            # Initial guess
            x0 = np.array([1/n_assets] * n_assets)

            # Optimize
            result = optimize.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

            if result.success:
                weights = result.x
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_risk = np.sqrt(objective(weights))

                return {
                    'weights': weights.tolist(),
                    'return': portfolio_return,
                    'risk': portfolio_risk,
                    'sharpe_ratio': (portfolio_return - self.config.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
                }
            else:
                return None

        except Exception:
            return None

    def _find_tangency_portfolio(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Find tangency portfolio (maximum Sharpe ratio)."""
        try:
            n_assets = len(mean_returns)

            # Objective function (minimize negative Sharpe ratio)
            def objective(weights):
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_risk
                return -sharpe_ratio  # Minimize negative for maximization

            # Constraints
            constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]

            # Bounds
            bounds = tuple((0, self.config.max_weight) for _ in range(n_assets))

            # Initial guess
            x0 = np.array([1/n_assets] * n_assets)

            # Optimize
            result = optimize.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

            if result.success:
                weights = result.x
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_risk

                return {
                    'weights': [round(w, self.config.precision) for w in weights],
                    'return': round(portfolio_return, self.config.precision),
                    'risk': round(portfolio_risk, self.config.precision),
                    'sharpe_ratio': round(sharpe_ratio, self.config.precision)
                }
            else:
                return {'error': 'Optimization failed'}

        except Exception as e:
            return {'error': str(e)}

    def _find_minimum_variance_portfolio(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Find minimum variance portfolio."""
        try:
            n_assets = len(mean_returns)

            # Objective function (minimize variance)
            def objective(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))

            # Constraints
            constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]

            # Bounds
            bounds = tuple((0, self.config.max_weight) for _ in range(n_assets))

            # Initial guess
            x0 = np.array([1/n_assets] * n_assets)

            # Optimize
            result = optimize.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

            if result.success:
                weights = result.x
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_risk = np.sqrt(objective(weights))

                return {
                    'weights': [round(w, self.config.precision) for w in weights],
                    'return': round(portfolio_return, self.config.precision),
                    'risk': round(portfolio_risk, self.config.precision)
                }
            else:
                return {'error': 'Optimization failed'}

        except Exception as e:
            return {'error': str(e)}

    def _get_periods_per_year(self) -> int:
        """Get number of periods per year based on data frequency."""
        frequency_map = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4,
            'annually': 1
        }
        return frequency_map.get(self.config.data_frequency, 252)


class CAPMAnalysis:
    """
    Capital Asset Pricing Model (CAPM) analysis and beta calculations.
    """

    def __init__(self, config: PortfolioConfig = None):
        self.config = config or PortfolioConfig()

    def capm_expected_return(self, beta: float, risk_free_rate: float = None,
                           market_return: float = None) -> Dict[str, Any]:
        """
        Calculate CAPM expected return.

        Parameters:
        -----------
        beta : float
            Stock beta
        risk_free_rate : float
            Risk-free rate
        market_return : float
            Expected market return

        Returns:
        --------
        Dict with CAPM expected return calculation
        """
        try:
            if risk_free_rate is None:
                risk_free_rate = self.config.risk_free_rate

            if market_return is None:
                # Use historical market return as proxy
                market_return = 0.10  # Default 10% market return

            # CAPM formula: E(R) = Rf + β(E(Rm) - Rf)
            market_premium = market_return - risk_free_rate
            expected_return = risk_free_rate + beta * market_premium

            # Calculate required risk premium
            required_risk_premium = beta * market_premium

            # Security market line analysis
            sml_analysis = self._analyze_security_market_line(beta, expected_return, risk_free_rate, market_return)

            return {
                'beta': round(beta, self.config.precision),
                'risk_free_rate': round(risk_free_rate, self.config.precision),
                'market_return': round(market_return, self.config.precision),
                'market_risk_premium': round(market_premium, self.config.precision),
                'expected_return': round(expected_return, self.config.precision),
                'required_risk_premium': round(required_risk_premium, self.config.precision),
                'annualized_expected_return': round(expected_return, self.config.precision),
                'sml_analysis': sml_analysis,
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def calculate_beta_variance_covariance(self, stock_returns: Union[pd.Series, np.ndarray],
                                         market_returns: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Calculate beta using variance-covariance method.

        Parameters:
        -----------
        stock_returns : Series or array
            Stock return series
        market_returns : Series or array
            Market return series

        Returns:
        --------
        Dict with beta calculation using variance-covariance method
        """
        try:
            # Align data
            if isinstance(stock_returns, pd.Series) and isinstance(market_returns, pd.Series):
                common_index = stock_returns.index.intersection(market_returns.index)
                stock_data = stock_returns.loc[common_index].dropna()
                market_data = market_returns.loc[common_index].dropna()

                # Further align after dropna
                common_index = stock_data.index.intersection(market_data.index)
                stock_array = stock_data.loc[common_index].values
                market_array = market_data.loc[common_index].values
            else:
                min_length = min(len(stock_returns), len(market_returns))
                stock_array = np.array(stock_returns)[:min_length]
                market_array = np.array(market_returns)[:min_length]

            if len(stock_array) < self.config.min_periods:
                return {'error': f'Insufficient data points. Need at least {self.config.min_periods}'}

            # Calculate covariance and variance
            covariance = np.cov(stock_array, market_array)[0, 1]
            market_variance = np.var(market_array, ddof=1)

            if market_variance == 0:
                return {'error': 'Market variance is zero'}

            # Beta = Cov(Rs, Rm) / Var(Rm)
            beta = covariance / market_variance

            # Calculate alpha (Jensen's alpha)
            stock_mean = np.mean(stock_array)
            market_mean = np.mean(market_array)
            alpha = stock_mean - beta * market_mean

            # R-squared (coefficient of determination)
            correlation = np.corrcoef(stock_array, market_array)[0, 1]
            r_squared = correlation ** 2

            # Standard error of beta
            stock_variance = np.var(stock_array, ddof=1)
            residual_variance = stock_variance * (1 - r_squared)
            beta_std_error = np.sqrt(residual_variance / (market_variance * (len(stock_array) - 2)))

            # Confidence intervals for beta
            beta_ci = self._calculate_beta_confidence_interval(beta, beta_std_error, len(stock_array))

            # Statistical significance
            t_statistic = beta / beta_std_error if beta_std_error > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), len(stock_array) - 2))

            return {
                'method': 'variance_covariance',
                'beta': round(beta, self.config.precision),
                'alpha': round(alpha, self.config.precision),
                'correlation': round(correlation, self.config.precision),
                'r_squared': round(r_squared, self.config.precision),
                'covariance': round(covariance, self.config.precision),
                'market_variance': round(market_variance, self.config.precision),
                'beta_standard_error': round(beta_std_error, self.config.precision),
                'beta_confidence_interval': {
                    'lower': round(beta_ci[0], self.config.precision),
                    'upper': round(beta_ci[1], self.config.precision)
                },
                't_statistic': round(t_statistic, self.config.precision),
                'p_value': round(p_value, self.config.precision),
                'sample_size': len(stock_array),
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def calculate_beta_correlation(self, stock_returns: Union[pd.Series, np.ndarray],
                                 market_returns: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Calculate beta using correlation method.

        Parameters:
        -----------
        stock_returns : Series or array
            Stock return series
        market_returns : Series or array
            Market return series

        Returns:
        --------
        Dict with beta calculation using correlation method
        """
        try:
            # Align data (same as variance-covariance method)
            if isinstance(stock_returns, pd.Series) and isinstance(market_returns, pd.Series):
                common_index = stock_returns.index.intersection(market_returns.index)
                stock_data = stock_returns.loc[common_index].dropna()
                market_data = market_returns.loc[common_index].dropna()

                common_index = stock_data.index.intersection(market_data.index)
                stock_array = stock_data.loc[common_index].values
                market_array = market_data.loc[common_index].values
            else:
                min_length = min(len(stock_returns), len(market_returns))
                stock_array = np.array(stock_returns)[:min_length]
                market_array = np.array(market_returns)[:min_length]

            if len(stock_array) < self.config.min_periods:
                return {'error': f'Insufficient data points. Need at least {self.config.min_periods}'}

            # Calculate correlation and standard deviations
            correlation = np.corrcoef(stock_array, market_array)[0, 1]
            stock_std = np.std(stock_array, ddof=1)
            market_std = np.std(market_array, ddof=1)

            if market_std == 0:
                return {'error': 'Market standard deviation is zero'}

            # Beta = ρ * (σs / σm)
            beta = correlation * (stock_std / market_std)

            # Calculate additional statistics
            stock_mean = np.mean(stock_array)
            market_mean = np.mean(market_array)

            # Alpha calculation
            alpha = stock_mean - beta * market_mean

            # Systematic and idiosyncratic risk
            systematic_risk = (beta * market_std) ** 2
            total_risk = stock_std ** 2
            idiosyncratic_risk = total_risk - systematic_risk

            # Risk decomposition
            systematic_risk_pct = systematic_risk / total_risk * 100
            idiosyncratic_risk_pct = idiosyncratic_risk / total_risk * 100

            return {
                'method': 'correlation',
                'beta': round(beta, self.config.precision),
                'alpha': round(alpha, self.config.precision),
                'correlation': round(correlation, self.config.precision),
                'stock_volatility': round(stock_std, self.config.precision),
                'market_volatility': round(market_std, self.config.precision),
                'systematic_risk': round(systematic_risk, self.config.precision),
                'idiosyncratic_risk': round(idiosyncratic_risk, self.config.precision),
                'systematic_risk_percentage': round(systematic_risk_pct, self.config.precision),
                'idiosyncratic_risk_percentage': round(idiosyncratic_risk_pct, self.config.precision),
                'r_squared': round(correlation ** 2, self.config.precision),
                'sample_size': len(stock_array),
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def rolling_beta(self, stock_returns: Union[pd.Series, np.ndarray],
                    market_returns: Union[pd.Series, np.ndarray],
                    window: int = None) -> Dict[str, Any]:
        """
        Calculate rolling beta over time.

        Parameters:
        -----------
        stock_returns : Series or array
            Stock return series
        market_returns : Series or array
            Market return series
        window : int
            Rolling window size

        Returns:
        --------
        Dict with rolling beta analysis
        """
        try:
            if window is None:
                window = self.config.rolling_window

            # Convert to pandas Series if needed
            if not isinstance(stock_returns, pd.Series):
                stock_returns = pd.Series(stock_returns)
            if not isinstance(market_returns, pd.Series):
                market_returns = pd.Series(market_returns)

            # Align data
            common_index = stock_returns.index.intersection(market_returns.index)
            stock_data = stock_returns.loc[common_index]
            market_data = market_returns.loc[common_index]

            if len(stock_data) < window + self.config.min_periods:
                return {'error': f'Insufficient data for rolling calculation'}

            # Calculate rolling statistics
            rolling_betas = []
            rolling_alphas = []
            rolling_r_squared = []
            dates = []

            for i in range(window, len(stock_data)):
                window_stock = stock_data.iloc[i-window:i]
                window_market = market_data.iloc[i-window:i]

                # Calculate beta for this window
                covariance = np.cov(window_stock, window_market)[0, 1]
                market_variance = np.var(window_market, ddof=1)

                if market_variance > 0:
                    beta = covariance / market_variance
                    alpha = np.mean(window_stock) - beta * np.mean(window_market)
                    correlation = np.corrcoef(window_stock, window_market)[0, 1]
                    r_squared = correlation ** 2

                    rolling_betas.append(beta)
                    rolling_alphas.append(alpha)
                    rolling_r_squared.append(r_squared)
                    dates.append(stock_data.index[i])

            # Convert to Series
            rolling_beta_series = pd.Series(rolling_betas, index=dates)
            rolling_alpha_series = pd.Series(rolling_alphas, index=dates)
            rolling_r_squared_series = pd.Series(rolling_r_squared, index=dates)

            # Calculate statistics on rolling betas
            beta_stats = {
                'mean': round(rolling_beta_series.mean(), self.config.precision),
                'std': round(rolling_beta_series.std(), self.config.precision),
                'min': round(rolling_beta_series.min(), self.config.precision),
                'max': round(rolling_beta_series.max(), self.config.precision),
                'current': round(rolling_beta_series.iloc[-1], self.config.precision)
            }

            # Beta stability analysis
            beta_stability = self._analyze_beta_stability(rolling_beta_series)

            return {
                'rolling_beta': rolling_beta_series.round(self.config.precision),
                'rolling_alpha': rolling_alpha_series.round(self.config.precision),
                'rolling_r_squared': rolling_r_squared_series.round(self.config.precision),
                'beta_statistics': beta_stats,
                'beta_stability': beta_stability,
                'window_size': window,
                'number_of_observations': len(rolling_betas),
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def multi_factor_beta(self, stock_returns: Union[pd.Series, np.ndarray],
                         factor_returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate multi-factor beta using multiple risk factors.

        Parameters:
        -----------
        stock_returns : Series or array
            Stock return series
        factor_returns : DataFrame
            Risk factor returns (each column is a factor)

        Returns:
        --------
        Dict with multi-factor beta analysis
        """
        try:
            # Convert stock returns to Series if needed
            if not isinstance(stock_returns, pd.Series):
                stock_returns = pd.Series(stock_returns)

            # Align data
            common_index = stock_returns.index.intersection(factor_returns.index)
            stock_data = stock_returns.loc[common_index]
            factor_data = factor_returns.loc[common_index]

            if len(stock_data) < self.config.min_periods:
                return {'error': f'Insufficient data points. Need at least {self.config.min_periods}'}

            # Prepare data for regression
            X = factor_data.values
            y = stock_data.values

            # Fit multiple regression
            model = LinearRegression()
            model.fit(X, y)

            # Get regression results
            factor_betas = model.coef_
            alpha = model.intercept_
            r_squared = model.score(X, y)

            # Calculate factor contributions
            factor_contributions = {}
            for i, factor_name in enumerate(factor_data.columns):
                factor_mean = factor_data[factor_name].mean()
                contribution = factor_betas[i] * factor_mean
                factor_contributions[factor_name] = round(contribution, self.config.precision)

            # Residual analysis
            predicted = model.predict(X)
            residuals = y - predicted
            residual_std = np.std(residuals, ddof=1)

            # Factor loadings with statistical significance
            factor_loadings = {}
            for i, factor_name in enumerate(factor_data.columns):
                factor_loadings[factor_name] = {
                    'beta': round(factor_betas[i], self.config.precision),
                    'contribution': factor_contributions[factor_name]
                }

            return {
                'alpha': round(alpha, self.config.precision),
                'factor_loadings': factor_loadings,
                'r_squared': round(r_squared, self.config.precision),
                'adjusted_r_squared': round(1 - (1 - r_squared) * (len(y) - 1) / (len(y) - len(factor_betas) - 1), self.config.precision),
                'residual_standard_error': round(residual_std, self.config.precision),
                'factor_names': list(factor_data.columns),
                'sample_size': len(stock_data),
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def _analyze_security_market_line(self, beta: float, expected_return: float,
                                    risk_free_rate: float, market_return: float) -> Dict[str, Any]:
        """Analyze position relative to Security Market Line."""
        try:
            # SML expected return for given beta
            sml_expected_return = risk_free_rate + beta * (market_return - risk_free_rate)

            # Alpha relative to SML
            sml_alpha = expected_return - sml_expected_return

            # Classification
            if abs(sml_alpha) < 0.01:  # Within 1%
                classification = 'Fairly Priced'
            elif sml_alpha > 0:
                classification = 'Undervalued'
            else:
                classification = 'Overvalued'

            return {
                'sml_expected_return': round(sml_expected_return, self.config.precision),
                'sml_alpha': round(sml_alpha, self.config.precision),
                'classification': classification,
                'risk_level': 'High' if beta > 1.5 else 'Moderate' if beta > 0.7 else 'Low'
            }

        except Exception:
            return {}

    def _calculate_beta_confidence_interval(self, beta: float, std_error: float,
                                          sample_size: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for beta."""
        try:
            degrees_freedom = sample_size - 2
            t_critical = stats.t.ppf((1 + confidence) / 2, degrees_freedom)
            margin_error = t_critical * std_error

            lower_bound = beta - margin_error
            upper_bound = beta + margin_error

            return lower_bound, upper_bound

        except Exception:
            return beta, beta

    def _analyze_beta_stability(self, rolling_beta: pd.Series) -> Dict[str, Any]:
        """Analyze stability of rolling beta."""
        try:
            # Calculate coefficient of variation
            mean_beta = rolling_beta.mean()
            std_beta = rolling_beta.std()
            cv_beta = std_beta / abs(mean_beta) if mean_beta != 0 else float('inf')

            # Trend analysis
            time_trend = np.arange(len(rolling_beta))
            slope, _, r_value, p_value, _ = stats.linregress(time_trend, rolling_beta.values)

            # Stability classification
            if cv_beta < 0.2:
                stability = 'Very Stable'
            elif cv_beta < 0.4:
                stability = 'Stable'
            elif cv_beta < 0.6:
                stability = 'Moderately Stable'
            else:
                stability = 'Unstable'

            return {
                'coefficient_of_variation': round(cv_beta, self.config.precision),
                'trend_slope': round(slope, self.config.precision),
                'trend_r_squared': round(r_value ** 2, self.config.precision),
                'trend_p_value': round(p_value, self.config.precision),
                'stability_classification': stability
            }

        except Exception:
            return {'stability_classification': 'Unknown'}


# Portfolio optimization and analysis utilities
def get_stock_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch stock data for portfolio analysis.

    Parameters:
    -----------
    tickers : List[str]
        List of stock tickers
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)

    Returns:
    --------
    DataFrame with adjusted close prices
    """
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        return data.dropna()
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()


def calculate_returns(prices: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
    """
    Calculate returns from price data.

    Parameters:
    -----------
    prices : DataFrame
        Price data
    method : str
        Return calculation method ('simple' or 'log')

    Returns:
    --------
    DataFrame with returns
    """
    try:
        if method == 'simple':
            returns = prices.pct_change().dropna()
        elif method == 'log':
            returns = np.log(prices / prices.shift(1)).dropna()
        else:
            raise ValueError("Method must be 'simple' or 'log'")

        return returns
    except Exception as e:
        logger.error(f"Error calculating returns: {e}")
        return pd.DataFrame()


def plot_efficient_frontier(frontier_data: Dict[str, Any], title: str = "Efficient Frontier"):
    """
    Plot efficient frontier.

    Parameters:
    -----------
    frontier_data : Dict
        Efficient frontier data from efficient_frontier method
    title : str
        Plot title
    """
    try:
        plt.figure(figsize=(12, 8))

        risks = frontier_data['portfolio_risks']
        returns = frontier_data['portfolio_returns']

        plt.plot(risks, returns, 'b-', linewidth=2, label='Efficient Frontier')

        # Mark special portfolios
        if 'tangency_portfolio' in frontier_data and 'error' not in frontier_data['tangency_portfolio']:
            tang = frontier_data['tangency_portfolio']
            plt.plot(tang['risk'], tang['return'], 'ro', markersize=10, label='Tangency Portfolio')

        if 'minimum_variance_portfolio' in frontier_data and 'error' not in frontier_data['minimum_variance_portfolio']:
            min_var = frontier_data['minimum_variance_portfolio']
            plt.plot(min_var['risk'], min_var['return'], 'go', markersize=10, label='Minimum Variance Portfolio')

        plt.xlabel('Risk (Standard Deviation)')
        plt.ylabel('Expected Return')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    except Exception as e:
        logger.error(f"Error plotting efficient frontier: {e}")


# Example usage functions
def example_performance_metrics():
    """Example of performance metrics calculations."""
    # Generate sample returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year

    perf_metrics = PerformanceMetrics()

    # Calculate metrics
    cv_result = perf_metrics.coefficient_of_variation(returns)
    sharpe_result = perf_metrics.sharpe_ratio(returns)
    sortino_result = perf_metrics.sortino_ratio(returns)

    return {
        'coefficient_of_variation': cv_result,
        'sharpe_ratio': sharpe_result,
        'sortino_ratio': sortino_result
    }


def example_portfolio_calculations():
    """Example of portfolio calculations."""
    # Generate sample returns for 3 assets
    np.random.seed(42)
    n_assets = 3
    n_periods = 252

    returns = pd.DataFrame(
        np.random.multivariate_normal(
            mean=[0.001, 0.0008, 0.0012],
            cov=[[0.0004, 0.0001, 0.0002],
                 [0.0001, 0.0009, 0.0001],
                 [0.0002, 0.0001, 0.0016]],
            size=n_periods
        ),
        columns=['Stock_A', 'Stock_B', 'Stock_C']
    )

    weights = [0.4, 0.3, 0.3]

    portfolio_calc = PortfolioCalculations()

    # Calculate portfolio metrics
    portfolio_returns = portfolio_calc.weighted_portfolio_returns(returns, weights)
    portfolio_risk = portfolio_calc.portfolio_variance_standard_deviation(returns, weights)
    efficient_frontier = portfolio_calc.efficient_frontier(returns)

    return {
        'portfolio_returns': portfolio_returns,
        'portfolio_risk': portfolio_risk,
        'efficient_frontier': efficient_frontier
    }


def example_capm_analysis():
    """Example of CAPM analysis."""
    # Generate sample data
    np.random.seed(42)
    market_returns = np.random.normal(0.0008, 0.015, 252)
    stock_returns = 0.5 * market_returns + np.random.normal(0, 0.01, 252)

    capm = CAPMAnalysis()

    # Calculate beta
    beta_var_cov = capm.calculate_beta_variance_covariance(stock_returns, market_returns)
    beta_correlation = capm.calculate_beta_correlation(stock_returns, market_returns)

    # Calculate expected return
    beta_value = beta_var_cov['beta']
    expected_return = capm.capm_expected_return(beta_value)

    return {
        'beta_variance_covariance': beta_var_cov,
        'beta_correlation': beta_correlation,
        'expected_return': expected_return
    }


if __name__ == "__main__":
    print("Portfolio Metrics Framework")
    print("=" * 50)

    # Example performance metrics
    print("\n1. Performance Metrics:")
    perf_examples = example_performance_metrics()
    if 'sharpe_ratio' in perf_examples['sharpe_ratio']:
        print(f"Sharpe Ratio: {perf_examples['sharpe_ratio']['annualized_sharpe_ratio']:.3f}")
    if 'sortino_ratio' in perf_examples['sortino_ratio']:
        print(f"Sortino Ratio: {perf_examples['sortino_ratio']['annualized_sortino_ratio']:.3f}")

    # Example portfolio calculations
    print("\n2. Portfolio Analysis:")
    portfolio_examples = example_portfolio_calculations()
    if 'annualized_return' in portfolio_examples['portfolio_returns']:
        print(f"Portfolio Return: {portfolio_examples['portfolio_returns']['annualized_return']:.3f}")
    if 'portfolio_standard_deviation' in portfolio_examples['portfolio_risk']:
        print(f"Portfolio Risk: {portfolio_examples['portfolio_risk']['annualized_standard_deviation']:.3f}")

    # Example CAPM analysis
    print("\n3. CAPM Analysis:")
    capm_examples = example_capm_analysis()
    if 'beta' in capm_examples['beta_variance_covariance']:
        print(f"Beta: {capm_examples['beta_variance_covariance']['beta']:.3f}")
    if 'expected_return' in capm_examples['expected_return']:
        print(f"Expected Return: {camp_examples['expected_return']['expected_return']:.3f}")

    print("\nPortfolio metrics calculations completed!")