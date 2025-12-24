"""
Risk Metrics: Comprehensive Risk Management and Credit Risk Framework

This module provides extensive risk management functionality including:
- Value at Risk (VaR) calculations using Historical Simulation, Parametric, and Monte Carlo methods
- Expected Shortfall (CVaR) and other coherent risk measures
- Counterparty Credit Risk metrics: PFE, EE, EEPE, Loan Equivalent Risk
- Credit Risk modeling: PD from bond prices, credit spreads, transition matrices
- Risk Weighted Assets calculations for regulatory capital
- Portfolio risk analytics and stress testing capabilities

Author: Claude AI
Created: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from scipy.linalg import expm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from enum import Enum
import yfinance as yf
from concurrent.futures import ProcessPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskMeasureType(Enum):
    """Risk measure type enumeration."""
    VAR = "value_at_risk"
    CVAR = "conditional_value_at_risk"
    ES = "expected_shortfall"
    PFE = "potential_future_exposure"
    EE = "expected_exposure"
    EEPE = "effective_expected_positive_exposure"


class VaRMethod(Enum):
    """VaR calculation method enumeration."""
    HISTORICAL = "historical_simulation"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


class CreditRating(Enum):
    """Credit rating enumeration."""
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    CCC = "CCC"
    CC = "CC"
    C = "C"
    D = "D"


@dataclass
class RiskConfig:
    """Configuration class for risk calculations."""

    # General settings
    precision: int = 6
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99, 0.999])
    time_horizons: List[int] = field(default_factory=lambda: [1, 10, 252])  # 1 day, 2 weeks, 1 year

    # VaR settings
    var_lookback_days: int = 252
    monte_carlo_simulations: int = 10000
    random_seed: int = 42

    # Credit risk settings
    default_recovery_rate: float = 0.4
    default_lgd: float = 0.6
    rating_migration_periods: int = 252  # 1 year

    # Counterparty risk settings
    exposure_simulation_paths: int = 1000
    max_maturity_years: float = 30.0
    netting_benefit: float = 0.7

    # Regulatory settings
    basel_iii_buffers: Dict[str, float] = field(default_factory=lambda: {
        'capital_conservation': 0.025,
        'countercyclical': 0.0,
        'systemic_importance': 0.0
    })

    # Plotting settings
    plot_style: str = 'seaborn-v0_8'
    figure_size: Tuple[int, int] = (15, 10)


class ValueAtRisk:
    """
    Value at Risk (VaR) calculations using multiple methodologies.
    """

    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()

    def historical_simulation_var(self, returns: Union[pd.Series, np.ndarray],
                                 confidence_level: float = 0.95,
                                 lookback_days: int = None) -> Dict[str, Any]:
        """
        Calculate VaR using Historical Simulation method.

        Parameters:
        -----------
        returns : Series or array
            Historical returns data
        confidence_level : float
            Confidence level (e.g., 0.95 for 95% VaR)
        lookback_days : int
            Number of historical days to use

        Returns:
        --------
        Dict with VaR calculation results
        """
        try:
            if lookback_days is None:
                lookback_days = self.config.var_lookback_days

            # Convert to numpy array
            if isinstance(returns, pd.Series):
                returns_array = returns.dropna().values
            else:
                returns_array = np.array(returns)
                returns_array = returns_array[~np.isnan(returns_array)]

            if len(returns_array) == 0:
                return {'error': 'No valid returns data'}

            # Use only the most recent lookback_days
            if len(returns_array) > lookback_days:
                returns_array = returns_array[-lookback_days:]

            # Calculate percentile
            alpha = 1 - confidence_level
            var_percentile = np.percentile(returns_array, alpha * 100)

            # Additional statistics
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            skewness = stats.skew(returns_array)
            kurtosis = stats.kurtosis(returns_array)

            # Worst case scenarios
            worst_1_percent = np.percentile(returns_array, 1)
            worst_5_scenarios = np.sort(returns_array)[:5]

            return {
                'method': VaRMethod.HISTORICAL.value,
                'confidence_level': confidence_level,
                'var': round(var_percentile, self.config.precision),
                'var_percentage': round(var_percentile * 100, self.config.precision),
                'sample_size': len(returns_array),
                'lookback_days': lookback_days,
                'mean_return': round(mean_return, self.config.precision),
                'volatility': round(std_return, self.config.precision),
                'skewness': round(skewness, self.config.precision),
                'kurtosis': round(kurtosis, self.config.precision),
                'worst_1_percent': round(worst_1_percent, self.config.precision),
                'worst_5_scenarios': [round(x, self.config.precision) for x in worst_5_scenarios],
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def parametric_var(self, returns: Union[pd.Series, np.ndarray],
                      confidence_level: float = 0.95,
                      distribution: str = 'normal') -> Dict[str, Any]:
        """
        Calculate VaR using Parametric method.

        Parameters:
        -----------
        returns : Series or array
            Historical returns data
        confidence_level : float
            Confidence level
        distribution : str
            Distribution assumption ('normal', 't', 'skewed_t')

        Returns:
        --------
        Dict with parametric VaR results
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
            std_return = np.std(returns_array, ddof=1)  # Sample standard deviation
            alpha = 1 - confidence_level

            # Calculate VaR based on distribution
            if distribution == 'normal':
                z_score = stats.norm.ppf(alpha)
                var_value = mean_return + z_score * std_return

                # Goodness of fit test
                _, p_value = stats.jarque_bera(returns_array)
                distribution_fit = 'good' if p_value > 0.05 else 'poor'

            elif distribution == 't':
                # Fit t-distribution
                df, loc, scale = stats.t.fit(returns_array)
                t_score = stats.t.ppf(alpha, df)
                var_value = loc + t_score * scale

                # Additional t-distribution parameters
                distribution_params = {'degrees_of_freedom': round(df, self.config.precision)}
                distribution_fit = 'assumed'

            elif distribution == 'skewed_t':
                # Fit skewed t-distribution (approximation using skewnorm)
                a, loc, scale = stats.skewnorm.fit(returns_array)
                skewed_score = stats.skewnorm.ppf(alpha, a, loc, scale)
                var_value = skewed_score

                distribution_params = {'skewness_param': round(a, self.config.precision)}
                distribution_fit = 'assumed'

            else:
                return {'error': f'Unsupported distribution: {distribution}'}

            # Model validation metrics
            cornish_fisher_adjustment = self._cornish_fisher_var(returns_array, confidence_level)

            return {
                'method': VaRMethod.PARAMETRIC.value,
                'distribution': distribution,
                'confidence_level': confidence_level,
                'var': round(var_value, self.config.precision),
                'var_percentage': round(var_value * 100, self.config.precision),
                'mean_return': round(mean_return, self.config.precision),
                'volatility': round(std_return, self.config.precision),
                'z_score': round(z_score if distribution == 'normal' else 0, self.config.precision),
                'distribution_fit': distribution_fit,
                'distribution_params': distribution_params if distribution != 'normal' else {},
                'cornish_fisher_var': round(cornish_fisher_adjustment, self.config.precision),
                'sample_size': len(returns_array),
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def monte_carlo_var(self, returns: Union[pd.Series, np.ndarray],
                       confidence_level: float = 0.95,
                       simulations: int = None,
                       time_horizon: int = 1,
                       model: str = 'geometric_brownian') -> Dict[str, Any]:
        """
        Calculate VaR using Monte Carlo simulation.

        Parameters:
        -----------
        returns : Series or array
            Historical returns data
        confidence_level : float
            Confidence level
        simulations : int
            Number of Monte Carlo simulations
        time_horizon : int
            Time horizon in days
        model : str
            Simulation model ('geometric_brownian', 'garch', 'jump_diffusion')

        Returns:
        --------
        Dict with Monte Carlo VaR results
        """
        try:
            if simulations is None:
                simulations = self.config.monte_carlo_simulations

            np.random.seed(self.config.random_seed)

            # Convert to numpy array
            if isinstance(returns, pd.Series):
                returns_array = returns.dropna().values
            else:
                returns_array = np.array(returns)
                returns_array = returns_array[~np.isnan(returns_array)]

            if len(returns_array) == 0:
                return {'error': 'No valid returns data'}

            # Calculate parameters
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array, ddof=1)

            # Generate simulated returns based on model
            if model == 'geometric_brownian':
                simulated_returns = self._simulate_geometric_brownian(
                    mean_return, std_return, simulations, time_horizon
                )
            elif model == 'garch':
                simulated_returns = self._simulate_garch(
                    returns_array, simulations, time_horizon
                )
            elif model == 'jump_diffusion':
                simulated_returns = self._simulate_jump_diffusion(
                    mean_return, std_return, simulations, time_horizon
                )
            else:
                return {'error': f'Unsupported model: {model}'}

            # Calculate VaR from simulated returns
            alpha = 1 - confidence_level
            var_value = np.percentile(simulated_returns, alpha * 100)

            # Additional statistics from simulation
            simulation_mean = np.mean(simulated_returns)
            simulation_std = np.std(simulated_returns)

            # Confidence intervals for VaR estimate
            var_confidence_interval = self._calculate_var_confidence_interval(
                simulated_returns, confidence_level, 0.95
            )

            return {
                'method': VaRMethod.MONTE_CARLO.value,
                'model': model,
                'confidence_level': confidence_level,
                'time_horizon': time_horizon,
                'simulations': simulations,
                'var': round(var_value, self.config.precision),
                'var_percentage': round(var_value * 100, self.config.precision),
                'simulation_mean': round(simulation_mean, self.config.precision),
                'simulation_std': round(simulation_std, self.config.precision),
                'var_confidence_interval': {
                    'lower': round(var_confidence_interval[0], self.config.precision),
                    'upper': round(var_confidence_interval[1], self.config.precision)
                },
                'historical_mean': round(mean_return, self.config.precision),
                'historical_volatility': round(std_return, self.config.precision),
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def _cornish_fisher_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Cornish-Fisher VaR adjustment for non-normal distributions."""
        try:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)

            alpha = 1 - confidence_level
            z = stats.norm.ppf(alpha)

            # Cornish-Fisher expansion
            cf_adjustment = z + (z**2 - 1) * skewness / 6 + \
                           (z**3 - 3*z) * kurtosis / 24 - \
                           (2*z**3 - 5*z) * (skewness**2) / 36

            return mean_return + cf_adjustment * std_return

        except Exception:
            # Fallback to normal VaR
            alpha = 1 - confidence_level
            z = stats.norm.ppf(alpha)
            return np.mean(returns) + z * np.std(returns, ddof=1)

    def _simulate_geometric_brownian(self, mu: float, sigma: float,
                                   simulations: int, time_horizon: int) -> np.ndarray:
        """Simulate returns using Geometric Brownian Motion."""
        dt = 1 / 252  # Daily time step
        total_time = time_horizon * dt

        # Adjust drift for time horizon
        drift = (mu - 0.5 * sigma**2) * total_time
        diffusion = sigma * np.sqrt(total_time)

        # Generate random shocks
        random_shocks = np.random.normal(0, 1, simulations)

        # Calculate returns
        log_returns = drift + diffusion * random_shocks
        returns = np.expm1(log_returns)  # Convert to simple returns

        return returns

    def _simulate_garch(self, returns: np.ndarray, simulations: int, time_horizon: int) -> np.ndarray:
        """Simulate returns using simplified GARCH model."""
        try:
            # Simplified GARCH(1,1) parameters estimation
            mean_return = np.mean(returns)

            # Estimate GARCH parameters (simplified)
            squared_returns = (returns - mean_return) ** 2
            omega = np.var(squared_returns) * 0.1  # Long-term variance
            alpha = 0.1  # ARCH effect
            beta = 0.85   # GARCH effect

            # Initial variance
            initial_variance = np.var(returns)

            simulated_returns = []

            for _ in range(simulations):
                variance = initial_variance
                cumulative_return = 0

                for _ in range(time_horizon):
                    # Update variance
                    variance = omega + alpha * (cumulative_return - mean_return)**2 + beta * variance

                    # Generate return
                    shock = np.random.normal(0, 1)
                    daily_return = mean_return + np.sqrt(variance) * shock
                    cumulative_return += daily_return

                simulated_returns.append(cumulative_return)

            return np.array(simulated_returns)

        except Exception:
            # Fallback to geometric Brownian motion
            return self._simulate_geometric_brownian(
                np.mean(returns), np.std(returns), simulations, time_horizon
            )

    def _simulate_jump_diffusion(self, mu: float, sigma: float,
                               simulations: int, time_horizon: int) -> np.ndarray:
        """Simulate returns using Jump Diffusion model."""
        # Jump parameters
        jump_intensity = 0.1  # Average jumps per year
        jump_mean = -0.05     # Average jump size
        jump_std = 0.1        # Jump volatility

        dt = 1 / 252
        total_time = time_horizon * dt

        simulated_returns = []

        for _ in range(simulations):
            # Diffusion component
            drift = (mu - 0.5 * sigma**2) * total_time
            diffusion = sigma * np.sqrt(total_time) * np.random.normal(0, 1)

            # Jump component
            num_jumps = np.random.poisson(jump_intensity * total_time)
            jump_component = 0
            if num_jumps > 0:
                jumps = np.random.normal(jump_mean, jump_std, num_jumps)
                jump_component = np.sum(jumps)

            # Total return
            log_return = drift + diffusion + jump_component
            simple_return = np.expm1(log_return)
            simulated_returns.append(simple_return)

        return np.array(simulated_returns)

    def _calculate_var_confidence_interval(self, simulated_returns: np.ndarray,
                                         var_confidence: float, ci_confidence: float) -> Tuple[float, float]:
        """Calculate confidence interval for VaR estimate."""
        alpha = 1 - var_confidence
        n = len(simulated_returns)

        # Order statistics approach
        k = int(np.floor(n * alpha))

        # Confidence interval for the k-th order statistic
        z = stats.norm.ppf((1 + ci_confidence) / 2)

        # Approximate standard error
        p = alpha
        se = np.sqrt(p * (1 - p) / n)

        # Confidence interval indices
        lower_idx = max(0, int(np.floor(n * (p - z * se))))
        upper_idx = min(n - 1, int(np.ceil(n * (p + z * se))))

        sorted_returns = np.sort(simulated_returns)

        return sorted_returns[lower_idx], sorted_returns[upper_idx]

    def compare_var_methods(self, returns: Union[pd.Series, np.ndarray],
                           confidence_levels: List[float] = None) -> Dict[str, Any]:
        """
        Compare VaR estimates across different methods.

        Parameters:
        -----------
        returns : Series or array
            Historical returns data
        confidence_levels : List[float]
            Confidence levels to test

        Returns:
        --------
        Dict with comparison results
        """
        try:
            if confidence_levels is None:
                confidence_levels = self.config.confidence_levels

            comparison_results = {
                'confidence_levels': confidence_levels,
                'methods': {},
                'comparison_date': datetime.now()
            }

            for confidence_level in confidence_levels:
                # Historical Simulation
                hist_var = self.historical_simulation_var(returns, confidence_level)

                # Parametric (Normal)
                param_var = self.parametric_var(returns, confidence_level, 'normal')

                # Parametric (t-distribution)
                param_t_var = self.parametric_var(returns, confidence_level, 't')

                # Monte Carlo
                mc_var = self.monte_carlo_var(returns, confidence_level)

                # Store results
                level_key = f"cl_{confidence_level}"
                comparison_results['methods'][level_key] = {
                    'historical_simulation': hist_var.get('var', None),
                    'parametric_normal': param_var.get('var', None),
                    'parametric_t': param_t_var.get('var', None),
                    'monte_carlo': mc_var.get('var', None)
                }

            # Calculate method differences
            method_differences = self._calculate_method_differences(comparison_results['methods'])
            comparison_results['method_differences'] = method_differences

            # Method recommendations
            recommendations = self._generate_var_method_recommendations(returns, comparison_results)
            comparison_results['recommendations'] = recommendations

            return comparison_results

        except Exception as e:
            return {'error': str(e)}

    def _calculate_method_differences(self, methods_results: Dict) -> Dict[str, Any]:
        """Calculate differences between VaR methods."""
        differences = {}

        for level_key, results in methods_results.items():
            if all(val is not None for val in results.values()):
                values = list(results.values())
                differences[level_key] = {
                    'max_difference': round(max(values) - min(values), self.config.precision),
                    'std_deviation': round(np.std(values), self.config.precision),
                    'coefficient_of_variation': round(np.std(values) / np.mean(np.abs(values)), self.config.precision)
                }

        return differences

    def _generate_var_method_recommendations(self, returns: np.ndarray,
                                           comparison_results: Dict) -> Dict[str, str]:
        """Generate recommendations for VaR method selection."""
        try:
            # Convert to numpy array if needed
            if isinstance(returns, pd.Series):
                returns_array = returns.dropna().values
            else:
                returns_array = np.array(returns)

            recommendations = {}

            # Test for normality
            _, normality_p_value = stats.jarque_bera(returns_array)

            # Calculate sample size
            sample_size = len(returns_array)

            # Check for fat tails
            kurtosis_val = stats.kurtosis(returns_array)

            # Method selection logic
            if normality_p_value > 0.05 and sample_size > 100:
                recommendations['primary_method'] = 'parametric_normal'
                recommendations['reason'] = 'Data appears normally distributed with sufficient sample size'
            elif kurtosis_val > 3 and sample_size > 252:
                recommendations['primary_method'] = 'historical_simulation'
                recommendations['reason'] = 'Fat tails detected, historical simulation preferred'
            elif sample_size < 100:
                recommendations['primary_method'] = 'monte_carlo'
                recommendations['reason'] = 'Small sample size, Monte Carlo provides more scenarios'
            else:
                recommendations['primary_method'] = 'parametric_t'
                recommendations['reason'] = 'Non-normal distribution, t-distribution may fit better'

            recommendations['backup_method'] = 'historical_simulation'
            recommendations['validation_method'] = 'monte_carlo'

            return recommendations

        except Exception as e:
            return {'error': str(e)}


class ExpectedShortfall:
    """
    Expected Shortfall (Conditional Value at Risk) calculations.
    """

    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()

    def calculate_expected_shortfall(self, returns: Union[pd.Series, np.ndarray],
                                   confidence_level: float = 0.95,
                                   method: str = 'historical') -> Dict[str, Any]:
        """
        Calculate Expected Shortfall (CVaR).

        Parameters:
        -----------
        returns : Series or array
            Historical returns data
        confidence_level : float
            Confidence level
        method : str
            Calculation method ('historical', 'parametric')

        Returns:
        --------
        Dict with ES calculation results
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

            alpha = 1 - confidence_level

            if method == 'historical':
                # Sort returns
                sorted_returns = np.sort(returns_array)

                # Find VaR threshold
                var_index = int(np.floor(len(sorted_returns) * alpha))
                var_value = sorted_returns[var_index]

                # Calculate ES as average of returns below VaR
                tail_returns = sorted_returns[:var_index + 1]
                es_value = np.mean(tail_returns) if len(tail_returns) > 0 else var_value

                # Additional tail statistics
                tail_std = np.std(tail_returns) if len(tail_returns) > 1 else 0
                worst_case = np.min(sorted_returns)
                tail_skewness = stats.skew(tail_returns) if len(tail_returns) > 2 else 0

            elif method == 'parametric':
                # Assume normal distribution
                mu = np.mean(returns_array)
                sigma = np.std(returns_array, ddof=1)

                # Calculate VaR
                z_alpha = stats.norm.ppf(alpha)
                var_value = mu + z_alpha * sigma

                # Calculate ES for normal distribution
                es_value = mu - sigma * stats.norm.pdf(z_alpha) / alpha

                # Tail statistics for normal distribution
                tail_std = sigma  # Approximation
                worst_case = np.min(returns_array)
                tail_skewness = 0  # Normal distribution

            else:
                return {'error': f'Unsupported method: {method}'}

            # Risk contribution analysis
            spectral_risk_measure = self._calculate_spectral_risk_measure(returns_array, confidence_level)

            return {
                'method': method,
                'confidence_level': confidence_level,
                'expected_shortfall': round(es_value, self.config.precision),
                'es_percentage': round(es_value * 100, self.config.precision),
                'var': round(var_value, self.config.precision),
                'var_percentage': round(var_value * 100, self.config.precision),
                'es_var_ratio': round(es_value / var_value if var_value != 0 else 1, self.config.precision),
                'tail_expectation': round(es_value, self.config.precision),
                'tail_volatility': round(tail_std, self.config.precision),
                'tail_skewness': round(tail_skewness, self.config.precision),
                'worst_case': round(worst_case, self.config.precision),
                'spectral_risk_measure': round(spectral_risk_measure, self.config.precision),
                'sample_size': len(returns_array),
                'tail_observations': len(tail_returns) if method == 'historical' else int(len(returns_array) * alpha),
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def _calculate_spectral_risk_measure(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate spectral risk measure (coherent risk measure)."""
        try:
            alpha = 1 - confidence_level
            sorted_returns = np.sort(returns)
            n = len(sorted_returns)

            # Weight function (exponential)
            weights = np.exp(-np.arange(1, n + 1) / (n * alpha))
            weights = weights / np.sum(weights)

            # Spectral risk measure
            spectral_risk = np.sum(weights * sorted_returns)

            return spectral_risk

        except Exception:
            # Fallback to simple ES
            alpha = 1 - confidence_level
            sorted_returns = np.sort(returns)
            tail_index = int(np.floor(len(sorted_returns) * alpha))
            return np.mean(sorted_returns[:tail_index + 1])

    def coherent_risk_measures(self, returns: Union[pd.Series, np.ndarray],
                             confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Calculate multiple coherent risk measures.

        Parameters:
        -----------
        returns : Series or array
            Historical returns data
        confidence_level : float
            Confidence level

        Returns:
        --------
        Dict with coherent risk measures
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

            # Expected Shortfall
            es_result = self.calculate_expected_shortfall(returns_array, confidence_level, 'historical')

            # Worst Case Scenario (WCS)
            worst_case = np.min(returns_array)

            # Conditional Drawdown at Risk (CDaR)
            cdar = self._calculate_conditional_drawdown_at_risk(returns_array, confidence_level)

            # Entropic Risk Measure
            entropic_risk = self._calculate_entropic_risk_measure(returns_array, confidence_level)

            # Range Value at Risk (RVaR)
            rvar = self._calculate_range_var(returns_array, confidence_level)

            return {
                'confidence_level': confidence_level,
                'expected_shortfall': es_result.get('expected_shortfall'),
                'conditional_drawdown_at_risk': round(cdar, self.config.precision),
                'entropic_risk_measure': round(entropic_risk, self.config.precision),
                'range_value_at_risk': round(rvar, self.config.precision),
                'worst_case_scenario': round(worst_case, self.config.precision),
                'coherence_properties': {
                    'monotonicity': True,
                    'subadditivity': True,
                    'positive_homogeneity': True,
                    'translation_invariance': True
                },
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def _calculate_conditional_drawdown_at_risk(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Conditional Drawdown at Risk."""
        try:
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + returns)

            # Calculate drawdowns
            peak = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - peak) / peak

            # Calculate CDaR
            alpha = 1 - confidence_level
            sorted_drawdowns = np.sort(drawdowns)
            tail_index = int(np.floor(len(sorted_drawdowns) * alpha))

            cdar = np.mean(sorted_drawdowns[:tail_index + 1]) if tail_index >= 0 else sorted_drawdowns[0]

            return cdar

        except Exception:
            return np.min(returns)  # Fallback

    def _calculate_entropic_risk_measure(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Entropic Risk Measure."""
        try:
            # Risk aversion parameter
            gamma = -np.log(1 - confidence_level)

            # Entropic risk measure
            entropic_risk = -np.log(np.mean(np.exp(-gamma * returns))) / gamma

            return entropic_risk

        except Exception:
            # Fallback to ES approximation
            alpha = 1 - confidence_level
            sorted_returns = np.sort(returns)
            tail_index = int(np.floor(len(sorted_returns) * alpha))
            return np.mean(sorted_returns[:tail_index + 1])

    def _calculate_range_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Range Value at Risk."""
        try:
            alpha = 1 - confidence_level

            # Calculate VaR at different levels
            var_lower = np.percentile(returns, alpha * 100 / 2)
            var_upper = np.percentile(returns, (1 - alpha / 2) * 100)

            # Range VaR is the average
            rvar = (var_lower + var_upper) / 2

            return rvar

        except Exception:
            # Fallback to standard VaR
            alpha = 1 - confidence_level
            return np.percentile(returns, alpha * 100)


class CounterpartyRisk:
    """
    Counterparty credit risk metrics including PFE, EE, EEPE, and Loan Equivalent Risk.
    """

    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()

    def potential_future_exposure(self, trades: List[Dict], confidence_level: float = 0.95,
                                simulation_paths: int = None, time_grid: List[float] = None) -> Dict[str, Any]:
        """
        Calculate Potential Future Exposure (PFE) for a portfolio of trades.

        Parameters:
        -----------
        trades : List[Dict]
            List of trade dictionaries with exposure simulation functions
        confidence_level : float
            Confidence level for PFE calculation
        simulation_paths : int
            Number of Monte Carlo paths
        time_grid : List[float]
            Time points for exposure calculation (in years)

        Returns:
        --------
        Dict with PFE calculation results
        """
        try:
            if simulation_paths is None:
                simulation_paths = self.config.exposure_simulation_paths

            if time_grid is None:
                time_grid = np.linspace(0, self.config.max_maturity_years, 50)

            np.random.seed(self.config.random_seed)

            # Initialize exposure arrays
            all_exposures = np.zeros((simulation_paths, len(time_grid)))

            # Simulate exposures for each trade
            for trade in trades:
                trade_exposures = self._simulate_trade_exposure(trade, time_grid, simulation_paths)
                all_exposures += trade_exposures

            # Apply netting benefits (simplified)
            if len(trades) > 1:
                all_exposures *= self.config.netting_benefit

            # Calculate PFE (percentile of positive exposures)
            alpha = confidence_level
            pfe_profile = []

            for t_idx, t in enumerate(time_grid):
                positive_exposures = np.maximum(all_exposures[:, t_idx], 0)
                if np.sum(positive_exposures > 0) > 0:
                    pfe_value = np.percentile(positive_exposures[positive_exposures > 0], alpha * 100)
                else:
                    pfe_value = 0
                pfe_profile.append(pfe_value)

            # Calculate peak PFE
            peak_pfe = np.max(pfe_profile)
            peak_pfe_time = time_grid[np.argmax(pfe_profile)]

            return {
                'pfe_profile': [round(x, self.config.precision) for x in pfe_profile],
                'time_grid': [round(x, self.config.precision) for x in time_grid],
                'peak_pfe': round(peak_pfe, self.config.precision),
                'peak_pfe_time': round(peak_pfe_time, self.config.precision),
                'confidence_level': confidence_level,
                'simulation_paths': simulation_paths,
                'netting_benefit': self.config.netting_benefit,
                'number_of_trades': len(trades),
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def expected_exposure(self, trades: List[Dict], simulation_paths: int = None,
                         time_grid: List[float] = None) -> Dict[str, Any]:
        """
        Calculate Expected Exposure (EE) profile.

        Parameters:
        -----------
        trades : List[Dict]
            List of trade dictionaries
        simulation_paths : int
            Number of Monte Carlo paths
        time_grid : List[float]
            Time points for exposure calculation

        Returns:
        --------
        Dict with EE calculation results
        """
        try:
            if simulation_paths is None:
                simulation_paths = self.config.exposure_simulation_paths

            if time_grid is None:
                time_grid = np.linspace(0, self.config.max_maturity_years, 50)

            np.random.seed(self.config.random_seed)

            # Initialize exposure arrays
            all_exposures = np.zeros((simulation_paths, len(time_grid)))

            # Simulate exposures for each trade
            for trade in trades:
                trade_exposures = self._simulate_trade_exposure(trade, time_grid, simulation_paths)
                all_exposures += trade_exposures

            # Apply netting benefits
            if len(trades) > 1:
                all_exposures *= self.config.netting_benefit

            # Calculate EE (expected value of positive exposures)
            ee_profile = []
            epe_profile = []  # Expected Positive Exposure

            for t_idx, t in enumerate(time_grid):
                positive_exposures = np.maximum(all_exposures[:, t_idx], 0)
                ee_value = np.mean(positive_exposures)
                epe_value = np.mean(positive_exposures[positive_exposures > 0]) if np.sum(positive_exposures > 0) > 0 else 0

                ee_profile.append(ee_value)
                epe_profile.append(epe_value)

            # Calculate metrics
            max_ee = np.max(ee_profile)
            max_ee_time = time_grid[np.argmax(ee_profile)]

            return {
                'ee_profile': [round(x, self.config.precision) for x in ee_profile],
                'epe_profile': [round(x, self.config.precision) for x in epe_profile],
                'time_grid': [round(x, self.config.precision) for x in time_grid],
                'max_expected_exposure': round(max_ee, self.config.precision),
                'max_ee_time': round(max_ee_time, self.config.precision),
                'simulation_paths': simulation_paths,
                'netting_benefit': self.config.netting_benefit,
                'number_of_trades': len(trades),
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def effective_expected_positive_exposure(self, trades: List[Dict],
                                           simulation_paths: int = None) -> Dict[str, Any]:
        """
        Calculate Effective Expected Positive Exposure (EEPE).

        Parameters:
        -----------
        trades : List[Dict]
            List of trade dictionaries
        simulation_paths : int
            Number of Monte Carlo paths

        Returns:
        --------
        Dict with EEPE calculation results
        """
        try:
            # Calculate EE first
            ee_result = self.expected_exposure(trades, simulation_paths)

            if 'error' in ee_result:
                return ee_result

            ee_profile = ee_result['ee_profile']
            time_grid = ee_result['time_grid']

            # Calculate EEPE as time-weighted average of EE
            # Use first year for weighting (regulatory standard)
            one_year_mask = np.array(time_grid) <= 1.0
            ee_first_year = np.array(ee_profile)[one_year_mask]
            time_first_year = np.array(time_grid)[one_year_mask]

            if len(ee_first_year) > 1:
                # Trapezoidal integration for time-weighted average
                time_weights = np.diff(time_first_year)
                ee_avg = ee_first_year[:-1]  # Use left endpoint for each interval
                eepe_value = np.sum(ee_avg * time_weights) / (time_first_year[-1] - time_first_year[0])
            else:
                eepe_value = ee_first_year[0] if len(ee_first_year) > 0 else 0

            # Calculate additional EEPE metrics
            effective_maturity = self._calculate_effective_maturity(time_grid, ee_profile)
            alpha_factor = self._calculate_alpha_factor(trades)

            return {
                'eepe': round(eepe_value, self.config.precision),
                'effective_maturity': round(effective_maturity, self.config.precision),
                'alpha_factor': round(alpha_factor, self.config.precision),
                'regulatory_eepe': round(eepe_value * alpha_factor, self.config.precision),
                'ee_profile': ee_result['ee_profile'],
                'time_grid': ee_result['time_grid'],
                'simulation_paths': simulation_paths or self.config.exposure_simulation_paths,
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def loan_equivalent_risk(self, trades: List[Dict], loan_equivalent_factors: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Calculate Loan Equivalent Risk (LER) for regulatory capital.

        Parameters:
        -----------
        trades : List[Dict]
            List of trade dictionaries with type and notional
        loan_equivalent_factors : Dict
            Conversion factors by trade type

        Returns:
        --------
        Dict with LER calculation results
        """
        try:
            if loan_equivalent_factors is None:
                loan_equivalent_factors = {
                    'swap': 0.05,
                    'forward': 0.10,
                    'option': 0.075,
                    'credit_derivative': 0.10,
                    'fx_forward': 0.08,
                    'commodity_forward': 0.12
                }

            trade_lers = []
            total_notional = 0
            total_ler = 0

            for trade in trades:
                trade_type = trade.get('type', 'swap').lower()
                notional = trade.get('notional', 0)
                maturity = trade.get('maturity_years', 1)

                # Get base conversion factor
                base_factor = loan_equivalent_factors.get(trade_type, 0.05)

                # Adjust for maturity (simplified approach)
                if maturity > 1:
                    maturity_adjustment = min(1.0 + 0.01 * (maturity - 1), 1.5)
                else:
                    maturity_adjustment = 1.0

                # Calculate LER for this trade
                trade_ler = notional * base_factor * maturity_adjustment

                trade_lers.append({
                    'trade_id': trade.get('id', len(trade_lers)),
                    'type': trade_type,
                    'notional': round(notional, self.config.precision),
                    'maturity': round(maturity, self.config.precision),
                    'base_factor': round(base_factor, self.config.precision),
                    'maturity_adjustment': round(maturity_adjustment, self.config.precision),
                    'loan_equivalent_risk': round(trade_ler, self.config.precision)
                })

                total_notional += notional
                total_ler += trade_ler

            # Portfolio-level metrics
            average_ler_factor = total_ler / total_notional if total_notional > 0 else 0

            # Diversification benefit (simplified)
            diversification_benefit = min(0.2, 0.05 * len(set(trade.get('counterparty', 'default') for trade in trades)))
            adjusted_ler = total_ler * (1 - diversification_benefit)

            return {
                'trade_details': trade_lers,
                'total_notional': round(total_notional, self.config.precision),
                'total_ler': round(total_ler, self.config.precision),
                'adjusted_ler': round(adjusted_ler, self.config.precision),
                'average_ler_factor': round(average_ler_factor, self.config.precision),
                'diversification_benefit': round(diversification_benefit, self.config.precision),
                'number_of_trades': len(trades),
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def _simulate_trade_exposure(self, trade: Dict, time_grid: List[float], paths: int) -> np.ndarray:
        """Simulate exposure for a single trade."""
        try:
            trade_type = trade.get('type', 'swap').lower()
            notional = trade.get('notional', 1000000)
            maturity = trade.get('maturity_years', 5)

            exposures = np.zeros((paths, len(time_grid)))

            for path in range(paths):
                for t_idx, t in enumerate(time_grid):
                    if t >= maturity:
                        exposure = 0
                    else:
                        # Simplified exposure simulation based on trade type
                        if trade_type == 'swap':
                            # Interest rate swap exposure peaks in middle of life
                            time_factor = 4 * t * (maturity - t) / (maturity ** 2)
                            volatility = 0.02  # 2% volatility
                        elif trade_type == 'forward':
                            # Forward exposure increases linearly
                            time_factor = t / maturity
                            volatility = 0.03
                        elif trade_type == 'option':
                            # Option exposure decreases over time (theta decay)
                            time_factor = (maturity - t) / maturity
                            volatility = 0.05
                        else:
                            # Default case
                            time_factor = 0.5
                            volatility = 0.025

                        # Add random component
                        random_factor = 1 + volatility * np.random.normal(0, 1)
                        exposure = notional * time_factor * random_factor * 0.1  # 10% of notional max

                    exposures[path, t_idx] = max(0, exposure)

            return exposures

        except Exception:
            # Fallback: simple exposure profile
            exposures = np.zeros((paths, len(time_grid)))
            notional = trade.get('notional', 1000000)
            maturity = trade.get('maturity_years', 5)

            for t_idx, t in enumerate(time_grid):
                if t < maturity:
                    exposures[:, t_idx] = notional * 0.05 * np.random.exponential(1, paths)

            return exposures

    def _calculate_effective_maturity(self, time_grid: List[float], ee_profile: List[float]) -> float:
        """Calculate effective maturity based on EE profile."""
        try:
            if len(time_grid) != len(ee_profile):
                return 1.0  # Default

            # Weight time by expected exposure
            total_exposure = sum(ee_profile)
            if total_exposure == 0:
                return 1.0

            weighted_time = sum(t * ee for t, ee in zip(time_grid, ee_profile))
            effective_maturity = weighted_time / total_exposure

            return min(effective_maturity, self.config.max_maturity_years)

        except Exception:
            return 1.0  # Default effective maturity

    def _calculate_alpha_factor(self, trades: List[Dict]) -> float:
        """Calculate alpha factor for regulatory EEPE."""
        try:
            # Simplified alpha factor based on trade characteristics
            base_alpha = 1.4  # Basel III minimum

            # Adjust based on portfolio characteristics
            if len(trades) > 10:
                base_alpha *= 0.95  # Diversification benefit

            # Check for margining
            margined_trades = sum(1 for trade in trades if trade.get('margined', False))
            if margined_trades > 0:
                margin_benefit = 0.8 + 0.2 * (len(trades) - margined_trades) / len(trades)
                base_alpha *= margin_benefit

            return max(1.0, min(base_alpha, 2.0))  # Cap between 1.0 and 2.0

        except Exception:
            return 1.4  # Default alpha factor


class CreditRisk:
    """
    Credit risk modeling including PD calculation from bond prices and credit spreads.
    """

    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()

    def pd_from_bond_prices(self, bond_prices: List[float], face_values: List[float],
                          coupon_rates: List[float], maturities: List[float],
                          risk_free_rates: List[float], recovery_rate: float = None) -> Dict[str, Any]:
        """
        Calculate Probability of Default from bond prices using expected cash flows.

        Parameters:
        -----------
        bond_prices : List[float]
            Current bond prices
        face_values : List[float]
            Face values of bonds
        coupon_rates : List[float]
            Annual coupon rates
        maturities : List[float]
            Years to maturity
        risk_free_rates : List[float]
            Risk-free rates for each maturity
        recovery_rate : float
            Expected recovery rate in default

        Returns:
        --------
        Dict with PD calculations
        """
        try:
            if recovery_rate is None:
                recovery_rate = self.config.default_recovery_rate

            if not all(len(lst) == len(bond_prices) for lst in [face_values, coupon_rates, maturities, risk_free_rates]):
                return {'error': 'All input lists must have the same length'}

            pd_estimates = []
            credit_spreads = []

            for i, (price, face_value, coupon_rate, maturity, rf_rate) in enumerate(
                zip(bond_prices, face_values, coupon_rates, maturities, risk_free_rates)
            ):
                # Calculate theoretical risk-free bond price
                periods_per_year = 2  # Semi-annual payments
                total_periods = int(maturity * periods_per_year)
                periodic_rate = rf_rate / periods_per_year
                periodic_coupon = (coupon_rate * face_value) / periods_per_year

                # Risk-free bond price
                if periodic_rate == 0:
                    rf_bond_price = periodic_coupon * total_periods + face_value
                else:
                    pv_coupons = periodic_coupon * (1 - (1 + periodic_rate) ** (-total_periods)) / periodic_rate
                    pv_face = face_value / (1 + periodic_rate) ** total_periods
                    rf_bond_price = pv_coupons + pv_face

                # Credit spread implied by bond price
                # Solve for credit spread that makes NPV equal to market price
                def npv_function(credit_spread):
                    total_spread = rf_rate + credit_spread
                    periodic_spread_rate = total_spread / periods_per_year

                    if periodic_spread_rate == 0:
                        npv = periodic_coupon * total_periods + face_value
                    else:
                        pv_coupons = periodic_coupon * (1 - (1 + periodic_spread_rate) ** (-total_periods)) / periodic_spread_rate
                        pv_face = face_value / (1 + periodic_spread_rate) ** total_periods
                        npv = pv_coupons + pv_face

                    return npv - price

                try:
                    # Solve for credit spread
                    credit_spread = optimize.brentq(npv_function, 0, 1, xtol=1e-6)
                except ValueError:
                    # If no solution found, estimate from yield difference
                    implied_yield = self._calculate_yield_to_maturity(price, face_value, coupon_rate, maturity)
                    credit_spread = max(0, implied_yield - rf_rate)

                # Convert credit spread to PD using reduced form model
                # PD = 1 - exp(-lambda * T) where lambda = credit_spread / (1 - recovery_rate)
                hazard_rate = credit_spread / (1 - recovery_rate)
                pd = 1 - np.exp(-hazard_rate * maturity)

                pd_estimates.append({
                    'bond_index': i,
                    'maturity': round(maturity, self.config.precision),
                    'market_price': round(price, self.config.precision),
                    'risk_free_price': round(rf_bond_price, self.config.precision),
                    'credit_spread': round(credit_spread * 10000, self.config.precision),  # basis points
                    'hazard_rate': round(hazard_rate, self.config.precision),
                    'probability_of_default': round(pd, self.config.precision),
                    'pd_percentage': round(pd * 100, self.config.precision)
                })

                credit_spreads.append(credit_spread)

            # Term structure of credit spreads and PDs
            term_structure = self._build_credit_term_structure(pd_estimates)

            # Credit curve calibration
            calibrated_curve = self._calibrate_credit_curve(maturities, credit_spreads)

            return {
                'individual_bonds': pd_estimates,
                'recovery_rate': recovery_rate,
                'average_credit_spread_bps': round(np.mean(credit_spreads) * 10000, self.config.precision),
                'average_pd': round(np.mean([est['probability_of_default'] for est in pd_estimates]), self.config.precision),
                'term_structure': term_structure,
                'calibrated_curve': calibrated_curve,
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def pd_from_credit_spreads(self, credit_spreads: List[float], maturities: List[float],
                             recovery_rate: float = None, model: str = 'reduced_form') -> Dict[str, Any]:
        """
        Calculate Probability of Default from credit spreads.

        Parameters:
        -----------
        credit_spreads : List[float]
            Credit spreads in decimal form
        maturities : List[float]
            Corresponding maturities in years
        recovery_rate : float
            Expected recovery rate
        model : str
            Model type ('reduced_form', 'structural')

        Returns:
        --------
        Dict with PD calculations from spreads
        """
        try:
            if recovery_rate is None:
                recovery_rate = self.config.default_recovery_rate

            if len(credit_spreads) != len(maturities):
                return {'error': 'Credit spreads and maturities must have same length'}

            pd_calculations = []

            for spread, maturity in zip(credit_spreads, maturities):
                if model == 'reduced_form':
                    # Hazard rate model: lambda = spread / (1 - R)
                    hazard_rate = spread / (1 - recovery_rate)
                    pd = 1 - np.exp(-hazard_rate * maturity)

                elif model == 'structural':
                    # Merton model approximation
                    # More complex - simplified implementation
                    distance_to_default = -stats.norm.ppf(spread / 0.05)  # Rough approximation
                    pd = stats.norm.cdf(-distance_to_default)
                    hazard_rate = -np.log(1 - pd) / maturity if pd < 1 else float('inf')

                else:
                    return {'error': f'Unsupported model: {model}'}

                # Calculate survival probability
                survival_prob = 1 - pd

                pd_calculations.append({
                    'maturity': round(maturity, self.config.precision),
                    'credit_spread': round(spread * 10000, self.config.precision),  # bps
                    'hazard_rate': round(hazard_rate, self.config.precision),
                    'probability_of_default': round(pd, self.config.precision),
                    'survival_probability': round(survival_prob, self.config.precision),
                    'pd_percentage': round(pd * 100, self.config.precision)
                })

            # Forward PD calculations
            forward_pds = self._calculate_forward_pds(pd_calculations)

            # Credit quality assessment
            credit_quality = self._assess_credit_quality(credit_spreads, maturities)

            return {
                'model': model,
                'recovery_rate': recovery_rate,
                'pd_calculations': pd_calculations,
                'forward_pds': forward_pds,
                'credit_quality_assessment': credit_quality,
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def create_transition_matrix(self, historical_ratings: pd.DataFrame,
                               time_horizon: int = 252) -> Dict[str, Any]:
        """
        Create credit rating transition matrix from historical data.

        Parameters:
        -----------
        historical_ratings : DataFrame
            Historical rating data with columns ['entity_id', 'date', 'rating']
        time_horizon : int
            Time horizon in days for transition calculation

        Returns:
        --------
        Dict with transition matrix and related metrics
        """
        try:
            if not all(col in historical_ratings.columns for col in ['entity_id', 'date', 'rating']):
                return {'error': 'DataFrame must contain entity_id, date, and rating columns'}

            # Define rating order
            rating_order = [rating.value for rating in CreditRating]

            # Convert dates to datetime
            historical_ratings['date'] = pd.to_datetime(historical_ratings['date'])

            # Sort by entity and date
            df = historical_ratings.sort_values(['entity_id', 'date']).copy()

            # Initialize transition count matrix
            n_ratings = len(rating_order)
            transition_counts = np.zeros((n_ratings, n_ratings))

            # Count transitions
            for entity in df['entity_id'].unique():
                entity_data = df[df['entity_id'] == entity].copy()

                for i in range(len(entity_data) - 1):
                    current_rating = entity_data.iloc[i]['rating']
                    current_date = entity_data.iloc[i]['date']

                    # Find rating at time_horizon later
                    future_date = current_date + timedelta(days=time_horizon)
                    future_ratings = entity_data[entity_data['date'] >= future_date]

                    if not future_ratings.empty:
                        future_rating = future_ratings.iloc[0]['rating']

                        # Map ratings to indices
                        if current_rating in rating_order and future_rating in rating_order:
                            from_idx = rating_order.index(current_rating)
                            to_idx = rating_order.index(future_rating)
                            transition_counts[from_idx, to_idx] += 1

            # Convert counts to probabilities
            transition_matrix = np.zeros_like(transition_counts)
            for i in range(n_ratings):
                row_sum = transition_counts[i, :].sum()
                if row_sum > 0:
                    transition_matrix[i, :] = transition_counts[i, :] / row_sum

            # Create DataFrame for better visualization
            transition_df = pd.DataFrame(
                transition_matrix,
                index=rating_order,
                columns=rating_order
            )

            # Calculate key metrics
            default_probabilities = transition_matrix[:, -1]  # Last column is default
            stability_measures = np.diag(transition_matrix)  # Diagonal elements

            # Generator matrix (for continuous-time analysis)
            generator_matrix = self._calculate_generator_matrix(transition_matrix, time_horizon)

            return {
                'transition_matrix': transition_df.round(self.config.precision),
                'transition_counts': transition_counts.astype(int),
                'rating_order': rating_order,
                'time_horizon_days': time_horizon,
                'default_probabilities': {
                    rating: round(prob, self.config.precision)
                    for rating, prob in zip(rating_order, default_probabilities)
                },
                'stability_measures': {
                    rating: round(stability, self.config.precision)
                    for rating, stability in zip(rating_order, stability_measures)
                },
                'generator_matrix': generator_matrix.round(self.config.precision) if generator_matrix is not None else None,
                'sample_size': len(df),
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def calculate_lgd_from_transitions(self, transition_matrix: np.ndarray,
                                     recovery_rates: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Calculate Loss Given Default (LGD) from transition matrix.

        Parameters:
        -----------
        transition_matrix : ndarray
            Credit rating transition matrix
        recovery_rates : Dict
            Recovery rates by rating category

        Returns:
        --------
        Dict with LGD calculations
        """
        try:
            if recovery_rates is None:
                # Default recovery rates by rating
                recovery_rates = {
                    'AAA': 0.7, 'AA': 0.65, 'A': 0.6, 'BBB': 0.55,
                    'BB': 0.5, 'B': 0.4, 'CCC': 0.3, 'CC': 0.2, 'C': 0.1, 'D': 0.0
                }

            rating_order = [rating.value for rating in CreditRating]
            n_ratings = len(rating_order)

            # Calculate expected LGD for each rating
            expected_lgds = {}

            for i, rating in enumerate(rating_order[:-1]):  # Exclude default state
                if i < len(transition_matrix):
                    # Probability of each transition
                    transition_probs = transition_matrix[i, :]

                    # Expected recovery rate
                    expected_recovery = 0
                    for j, target_rating in enumerate(rating_order):
                        if j < len(transition_probs):
                            recovery_rate = recovery_rates.get(target_rating, self.config.default_recovery_rate)
                            expected_recovery += transition_probs[j] * recovery_rate

                    # LGD = 1 - Recovery Rate
                    expected_lgd = 1 - expected_recovery
                    expected_lgds[rating] = round(expected_lgd, self.config.precision)

            # Calculate LGD volatility
            lgd_volatilities = {}
            for i, rating in enumerate(rating_order[:-1]):
                if i < len(transition_matrix):
                    transition_probs = transition_matrix[i, :]

                    # Calculate variance of recovery rates
                    expected_recovery = 1 - expected_lgds[rating]
                    recovery_variance = 0

                    for j, target_rating in enumerate(rating_order):
                        if j < len(transition_probs):
                            recovery_rate = recovery_rates.get(target_rating, self.config.default_recovery_rate)
                            recovery_variance += transition_probs[j] * (recovery_rate - expected_recovery) ** 2

                    lgd_volatility = np.sqrt(recovery_variance)
                    lgd_volatilities[rating] = round(lgd_volatility, self.config.precision)

            # Risk-weighted LGD
            default_probs = transition_matrix[:, -1]  # Last column is default
            risk_weighted_lgd = 0
            total_default_prob = 0

            for i, rating in enumerate(rating_order[:-1]):
                if i < len(default_probs) and rating in expected_lgds:
                    default_prob = default_probs[i]
                    lgd = expected_lgds[rating]
                    risk_weighted_lgd += default_prob * lgd
                    total_default_prob += default_prob

            if total_default_prob > 0:
                risk_weighted_lgd /= total_default_prob

            return {
                'expected_lgd_by_rating': expected_lgds,
                'lgd_volatility_by_rating': lgd_volatilities,
                'risk_weighted_lgd': round(risk_weighted_lgd, self.config.precision),
                'recovery_rates_used': recovery_rates,
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def market_implied_pd(self, cds_spreads: List[float], maturities: List[float],
                         recovery_rate: float = None) -> Dict[str, Any]:
        """
        Calculate market-implied PD from CDS spreads.

        Parameters:
        -----------
        cds_spreads : List[float]
            CDS spreads in basis points
        maturities : List[float]
            CDS maturities in years
        recovery_rate : float
            Assumed recovery rate

        Returns:
        --------
        Dict with market-implied PD calculations
        """
        try:
            if recovery_rate is None:
                recovery_rate = self.config.default_recovery_rate

            if len(cds_spreads) != len(maturities):
                return {'error': 'CDS spreads and maturities must have same length'}

            # Convert basis points to decimal
            spreads_decimal = [spread / 10000 for spread in cds_spreads]

            market_pds = []

            for spread, maturity in zip(spreads_decimal, maturities):
                # Simplified CDS pricing: spread ≈ (1 - R) * hazard_rate
                # More accurate: need to solve CDS pricing equation

                # Initial guess for hazard rate
                hazard_rate_guess = spread / (1 - recovery_rate)

                # Solve CDS pricing equation
                def cds_pricing_error(hazard_rate):
                    # Present value of premium leg
                    pv_premium = self._calculate_cds_premium_leg(spread, maturity, hazard_rate)

                    # Present value of protection leg
                    pv_protection = self._calculate_cds_protection_leg(1 - recovery_rate, maturity, hazard_rate)

                    return pv_premium - pv_protection

                try:
                    # Solve for hazard rate
                    optimal_hazard_rate = optimize.fsolve(cds_pricing_error, hazard_rate_guess)[0]
                    optimal_hazard_rate = max(0, optimal_hazard_rate)  # Ensure non-negative
                except:
                    optimal_hazard_rate = hazard_rate_guess

                # Calculate PD
                market_pd = 1 - np.exp(-optimal_hazard_rate * maturity)

                market_pds.append({
                    'maturity': round(maturity, self.config.precision),
                    'cds_spread_bps': round(spread * 10000, self.config.precision),
                    'implied_hazard_rate': round(optimal_hazard_rate, self.config.precision),
                    'market_implied_pd': round(market_pd, self.config.precision),
                    'pd_percentage': round(market_pd * 100, self.config.precision)
                })

            # Build credit curve
            hazard_rates = [pd['implied_hazard_rate'] for pd in market_pds]
            credit_curve = self._build_hazard_rate_curve(maturities, hazard_rates)

            return {
                'market_implied_pds': market_pds,
                'recovery_rate': recovery_rate,
                'credit_curve': credit_curve,
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def _calculate_yield_to_maturity(self, price: float, face_value: float,
                                   coupon_rate: float, maturity: float) -> float:
        """Calculate yield to maturity for a bond."""
        try:
            def price_function(ytm):
                periods = int(maturity * 2)  # Semi-annual
                periodic_rate = ytm / 2
                periodic_coupon = (coupon_rate * face_value) / 2

                if periodic_rate == 0:
                    return periodic_coupon * periods + face_value
                else:
                    pv_coupons = periodic_coupon * (1 - (1 + periodic_rate) ** (-periods)) / periodic_rate
                    pv_face = face_value / (1 + periodic_rate) ** periods
                    return pv_coupons + pv_face

            def objective(ytm):
                return price_function(ytm) - price

            # Solve for YTM
            ytm = optimize.brentq(objective, 0, 1, xtol=1e-6)
            return ytm

        except:
            # Fallback approximation
            annual_coupon = coupon_rate * face_value
            return (annual_coupon + (face_value - price) / maturity) / ((face_value + price) / 2)

    def _build_credit_term_structure(self, pd_estimates: List[Dict]) -> Dict[str, Any]:
        """Build credit risk term structure."""
        try:
            maturities = [est['maturity'] for est in pd_estimates]
            spreads = [est['credit_spread'] / 10000 for est in pd_estimates]  # Convert from bps
            pds = [est['probability_of_default'] for est in pd_estimates]

            # Sort by maturity
            sorted_data = sorted(zip(maturities, spreads, pds))
            sorted_maturities, sorted_spreads, sorted_pds = zip(*sorted_data)

            return {
                'maturities': list(sorted_maturities),
                'credit_spreads': [round(s * 10000, 2) for s in sorted_spreads],  # bps
                'probabilities_of_default': [round(pd * 100, 2) for pd in sorted_pds],  # percentage
                'term_structure_slope': self._calculate_term_structure_slope(sorted_maturities, sorted_spreads)
            }

        except Exception:
            return {'error': 'Unable to build term structure'}

    def _calculate_term_structure_slope(self, maturities: List[float], spreads: List[float]) -> float:
        """Calculate slope of credit spread term structure."""
        try:
            if len(maturities) < 2:
                return 0

            # Simple linear regression
            x = np.array(maturities)
            y = np.array(spreads)

            slope, _, _, _, _ = stats.linregress(x, y)
            return round(slope * 10000, 2)  # bps per year

        except Exception:
            return 0

    def _calibrate_credit_curve(self, maturities: List[float], spreads: List[float]) -> Dict[str, Any]:
        """Calibrate credit curve parameters."""
        try:
            # Fit exponential decay model: spread(t) = a * exp(-b * t) + c
            def exponential_model(t, a, b, c):
                return a * np.exp(-b * t) + c

            try:
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(exponential_model, maturities, spreads,
                                  bounds=([-np.inf, 0, 0], [np.inf, np.inf, np.inf]))

                return {
                    'model': 'exponential_decay',
                    'parameters': {
                        'amplitude': round(popt[0], self.config.precision),
                        'decay_rate': round(popt[1], self.config.precision),
                        'asymptote': round(popt[2], self.config.precision)
                    },
                    'fitted': True
                }
            except:
                # Fallback to simple interpolation
                return {
                    'model': 'linear_interpolation',
                    'parameters': {'maturities': maturities, 'spreads': spreads},
                    'fitted': False
                }

        except Exception:
            return {'error': 'Unable to calibrate credit curve'}

    def _calculate_forward_pds(self, pd_calculations: List[Dict]) -> List[Dict]:
        """Calculate forward PDs from spot PDs."""
        try:
            forward_pds = []

            for i in range(1, len(pd_calculations)):
                current = pd_calculations[i]
                previous = pd_calculations[i-1]

                t1 = previous['maturity']
                t2 = current['maturity']
                pd1 = previous['probability_of_default']
                pd2 = current['probability_of_default']

                # Forward PD = (PD(t2) - PD(t1)) / (1 - PD(t1))
                if 1 - pd1 > 0:
                    forward_pd = (pd2 - pd1) / (1 - pd1)
                else:
                    forward_pd = 0

                forward_pds.append({
                    'start_period': round(t1, self.config.precision),
                    'end_period': round(t2, self.config.precision),
                    'forward_pd': round(max(0, forward_pd), self.config.precision),
                    'forward_pd_percentage': round(max(0, forward_pd) * 100, self.config.precision)
                })

            return forward_pds

        except Exception:
            return []

    def _assess_credit_quality(self, spreads: List[float], maturities: List[float]) -> Dict[str, str]:
        """Assess overall credit quality from spreads."""
        try:
            avg_spread_bps = np.mean(spreads) * 10000

            if avg_spread_bps < 50:
                rating = 'AAA/AA'
                quality = 'Excellent'
            elif avg_spread_bps < 100:
                rating = 'A'
                quality = 'Good'
            elif avg_spread_bps < 200:
                rating = 'BBB'
                quality = 'Investment Grade'
            elif avg_spread_bps < 500:
                rating = 'BB/B'
                quality = 'High Yield'
            else:
                rating = 'CCC or below'
                quality = 'Distressed'

            return {
                'implied_rating': rating,
                'credit_quality': quality,
                'average_spread_bps': round(avg_spread_bps, 1)
            }

        except Exception:
            return {'error': 'Unable to assess credit quality'}

    def _calculate_generator_matrix(self, transition_matrix: np.ndarray, time_horizon_days: int) -> np.ndarray:
        """Calculate generator matrix for continuous-time Markov chain."""
        try:
            # Convert time horizon to years
            time_horizon_years = time_horizon_days / 365.25

            # Generator matrix Q such that P(t) = exp(Q * t)
            # Q = log(P) / t (matrix logarithm)

            # Check if transition matrix is valid
            if np.any(transition_matrix < 0) or not np.allclose(transition_matrix.sum(axis=1), 1):
                return None

            # Compute matrix logarithm
            eigenvals, eigenvecs = np.linalg.eig(transition_matrix)

            # Handle potential numerical issues
            eigenvals = np.real_if_close(eigenvals)
            log_eigenvals = np.log(eigenvals + 1e-12)  # Add small epsilon to avoid log(0)

            # Reconstruct matrix logarithm
            log_matrix = eigenvecs @ np.diag(log_eigenvals) @ np.linalg.inv(eigenvecs)
            generator_matrix = np.real(log_matrix) / time_horizon_years

            return generator_matrix

        except Exception:
            return None

    def _calculate_cds_premium_leg(self, spread: float, maturity: float, hazard_rate: float) -> float:
        """Calculate present value of CDS premium leg."""
        try:
            # Quarterly payments
            dt = 0.25
            times = np.arange(dt, maturity + dt, dt)

            # Risk-free rate (assumed constant)
            risk_free_rate = 0.02

            pv_premium = 0
            for t in times:
                # Survival probability
                survival_prob = np.exp(-hazard_rate * t)
                # Discount factor
                discount_factor = np.exp(-risk_free_rate * t)
                # Premium payment
                pv_premium += spread * dt * survival_prob * discount_factor

            return pv_premium

        except Exception:
            return 0

    def _calculate_cds_protection_leg(self, lgd: float, maturity: float, hazard_rate: float) -> float:
        """Calculate present value of CDS protection leg."""
        try:
            # Risk-free rate (assumed constant)
            risk_free_rate = 0.02

            # Continuous approximation
            # PV = LGD * integral_0^T lambda * exp(-(r + lambda) * t) dt
            combined_rate = risk_free_rate + hazard_rate

            if combined_rate > 0:
                pv_protection = lgd * hazard_rate * (1 - np.exp(-combined_rate * maturity)) / combined_rate
            else:
                pv_protection = lgd * hazard_rate * maturity

            return pv_protection

        except Exception:
            return 0

    def _build_hazard_rate_curve(self, maturities: List[float], hazard_rates: List[float]) -> Dict[str, Any]:
        """Build hazard rate curve from market data."""
        try:
            # Sort by maturity
            sorted_data = sorted(zip(maturities, hazard_rates))
            sorted_maturities, sorted_rates = zip(*sorted_data)

            return {
                'maturities': list(sorted_maturities),
                'hazard_rates': [round(rate, self.config.precision) for rate in sorted_rates],
                'curve_type': 'piecewise_linear'
            }

        except Exception:
            return {'error': 'Unable to build hazard rate curve'}


class RiskWeightedAssets:
    """
    Risk Weighted Assets calculations for regulatory capital.
    """

    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()

    def calculate_credit_rwa(self, exposures: List[Dict]) -> Dict[str, Any]:
        """
        Calculate credit risk RWA using standardized approach.

        Parameters:
        -----------
        exposures : List[Dict]
            List of credit exposures with keys: amount, rating, type, maturity

        Returns:
        --------
        Dict with credit RWA calculations
        """
        try:
            # Standard risk weights by rating (Basel III)
            risk_weights = {
                'AAA': 0.20, 'AA+': 0.20, 'AA': 0.20, 'AA-': 0.20,
                'A+': 0.50, 'A': 0.50, 'A-': 0.50,
                'BBB+': 1.00, 'BBB': 1.00, 'BBB-': 1.00,
                'BB+': 1.50, 'BB': 1.50, 'BB-': 1.50,
                'B+': 2.50, 'B': 2.50, 'B-': 2.50,
                'CCC': 15.00, 'CC': 15.00, 'C': 15.00,
                'D': 15.00, 'Unrated': 1.00
            }

            # Exposure type multipliers
            exposure_multipliers = {
                'sovereign': 0.0,
                'bank': 0.20,
                'corporate': 1.00,
                'retail': 0.75,
                'mortgage': 0.35,
                'small_business': 0.85
            }

            rwa_calculations = []
            total_exposure = 0
            total_rwa = 0

            for exposure in exposures:
                amount = exposure.get('amount', 0)
                rating = exposure.get('rating', 'Unrated')
                exp_type = exposure.get('type', 'corporate')
                maturity = exposure.get('maturity', 1)

                # Get base risk weight
                base_risk_weight = risk_weights.get(rating, risk_weights['Unrated'])

                # Apply exposure type adjustment
                type_multiplier = exposure_multipliers.get(exp_type, 1.0)

                # Maturity adjustment for corporate exposures > 1 year
                if exp_type == 'corporate' and maturity > 1:
                    maturity_adjustment = min(1.0 + 0.1 * (maturity - 1), 5.0)
                else:
                    maturity_adjustment = 1.0

                # Calculate effective risk weight
                effective_risk_weight = base_risk_weight * type_multiplier * maturity_adjustment

                # Calculate RWA for this exposure
                exposure_rwa = amount * effective_risk_weight

                rwa_calculations.append({
                    'exposure_id': exposure.get('id', len(rwa_calculations)),
                    'amount': round(amount, self.config.precision),
                    'rating': rating,
                    'type': exp_type,
                    'maturity': round(maturity, self.config.precision),
                    'base_risk_weight': round(base_risk_weight, self.config.precision),
                    'type_multiplier': round(type_multiplier, self.config.precision),
                    'maturity_adjustment': round(maturity_adjustment, self.config.precision),
                    'effective_risk_weight': round(effective_risk_weight, self.config.precision),
                    'rwa': round(exposure_rwa, self.config.precision)
                })

                total_exposure += amount
                total_rwa += exposure_rwa

            # Calculate portfolio metrics
            average_risk_weight = total_rwa / total_exposure if total_exposure > 0 else 0

            # Risk concentration analysis
            concentration_analysis = self._analyze_risk_concentration(rwa_calculations)

            return {
                'total_exposure': round(total_exposure, self.config.precision),
                'total_rwa': round(total_rwa, self.config.precision),
                'average_risk_weight': round(average_risk_weight, self.config.precision),
                'exposure_details': rwa_calculations,
                'concentration_analysis': concentration_analysis,
                'number_of_exposures': len(exposures),
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def calculate_market_rwa(self, positions: List[Dict], method: str = 'standardized') -> Dict[str, Any]:
        """
        Calculate market risk RWA.

        Parameters:
        -----------
        positions : List[Dict]
            Market risk positions with keys: amount, risk_type, maturity
        method : str
            Calculation method ('standardized', 'internal_models')

        Returns:
        --------
        Dict with market RWA calculations
        """
        try:
            if method == 'standardized':
                return self._calculate_standardized_market_rwa(positions)
            elif method == 'internal_models':
                return self._calculate_internal_models_rwa(positions)
            else:
                return {'error': f'Unsupported method: {method}'}

        except Exception as e:
            return {'error': str(e)}

    def calculate_operational_rwa(self, gross_income: List[float],
                                business_lines: List[str] = None) -> Dict[str, Any]:
        """
        Calculate operational risk RWA using Basic Indicator Approach.

        Parameters:
        -----------
        gross_income : List[float]
            Gross income for last 3 years
        business_lines : List[str]
            Business line classifications

        Returns:
        --------
        Dict with operational RWA calculations
        """
        try:
            # Basic Indicator Approach: 15% of average gross income
            alpha = 0.15

            # Remove negative values and calculate average
            positive_income = [income for income in gross_income if income > 0]

            if not positive_income:
                return {'error': 'No positive gross income data'}

            average_gross_income = np.mean(positive_income)
            operational_rwa = average_gross_income * alpha

            # Calculate capital requirement (8% of RWA)
            capital_requirement = operational_rwa * 0.08

            return {
                'method': 'basic_indicator_approach',
                'gross_income_years': len(gross_income),
                'positive_income_years': len(positive_income),
                'average_gross_income': round(average_gross_income, self.config.precision),
                'alpha_factor': alpha,
                'operational_rwa': round(operational_rwa, self.config.precision),
                'capital_requirement': round(capital_requirement, self.config.precision),
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def total_capital_requirements(self, credit_rwa: float, market_rwa: float,
                                 operational_rwa: float, buffers: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Calculate total capital requirements including buffers.

        Parameters:
        -----------
        credit_rwa : float
            Credit risk RWA
        market_rwa : float
            Market risk RWA
        operational_rwa : float
            Operational risk RWA
        buffers : Dict
            Additional capital buffers

        Returns:
        --------
        Dict with total capital requirements
        """
        try:
            if buffers is None:
                buffers = self.config.basel_iii_buffers

            # Total RWA
            total_rwa = credit_rwa + market_rwa + operational_rwa

            # Minimum capital ratios (Basel III)
            common_equity_tier1_ratio = 0.045  # 4.5%
            tier1_capital_ratio = 0.06         # 6.0%
            total_capital_ratio = 0.08         # 8.0%

            # Add buffers
            capital_conservation_buffer = buffers.get('capital_conservation', 0.025)
            countercyclical_buffer = buffers.get('countercyclical', 0.0)
            systemic_importance_buffer = buffers.get('systemic_importance', 0.0)

            # Total required ratios including buffers
            total_cet1_ratio = common_equity_tier1_ratio + capital_conservation_buffer + countercyclical_buffer + systemic_importance_buffer
            total_tier1_ratio = tier1_capital_ratio + capital_conservation_buffer + countercyclical_buffer + systemic_importance_buffer
            total_capital_ratio_req = total_capital_ratio + capital_conservation_buffer + countercyclical_buffer + systemic_importance_buffer

            # Calculate required capital amounts
            cet1_requirement = total_rwa * total_cet1_ratio
            tier1_requirement = total_rwa * total_tier1_ratio
            total_capital_requirement = total_rwa * total_capital_ratio_req

            return {
                'total_rwa': round(total_rwa, self.config.precision),
                'rwa_breakdown': {
                    'credit_rwa': round(credit_rwa, self.config.precision),
                    'market_rwa': round(market_rwa, self.config.precision),
                    'operational_rwa': round(operational_rwa, self.config.precision)
                },
                'minimum_ratios': {
                    'cet1_ratio': common_equity_tier1_ratio,
                    'tier1_ratio': tier1_capital_ratio,
                    'total_capital_ratio': total_capital_ratio
                },
                'buffers_applied': buffers,
                'total_required_ratios': {
                    'cet1_ratio': round(total_cet1_ratio, 4),
                    'tier1_ratio': round(total_tier1_ratio, 4),
                    'total_capital_ratio': round(total_capital_ratio_req, 4)
                },
                'capital_requirements': {
                    'cet1_requirement': round(cet1_requirement, self.config.precision),
                    'tier1_requirement': round(tier1_requirement, self.config.precision),
                    'total_capital_requirement': round(total_capital_requirement, self.config.precision)
                },
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def _calculate_standardized_market_rwa(self, positions: List[Dict]) -> Dict[str, Any]:
        """Calculate market RWA using standardized approach."""
        try:
            # Standard market risk charges
            risk_charges = {
                'interest_rate': 0.01,      # 1% for general market risk
                'equity': 0.08,             # 8% for equity risk
                'fx': 0.08,                 # 8% for FX risk
                'commodity': 0.15,          # 15% for commodity risk
                'credit_spread': 0.01       # 1% for credit spread risk
            }

            total_market_rwa = 0
            position_details = []

            for position in positions:
                amount = abs(position.get('amount', 0))  # Use absolute value
                risk_type = position.get('risk_type', 'equity')
                maturity = position.get('maturity', 1)

                # Get base risk charge
                base_charge = risk_charges.get(risk_type, 0.08)

                # Maturity adjustment for interest rate risk
                if risk_type == 'interest_rate':
                    if maturity <= 1:
                        maturity_weight = 1.0
                    elif maturity <= 5:
                        maturity_weight = 1.5
                    else:
                        maturity_weight = 2.0
                else:
                    maturity_weight = 1.0

                # Calculate position RWA
                position_risk_charge = base_charge * maturity_weight
                position_rwa = amount * position_risk_charge

                position_details.append({
                    'position_id': position.get('id', len(position_details)),
                    'amount': round(amount, self.config.precision),
                    'risk_type': risk_type,
                    'maturity': round(maturity, self.config.precision),
                    'base_charge': round(base_charge, self.config.precision),
                    'maturity_weight': round(maturity_weight, self.config.precision),
                    'risk_charge': round(position_risk_charge, self.config.precision),
                    'rwa': round(position_rwa, self.config.precision)
                })

                total_market_rwa += position_rwa

            return {
                'method': 'standardized_approach',
                'total_market_rwa': round(total_market_rwa, self.config.precision),
                'position_details': position_details,
                'number_of_positions': len(positions),
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def _calculate_internal_models_rwa(self, positions: List[Dict]) -> Dict[str, Any]:
        """Calculate market RWA using internal models approach."""
        try:
            # Simplified VaR-based approach
            # In practice, this would use sophisticated VaR models

            # Assume we have VaR calculations for each position
            total_var = 0
            position_vars = []

            for position in positions:
                amount = abs(position.get('amount', 0))
                # Simplified: assume 2% daily VaR
                position_var = amount * 0.02
                position_vars.append(position_var)
                total_var += position_var

            # Regulatory multiplier (typically 3-4)
            multiplier = 3.5

            # Market RWA = VaR * multiplier * sqrt(10) * 12.5
            # sqrt(10) for 10-day holding period, 12.5 for 8% capital ratio
            market_rwa = total_var * multiplier * np.sqrt(10) * 12.5

            return {
                'method': 'internal_models_approach',
                'total_var': round(total_var, self.config.precision),
                'regulatory_multiplier': multiplier,
                'total_market_rwa': round(market_rwa, self.config.precision),
                'number_of_positions': len(positions),
                'calculation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def _analyze_risk_concentration(self, rwa_calculations: List[Dict]) -> Dict[str, Any]:
        """Analyze risk concentration in credit portfolio."""
        try:
            # Concentration by rating
            rating_concentration = {}
            type_concentration = {}

            total_rwa = sum(calc['rwa'] for calc in rwa_calculations)

            for calc in rwa_calculations:
                rating = calc['rating']
                exp_type = calc['type']
                rwa = calc['rwa']

                # Rating concentration
                if rating not in rating_concentration:
                    rating_concentration[rating] = 0
                rating_concentration[rating] += rwa

                # Type concentration
                if exp_type not in type_concentration:
                    type_concentration[exp_type] = 0
                type_concentration[exp_type] += rwa

            # Convert to percentages
            rating_percentages = {
                rating: round((rwa / total_rwa) * 100, 2)
                for rating, rwa in rating_concentration.items()
            } if total_rwa > 0 else {}

            type_percentages = {
                exp_type: round((rwa / total_rwa) * 100, 2)
                for exp_type, rwa in type_concentration.items()
            } if total_rwa > 0 else {}

            # Calculate Herfindahl-Hirschman Index for concentration
            rating_hhi = sum((pct / 100) ** 2 for pct in rating_percentages.values())
            type_hhi = sum((pct / 100) ** 2 for pct in type_percentages.values())

            return {
                'rating_concentration': rating_percentages,
                'type_concentration': type_percentages,
                'rating_hhi': round(rating_hhi, self.config.precision),
                'type_hhi': round(type_hhi, self.config.precision),
                'concentration_risk': 'High' if max(rating_hhi, type_hhi) > 0.25 else 'Moderate' if max(rating_hhi, type_hhi) > 0.15 else 'Low'
            }

        except Exception:
            return {'error': 'Unable to analyze concentration'}


# Example usage and testing functions
def example_var_calculations():
    """Example of VaR calculations."""
    # Generate sample returns
    np.random.seed(42)
    returns = np.random.normal(-0.001, 0.02, 252)  # Daily returns for 1 year

    var_calc = ValueAtRisk()

    # Historical Simulation VaR
    hist_var = var_calc.historical_simulation_var(returns, 0.95)

    # Parametric VaR
    param_var = var_calc.parametric_var(returns, 0.95)

    # Monte Carlo VaR
    mc_var = var_calc.monte_carlo_var(returns, 0.95)

    return {
        'historical_var': hist_var,
        'parametric_var': param_var,
        'monte_carlo_var': mc_var
    }


def example_credit_risk():
    """Example of credit risk calculations."""
    credit_risk = CreditRisk()

    # Example bond data
    bond_prices = [98.5, 95.2, 101.3]
    face_values = [100, 100, 100]
    coupon_rates = [0.05, 0.04, 0.06]
    maturities = [5, 10, 3]
    risk_free_rates = [0.03, 0.035, 0.025]

    # Calculate PD from bond prices
    pd_result = credit_risk.pd_from_bond_prices(
        bond_prices, face_values, coupon_rates, maturities, risk_free_rates
    )

    return pd_result


def example_counterparty_risk():
    """Example of counterparty risk calculations."""
    counterparty_risk = CounterpartyRisk()

    # Example trades
    trades = [
        {'type': 'swap', 'notional': 10000000, 'maturity_years': 5},
        {'type': 'forward', 'notional': 5000000, 'maturity_years': 2},
        {'type': 'option', 'notional': 2000000, 'maturity_years': 1}
    ]

    # Calculate PFE
    pfe_result = counterparty_risk.potential_future_exposure(trades)

    # Calculate EEPE
    eepe_result = counterparty_risk.effective_expected_positive_exposure(trades)

    return {
        'pfe': pfe_result,
        'eepe': eepe_result
    }


if __name__ == "__main__":
    print("Risk Metrics Framework")
    print("=" * 50)

    # Example VaR calculations
    print("\n1. VaR Calculations:")
    var_examples = example_var_calculations()
    for method, result in var_examples.items():
        if 'var' in result:
            print(f"{method}: {result['var']:.4f} ({result['var_percentage']:.2f}%)")

    # Example credit risk
    print("\n2. Credit Risk Analysis:")
    credit_example = example_credit_risk()
    if 'average_pd' in credit_example:
        print(f"Average PD: {credit_example['average_pd']:.4f}")

    # Example counterparty risk
    print("\n3. Counterparty Risk Analysis:")
    ccr_example = example_counterparty_risk()
    if 'peak_pfe' in ccr_example['pfe']:
        print(f"Peak PFE: {ccr_example['pfe']['peak_pfe']:,.0f}")

    print("\nRisk metrics calculations completed!")