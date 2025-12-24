"""
Probability Distributions Module for Financial and Economic Data

This module provides comprehensive functionality for:
- Testing normality of data vectors using multiple statistical tests
- Modeling data using various probability distributions
- Parameter estimation and distribution fitting
- Visualization and diagnostic tools

Dependencies: scipy, numpy, pandas, matplotlib, seaborn, statsmodels
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Optional, Dict, Any, Tuple, Union, List
from scipy import stats
from scipy.optimize import minimize
import math

# Optional imports for enhanced functionality
try:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import jarque_bera
    from statsmodels.stats.stattools import durbin_watson
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("Statsmodels not available. Some advanced statistical tests will be disabled.")

try:
    from scipy.stats import skewnorm
    SKEWNORM_AVAILABLE = True
except ImportError:
    SKEWNORM_AVAILABLE = False
    warnings.warn("Skewed normal distribution not available in this scipy version.")


class NormalityTester:
    """
    Class for testing normality of data vectors using multiple statistical tests
    """

    def __init__(self):
        self.test_results = {}
        self.available_tests = [
            'shapiro_wilk',
            'kolmogorov_smirnov',
            'dagostino_pearson',
            'jarque_bera',
            'qq_plot'
        ]

    def shapiro_wilk_test(self, data: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Shapiro-Wilk test for normality

        Args:
            data: Data vector to test
            alpha: Significance level (default: 0.05)

        Returns:
            Dictionary with test results
        """
        # Shapiro-Wilk test works best with sample sizes between 3 and 5000
        if len(data) > 5000:
            warnings.warn("Shapiro-Wilk test may not be reliable for samples > 5000")
            data = np.random.choice(data, 5000, replace=False)

        statistic, p_value = stats.shapiro(data)

        result = {
            'test_name': 'Shapiro-Wilk W Test',
            'statistic': statistic,
            'p_value': p_value,
            'alpha': alpha,
            'is_normal': p_value > alpha,
            'interpretation': 'Normal' if p_value > alpha else 'Not Normal',
            'sample_size': len(data)
        }

        return result

    def kolmogorov_smirnov_test(self, data: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Kolmogorov-Smirnov test for normality

        Args:
            data: Data vector to test
            alpha: Significance level (default: 0.05)

        Returns:
            Dictionary with test results
        """
        # Standardize the data
        data_standardized = (data - np.mean(data)) / np.std(data)

        # Test against standard normal distribution
        statistic, p_value = stats.kstest(data_standardized, 'norm')

        result = {
            'test_name': 'Kolmogorov-Smirnov Test',
            'statistic': statistic,
            'p_value': p_value,
            'alpha': alpha,
            'is_normal': p_value > alpha,
            'interpretation': 'Normal' if p_value > alpha else 'Not Normal',
            'sample_size': len(data)
        }

        return result

    def dagostino_pearson_test(self, data: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
        """
        D'Agostino and Pearson test for normality

        Args:
            data: Data vector to test
            alpha: Significance level (default: 0.05)

        Returns:
            Dictionary with test results
        """
        statistic, p_value = stats.normaltest(data)

        result = {
            'test_name': "D'Agostino and Pearson Test",
            'statistic': statistic,
            'p_value': p_value,
            'alpha': alpha,
            'is_normal': p_value > alpha,
            'interpretation': 'Normal' if p_value > alpha else 'Not Normal',
            'sample_size': len(data),
            'note': 'Tests both skewness and kurtosis'
        }

        return result

    def jarque_bera_test(self, data: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Jarque-Bera test for normality

        Args:
            data: Data vector to test
            alpha: Significance level (default: 0.05)

        Returns:
            Dictionary with test results
        """
        if STATSMODELS_AVAILABLE:
            statistic, p_value, skewness, kurtosis = jarque_bera(data)
        else:
            # Fallback implementation using scipy
            n = len(data)
            skewness = stats.skew(data)
            kurt = stats.kurtosis(data)
            statistic = n * (skewness**2 / 6 + (kurt)**2 / 24)
            p_value = 1 - stats.chi2.cdf(statistic, df=2)

        result = {
            'test_name': 'Jarque-Bera Test',
            'statistic': statistic,
            'p_value': p_value,
            'alpha': alpha,
            'is_normal': p_value > alpha,
            'interpretation': 'Normal' if p_value > alpha else 'Not Normal',
            'sample_size': len(data),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data)
        }

        return result

    def qq_plot_analysis(self, data: np.ndarray,
                        figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, Dict[str, Any]]:
        """
        Q-Q plot analysis for normality assessment

        Args:
            data: Data vector to test
            figsize: Figure size for plots

        Returns:
            Tuple of (matplotlib figure, analysis results)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Q-Q plot
        stats.probplot(data, dist="norm", plot=ax1)
        ax1.set_title('Q-Q Plot against Normal Distribution')
        ax1.grid(True, alpha=0.3)

        # Calculate correlation coefficient for Q-Q plot
        theoretical_quantiles, ordered_values = stats.probplot(data, dist="norm")[:2]
        qq_correlation = np.corrcoef(theoretical_quantiles, ordered_values)[0, 1]

        # Histogram with normal overlay
        ax2.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')

        # Overlay fitted normal distribution
        mu, sigma = stats.norm.fit(data)
        x = np.linspace(data.min(), data.max(), 100)
        ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                label=f'Normal (μ={mu:.2f}, σ={sigma:.2f})')

        ax2.set_title('Histogram with Fitted Normal Distribution')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Analysis results
        analysis = {
            'test_name': 'Q-Q Plot Analysis',
            'qq_correlation': qq_correlation,
            'fitted_parameters': {'mean': mu, 'std': sigma},
            'interpretation': 'Good fit to normal' if qq_correlation > 0.95 else
                           'Moderate fit to normal' if qq_correlation > 0.90 else 'Poor fit to normal',
            'sample_size': len(data),
            'note': 'Higher Q-Q correlation indicates better fit to normal distribution'
        }

        return fig, analysis

    def comprehensive_normality_test(self, df: pd.DataFrame, column: str,
                                   alpha: float = 0.05,
                                   create_plots: bool = True) -> Dict[str, Any]:
        """
        Run all normality tests on a data vector

        Args:
            df: DataFrame containing the data
            column: Column name to test
            alpha: Significance level
            create_plots: Whether to create diagnostic plots

        Returns:
            Dictionary with all test results
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        data = df[column].dropna().values

        if len(data) < 3:
            raise ValueError("Insufficient data points for normality testing (need at least 3)")

        # Run all tests
        results = {
            'column': column,
            'sample_size': len(data),
            'alpha': alpha,
            'data_summary': {
                'mean': np.mean(data),
                'std': np.std(data),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data),
                'min': np.min(data),
                'max': np.max(data)
            }
        }

        # Individual tests
        results['shapiro_wilk'] = self.shapiro_wilk_test(data, alpha)
        results['kolmogorov_smirnov'] = self.kolmogorov_smirnov_test(data, alpha)
        results['dagostino_pearson'] = self.dagostino_pearson_test(data, alpha)
        results['jarque_bera'] = self.jarque_bera_test(data, alpha)

        # Q-Q plot analysis
        if create_plots:
            fig, qq_analysis = self.qq_plot_analysis(data)
            results['qq_plot'] = qq_analysis
            results['qq_plot_figure'] = fig
        else:
            # Just the correlation without plot
            theoretical_quantiles, ordered_values = stats.probplot(data, dist="norm")[:2]
            qq_correlation = np.corrcoef(theoretical_quantiles, ordered_values)[0, 1]
            results['qq_plot'] = {
                'test_name': 'Q-Q Plot Analysis',
                'qq_correlation': qq_correlation,
                'interpretation': 'Good fit to normal' if qq_correlation > 0.95 else
                               'Moderate fit to normal' if qq_correlation > 0.90 else 'Poor fit to normal'
            }

        # Overall assessment
        normal_tests = [
            results['shapiro_wilk']['is_normal'],
            results['kolmogorov_smirnov']['is_normal'],
            results['dagostino_pearson']['is_normal'],
            results['jarque_bera']['is_normal']
        ]

        results['overall_assessment'] = {
            'tests_indicating_normal': sum(normal_tests),
            'total_tests': len(normal_tests),
            'proportion_normal': sum(normal_tests) / len(normal_tests),
            'consensus': 'Strong evidence for normality' if sum(normal_tests) >= 3 else
                        'Moderate evidence for normality' if sum(normal_tests) == 2 else
                        'Weak evidence for normality' if sum(normal_tests) == 1 else
                        'Strong evidence against normality'
        }

        # Store results
        self.test_results[column] = results

        return results

    def compare_normality_across_columns(self, df: pd.DataFrame,
                                       columns: Optional[List[str]] = None,
                                       alpha: float = 0.05) -> pd.DataFrame:
        """
        Compare normality across multiple columns

        Args:
            df: DataFrame containing the data
            columns: List of columns to test (if None, uses all numeric columns)
            alpha: Significance level

        Returns:
            DataFrame with normality test comparison
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        comparison_results = []

        for col in columns:
            try:
                results = self.comprehensive_normality_test(df, col, alpha, create_plots=False)

                comparison_results.append({
                    'Column': col,
                    'Sample_Size': results['sample_size'],
                    'Shapiro_Wilk_p': results['shapiro_wilk']['p_value'],
                    'KS_p': results['kolmogorov_smirnov']['p_value'],
                    'DAgostino_Pearson_p': results['dagostino_pearson']['p_value'],
                    'Jarque_Bera_p': results['jarque_bera']['p_value'],
                    'QQ_Correlation': results['qq_plot']['qq_correlation'],
                    'Tests_Normal': results['overall_assessment']['tests_indicating_normal'],
                    'Consensus': results['overall_assessment']['consensus'],
                    'Skewness': results['data_summary']['skewness'],
                    'Kurtosis': results['data_summary']['kurtosis']
                })
            except Exception as e:
                comparison_results.append({
                    'Column': col,
                    'Error': str(e)
                })

        return pd.DataFrame(comparison_results)


class DistributionModeler:
    """
    Class for modeling data vectors using various probability distributions
    """

    def __init__(self):
        self.fitted_distributions = {}
        self.available_distributions = [
            'bernoulli',
            'binomial',
            'normal',
            'skew_normal',
            'student_t',
            'poisson',
            'beta',
            'lognormal'
        ]

    def fit_bernoulli(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Fit Bernoulli distribution to binary data

        Args:
            data: Binary data (0s and 1s)

        Returns:
            Dictionary with distribution parameters and fit statistics
        """
        # Validate binary data
        unique_values = np.unique(data)
        if not np.array_equal(unique_values, [0, 1]) and not np.array_equal(unique_values, [0]) and not np.array_equal(unique_values, [1]):
            warnings.warn("Data contains values other than 0 and 1. Converting to binary.")
            data = (data > np.mean(data)).astype(int)

        # Parameter estimation
        p = np.mean(data)

        # Create distribution object
        dist = stats.bernoulli(p)

        # Calculate goodness of fit
        observed_freq = np.bincount(data.astype(int), minlength=2)
        expected_freq = np.array([len(data) * (1-p), len(data) * p])
        chi2_stat = np.sum((observed_freq - expected_freq)**2 / expected_freq)
        chi2_p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)

        result = {
            'distribution': 'Bernoulli',
            'parameters': {'p': p},
            'distribution_object': dist,
            'log_likelihood': np.sum(dist.logpmf(data)),
            'aic': 2 * 1 - 2 * np.sum(dist.logpmf(data)),  # 1 parameter
            'bic': np.log(len(data)) * 1 - 2 * np.sum(dist.logpmf(data)),
            'chi2_statistic': chi2_stat,
            'chi2_p_value': chi2_p_value,
            'sample_size': len(data)
        }

        return result

    def fit_binomial(self, data: np.ndarray, n: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Fit Binomial distribution to count data

        Args:
            data: Count data
            n: Number of trials (if None, estimated from data)

        Returns:
            Dictionary with distribution parameters and fit statistics
        """
        # Estimate parameters
        if n is None:
            n = int(np.max(data))

        p = np.mean(data) / n

        # Create distribution object
        dist = stats.binom(n, p)

        # Calculate goodness of fit
        unique_values = np.unique(data)
        observed_freq = np.array([np.sum(data == val) for val in unique_values])
        expected_freq = np.array([len(data) * dist.pmf(val) for val in unique_values])

        # Chi-square test (only for values with expected frequency > 5)
        valid_idx = expected_freq >= 5
        if np.sum(valid_idx) > 1:
            chi2_stat = np.sum((observed_freq[valid_idx] - expected_freq[valid_idx])**2 / expected_freq[valid_idx])
            chi2_p_value = 1 - stats.chi2.cdf(chi2_stat, df=np.sum(valid_idx) - 1)
        else:
            chi2_stat = np.nan
            chi2_p_value = np.nan

        result = {
            'distribution': 'Binomial',
            'parameters': {'n': n, 'p': p},
            'distribution_object': dist,
            'log_likelihood': np.sum(dist.logpmf(data)),
            'aic': 2 * 2 - 2 * np.sum(dist.logpmf(data)),  # 2 parameters
            'bic': np.log(len(data)) * 2 - 2 * np.sum(dist.logpmf(data)),
            'chi2_statistic': chi2_stat,
            'chi2_p_value': chi2_p_value,
            'sample_size': len(data)
        }

        return result

    def fit_normal(self, data: np.ndarray, method: str = 'mle', **kwargs) -> Dict[str, Any]:
        """
        Fit Normal (Gaussian) distribution to data

        Args:
            data: Continuous data
            method: Parameter estimation method ('mle', 'moments')

        Returns:
            Dictionary with distribution parameters and fit statistics
        """
        if method == 'mle':
            # Maximum likelihood estimation
            mu, sigma = stats.norm.fit(data)
        elif method == 'moments':
            # Method of moments
            mu = np.mean(data)
            sigma = np.std(data, ddof=1)
        else:
            raise ValueError("Method must be 'mle' or 'moments'")

        # Create distribution object
        dist = stats.norm(mu, sigma)

        # Kolmogorov-Smirnov test
        ks_stat, ks_p_value = stats.kstest(data, dist.cdf)

        result = {
            'distribution': 'Normal',
            'parameters': {'mean': mu, 'std': sigma},
            'distribution_object': dist,
            'log_likelihood': np.sum(dist.logpdf(data)),
            'aic': 2 * 2 - 2 * np.sum(dist.logpdf(data)),  # 2 parameters
            'bic': np.log(len(data)) * 2 - 2 * np.sum(dist.logpdf(data)),
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p_value,
            'sample_size': len(data),
            'method': method
        }

        return result

    def fit_skew_normal(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Fit Skew-Normal distribution to data

        Args:
            data: Continuous data

        Returns:
            Dictionary with distribution parameters and fit statistics
        """
        if not SKEWNORM_AVAILABLE:
            raise ImportError("Skew-normal distribution not available in this scipy version")

        # Fit parameters
        a, loc, scale = stats.skewnorm.fit(data)

        # Create distribution object
        dist = stats.skewnorm(a, loc, scale)

        # Kolmogorov-Smirnov test
        ks_stat, ks_p_value = stats.kstest(data, dist.cdf)

        result = {
            'distribution': 'Skew-Normal',
            'parameters': {'skewness': a, 'location': loc, 'scale': scale},
            'distribution_object': dist,
            'log_likelihood': np.sum(dist.logpdf(data)),
            'aic': 2 * 3 - 2 * np.sum(dist.logpdf(data)),  # 3 parameters
            'bic': np.log(len(data)) * 3 - 2 * np.sum(dist.logpdf(data)),
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p_value,
            'sample_size': len(data)
        }

        return result

    def fit_student_t(self, data: np.ndarray, parameterization: str = 'centered', **kwargs) -> Dict[str, Any]:
        """
        Fit Student's t-distribution to data

        Args:
            data: Continuous data
            parameterization: 'centered' (location, scale, shape, df) or 'direct' (mean, var, skew, kurtosis)

        Returns:
            Dictionary with distribution parameters and fit statistics
        """
        # Fit parameters using maximum likelihood
        df, loc, scale = stats.t.fit(data)

        # Create distribution object
        dist = stats.t(df, loc, scale)

        # Calculate moments for direct parameterization
        if parameterization == 'direct' and df > 4:
            mean = loc
            variance = scale**2 * df / (df - 2) if df > 2 else np.inf
            skewness = 0 if df > 3 else np.nan  # t-distribution is symmetric
            kurtosis = 6 / (df - 4) if df > 4 else np.inf

            direct_params = {
                'mean': mean,
                'variance': variance,
                'skewness': skewness,
                'kurtosis': kurtosis
            }
        else:
            direct_params = None

        # Kolmogorov-Smirnov test
        ks_stat, ks_p_value = stats.kstest(data, dist.cdf)

        result = {
            'distribution': "Student's t",
            'parameterization': parameterization,
            'centered_parameters': {'degrees_of_freedom': df, 'location': loc, 'scale': scale},
            'direct_parameters': direct_params,
            'distribution_object': dist,
            'log_likelihood': np.sum(dist.logpdf(data)),
            'aic': 2 * 3 - 2 * np.sum(dist.logpdf(data)),  # 3 parameters
            'bic': np.log(len(data)) * 3 - 2 * np.sum(dist.logpdf(data)),
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p_value,
            'sample_size': len(data)
        }

        return result

    def fit_poisson(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Fit Poisson distribution to count data

        Args:
            data: Count data (non-negative integers)

        Returns:
            Dictionary with distribution parameters and fit statistics
        """
        # Validate count data
        if np.any(data < 0) or not np.allclose(data, np.round(data)):
            warnings.warn("Poisson distribution requires non-negative integer data")

        # Parameter estimation
        lambda_param = np.mean(data)

        # Create distribution object
        dist = stats.poisson(lambda_param)

        # Calculate goodness of fit
        unique_values = np.unique(data)
        observed_freq = np.array([np.sum(data == val) for val in unique_values])
        expected_freq = np.array([len(data) * dist.pmf(val) for val in unique_values])

        # Chi-square test (only for values with expected frequency > 5)
        valid_idx = expected_freq >= 5
        if np.sum(valid_idx) > 1:
            chi2_stat = np.sum((observed_freq[valid_idx] - expected_freq[valid_idx])**2 / expected_freq[valid_idx])
            chi2_p_value = 1 - stats.chi2.cdf(chi2_stat, df=np.sum(valid_idx) - 1)
        else:
            chi2_stat = np.nan
            chi2_p_value = np.nan

        result = {
            'distribution': 'Poisson',
            'parameters': {'lambda': lambda_param},
            'distribution_object': dist,
            'log_likelihood': np.sum(dist.logpmf(data)),
            'aic': 2 * 1 - 2 * np.sum(dist.logpmf(data)),  # 1 parameter
            'bic': np.log(len(data)) * 1 - 2 * np.sum(dist.logpmf(data)),
            'chi2_statistic': chi2_stat,
            'chi2_p_value': chi2_p_value,
            'sample_size': len(data)
        }

        return result

    def fit_beta(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Fit Beta distribution to data bounded between 0 and 1

        Args:
            data: Data bounded between 0 and 1

        Returns:
            Dictionary with distribution parameters and fit statistics
        """
        # Validate data bounds
        if np.any(data <= 0) or np.any(data >= 1):
            warnings.warn("Beta distribution requires data strictly between 0 and 1")
            # Optionally transform data to (0,1) interval
            data = np.clip(data, 1e-10, 1-1e-10)

        # Fit parameters
        a, b, loc, scale = stats.beta.fit(data)

        # Create distribution object
        dist = stats.beta(a, b, loc, scale)

        # Kolmogorov-Smirnov test
        ks_stat, ks_p_value = stats.kstest(data, dist.cdf)

        result = {
            'distribution': 'Beta',
            'parameters': {'alpha': a, 'beta': b, 'location': loc, 'scale': scale},
            'distribution_object': dist,
            'log_likelihood': np.sum(dist.logpdf(data)),
            'aic': 2 * 4 - 2 * np.sum(dist.logpdf(data)),  # 4 parameters
            'bic': np.log(len(data)) * 4 - 2 * np.sum(dist.logpdf(data)),
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p_value,
            'sample_size': len(data)
        }

        return result

    def fit_lognormal(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Fit Log-Normal distribution to positive data

        Args:
            data: Positive continuous data

        Returns:
            Dictionary with distribution parameters and fit statistics
        """
        # Validate positive data
        if np.any(data <= 0):
            warnings.warn("Log-normal distribution requires positive data")
            data = np.abs(data) + 1e-10

        # Fit parameters
        shape, loc, scale = stats.lognorm.fit(data)

        # Create distribution object
        dist = stats.lognorm(shape, loc, scale)

        # Kolmogorov-Smirnov test
        ks_stat, ks_p_value = stats.kstest(data, dist.cdf)

        # Alternative parameterization (mu, sigma of underlying normal)
        log_data = np.log(data - loc)
        mu = np.mean(log_data)
        sigma = np.std(log_data)

        result = {
            'distribution': 'Log-Normal',
            'parameters': {'shape': shape, 'location': loc, 'scale': scale},
            'alternative_parameters': {'mu': mu, 'sigma': sigma},
            'distribution_object': dist,
            'log_likelihood': np.sum(dist.logpdf(data)),
            'aic': 2 * 3 - 2 * np.sum(dist.logpdf(data)),  # 3 parameters
            'bic': np.log(len(data)) * 3 - 2 * np.sum(dist.logpdf(data)),
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p_value,
            'sample_size': len(data)
        }

        return result

    def fit_distribution(self, data: Union[pd.Series, np.ndarray],
                        distribution: str, **kwargs) -> Dict[str, Any]:
        """
        Fit a specified distribution to data

        Args:
            data: Data to fit
            distribution: Name of distribution to fit
            **kwargs: Additional parameters for specific distributions

        Returns:
            Dictionary with fit results
        """
        if isinstance(data, pd.Series):
            data = data.dropna().values

        if distribution not in self.available_distributions:
            raise ValueError(f"Distribution '{distribution}' not supported. "
                           f"Available: {self.available_distributions}")

        # Dispatch to appropriate fitting method
        if distribution == 'bernoulli':
            result = self.fit_bernoulli(data, **kwargs)
        elif distribution == 'binomial':
            result = self.fit_binomial(data, **kwargs)
        elif distribution == 'normal':
            result = self.fit_normal(data, **kwargs)
        elif distribution == 'skew_normal':
            result = self.fit_skew_normal(data, **kwargs)
        elif distribution == 'student_t':
            result = self.fit_student_t(data, **kwargs)
        elif distribution == 'poisson':
            result = self.fit_poisson(data, **kwargs)
        elif distribution == 'beta':
            result = self.fit_beta(data, **kwargs)
        elif distribution == 'lognormal':
            result = self.fit_lognormal(data, **kwargs)

        # Store results
        self.fitted_distributions[f"{distribution}_{len(self.fitted_distributions)}"] = result

        return result

    def compare_distributions(self, data: Union[pd.Series, np.ndarray],
                            distributions: Optional[List[str]] = None,
                            criterion: str = 'aic') -> pd.DataFrame:
        """
        Compare multiple distributions fitted to the same data

        Args:
            data: Data to fit
            distributions: List of distributions to compare (if None, tries all applicable)
            criterion: Comparison criterion ('aic', 'bic', 'log_likelihood')

        Returns:
            DataFrame with comparison results
        """
        if isinstance(data, pd.Series):
            data = data.dropna().values

        if distributions is None:
            # Automatically select applicable distributions based on data characteristics
            distributions = []

            # Check for binary data
            if np.array_equal(np.unique(data), [0, 1]):
                distributions.append('bernoulli')

            # Check for count data
            if np.all(data >= 0) and np.allclose(data, np.round(data)):
                distributions.extend(['poisson', 'binomial'])

            # Check for bounded data
            if np.all((data > 0) & (data < 1)):
                distributions.append('beta')

            # Check for positive data
            if np.all(data > 0):
                distributions.append('lognormal')

            # Always try these for continuous data
            distributions.extend(['normal', 'student_t'])

            # Try skew-normal if available
            if SKEWNORM_AVAILABLE:
                distributions.append('skew_normal')

            distributions = list(set(distributions))  # Remove duplicates

        comparison_results = []

        for dist_name in distributions:
            try:
                result = self.fit_distribution(data, dist_name)
                comparison_results.append({
                    'Distribution': result['distribution'],
                    'AIC': result.get('aic', np.nan),
                    'BIC': result.get('bic', np.nan),
                    'Log_Likelihood': result.get('log_likelihood', np.nan),
                    'KS_p_value': result.get('ks_p_value', np.nan),
                    'Chi2_p_value': result.get('chi2_p_value', np.nan),
                    'Parameters': str(result['parameters'])
                })
            except Exception as e:
                comparison_results.append({
                    'Distribution': dist_name,
                    'Error': str(e)
                })

        comparison_df = pd.DataFrame(comparison_results)

        # Sort by criterion (lower is better for AIC/BIC, higher for log-likelihood)
        if criterion in comparison_df.columns:
            ascending = criterion != 'Log_Likelihood'
            comparison_df = comparison_df.sort_values(criterion, ascending=ascending)

        return comparison_df

    def visualize_fit(self, data: Union[pd.Series, np.ndarray],
                     distribution: str,
                     figsize: Tuple[int, int] = (12, 8),
                     **kwargs) -> plt.Figure:
        """
        Visualize distribution fit

        Args:
            data: Data to fit and visualize
            distribution: Distribution name
            figsize: Figure size
            **kwargs: Additional parameters for distribution fitting

        Returns:
            Matplotlib figure object
        """
        if isinstance(data, pd.Series):
            data = data.dropna().values

        # Fit distribution
        result = self.fit_distribution(data, distribution, **kwargs)
        dist_obj = result['distribution_object']

        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Histogram with fitted PDF/PMF
        ax1 = axes[0, 0]
        ax1.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')

        if hasattr(dist_obj, 'pdf'):  # Continuous distribution
            x = np.linspace(data.min(), data.max(), 1000)
            ax1.plot(x, dist_obj.pdf(x), 'r-', linewidth=2, label=f'Fitted {distribution}')
        else:  # Discrete distribution
            x = np.arange(int(data.min()), int(data.max()) + 1)
            ax1.plot(x, dist_obj.pmf(x), 'ro-', linewidth=2, label=f'Fitted {distribution}')

        ax1.set_title(f'Histogram with Fitted {result["distribution"]}')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density/Probability')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Q-Q plot
        ax2 = axes[0, 1]
        if hasattr(dist_obj, 'ppf'):
            stats.probplot(data, dist=dist_obj, plot=ax2)
            ax2.set_title(f'Q-Q Plot: {result["distribution"]}')
        else:
            ax2.text(0.5, 0.5, 'Q-Q plot not available\nfor discrete distributions',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Q-Q Plot')

        # CDF comparison
        ax3 = axes[1, 0]
        sorted_data = np.sort(data)
        empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        theoretical_cdf = dist_obj.cdf(sorted_data)

        ax3.plot(sorted_data, empirical_cdf, 'b-', label='Empirical CDF', linewidth=2)
        ax3.plot(sorted_data, theoretical_cdf, 'r--', label=f'Theoretical CDF ({distribution})', linewidth=2)
        ax3.set_title('CDF Comparison')
        ax3.set_xlabel('Value')
        ax3.set_ylabel('Cumulative Probability')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Residuals (empirical - theoretical CDF)
        ax4 = axes[1, 1]
        residuals = empirical_cdf - theoretical_cdf
        ax4.plot(sorted_data, residuals, 'g-', linewidth=2)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.set_title('CDF Residuals')
        ax4.set_xlabel('Value')
        ax4.set_ylabel('Empirical - Theoretical CDF')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        return fig


# Utility Functions
def suggest_distributions(data: Union[pd.Series, np.ndarray]) -> List[str]:
    """
    Suggest appropriate distributions based on data characteristics

    Args:
        data: Data to analyze

    Returns:
        List of suggested distribution names
    """
    if isinstance(data, pd.Series):
        data = data.dropna().values

    suggestions = []

    # Data type and range analysis
    is_binary = np.array_equal(np.unique(data), [0, 1])
    is_count = np.all(data >= 0) and np.allclose(data, np.round(data))
    is_bounded_01 = np.all((data > 0) & (data < 1))
    is_positive = np.all(data > 0)

    # Distribution suggestions based on characteristics
    if is_binary:
        suggestions.append('bernoulli')

    if is_count:
        suggestions.extend(['poisson', 'binomial'])

    if is_bounded_01:
        suggestions.append('beta')

    if is_positive:
        suggestions.append('lognormal')

    # Always consider these for continuous data
    if not is_binary and not is_count:
        suggestions.extend(['normal', 'student_t'])
        if SKEWNORM_AVAILABLE:
            suggestions.append('skew_normal')

    # Check skewness for additional suggestions
    skewness = stats.skew(data)
    if abs(skewness) > 1:
        if SKEWNORM_AVAILABLE:
            suggestions.append('skew_normal')
        if is_positive:
            suggestions.append('lognormal')

    return list(set(suggestions))  # Remove duplicates


class EmpiricalDistributionAnalyzer:
    """
    Class for analyzing empirical distribution functions and cumulative distribution functions
    """

    def __init__(self):
        self.empirical_results = {}

    def compute_empirical_cdf(self, data: Union[pd.Series, np.ndarray],
                             x_values: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute empirical cumulative distribution function

        Args:
            data: Data vector
            x_values: Points at which to evaluate CDF (if None, uses data points)

        Returns:
            Dictionary with empirical CDF results
        """
        if isinstance(data, pd.Series):
            data = data.dropna().values

        n = len(data)
        sorted_data = np.sort(data)

        if x_values is None:
            x_values = sorted_data

        # Compute empirical CDF using step function
        empirical_cdf = np.zeros(len(x_values))
        for i, x in enumerate(x_values):
            empirical_cdf[i] = np.sum(sorted_data <= x) / n

        result = {
            'x_values': x_values,
            'empirical_cdf': empirical_cdf,
            'sorted_data': sorted_data,
            'sample_size': n,
            'min_value': np.min(data),
            'max_value': np.max(data),
            'median': np.median(data),
            'mean': np.mean(data)
        }

        return result

    def compute_empirical_pdf(self, data: Union[pd.Series, np.ndarray],
                             method: str = 'histogram',
                             bins: Union[int, str] = 'auto',
                             bandwidth: Optional[float] = None) -> Dict[str, Any]:
        """
        Compute empirical probability density function

        Args:
            data: Data vector
            method: Method for PDF estimation ('histogram', 'kde')
            bins: Number of bins for histogram or binning strategy
            bandwidth: Bandwidth for KDE (if None, uses Scott's rule)

        Returns:
            Dictionary with empirical PDF results
        """
        if isinstance(data, pd.Series):
            data = data.dropna().values

        if method == 'histogram':
            counts, bin_edges = np.histogram(data, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            result = {
                'method': 'histogram',
                'x_values': bin_centers,
                'pdf_values': counts,
                'bin_edges': bin_edges,
                'bin_width': np.mean(np.diff(bin_edges))
            }

        elif method == 'kde':
            try:
                from scipy.stats import gaussian_kde

                if bandwidth is None:
                    kde = gaussian_kde(data)
                else:
                    kde = gaussian_kde(data, bw_method=bandwidth)

                x_range = np.linspace(data.min(), data.max(), 100)
                pdf_values = kde(x_range)

                result = {
                    'method': 'kde',
                    'x_values': x_range,
                    'pdf_values': pdf_values,
                    'bandwidth': kde.factor * np.std(data),
                    'kde_object': kde
                }
            except ImportError:
                raise ImportError("Scipy required for KDE estimation")

        else:
            raise ValueError("Method must be 'histogram' or 'kde'")

        result.update({
            'sample_size': len(data),
            'data_range': (np.min(data), np.max(data))
        })

        return result

    def plot_empirical_distributions(self, data: Union[pd.Series, np.ndarray],
                                   compare_theoretical: Optional[str] = None,
                                   figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot empirical CDF and PDF

        Args:
            data: Data vector
            compare_theoretical: Theoretical distribution to compare ('normal', 'uniform', etc.)
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        if isinstance(data, pd.Series):
            data = data.dropna().values

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # Empirical CDF
        cdf_result = self.compute_empirical_cdf(data)
        ax1.step(cdf_result['x_values'], cdf_result['empirical_cdf'],
                 where='post', linewidth=2, label='Empirical CDF')
        ax1.set_title('Empirical Cumulative Distribution Function')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Cumulative Probability')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Empirical PDF (histogram)
        pdf_hist = self.compute_empirical_pdf(data, method='histogram')
        ax2.bar(pdf_hist['x_values'], pdf_hist['pdf_values'],
                width=pdf_hist['bin_width'], alpha=0.7,
                label='Empirical PDF (Histogram)', edgecolor='black')
        ax2.set_title('Empirical Probability Density Function')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Density')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Empirical PDF (KDE)
        try:
            pdf_kde = self.compute_empirical_pdf(data, method='kde')
            ax3.plot(pdf_kde['x_values'], pdf_kde['pdf_values'],
                     linewidth=2, label='Empirical PDF (KDE)')
            ax3.fill_between(pdf_kde['x_values'], pdf_kde['pdf_values'], alpha=0.3)
        except ImportError:
            ax3.hist(data, bins=30, density=True, alpha=0.7, label='Histogram fallback')

        ax3.set_title('Empirical PDF (Kernel Density Estimation)')
        ax3.set_xlabel('Value')
        ax3.set_ylabel('Density')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Comparison with theoretical distribution
        ax4.step(cdf_result['x_values'], cdf_result['empirical_cdf'],
                 where='post', linewidth=2, label='Empirical CDF')

        if compare_theoretical:
            if compare_theoretical == 'normal':
                mu, sigma = np.mean(data), np.std(data)
                theoretical_cdf = stats.norm.cdf(cdf_result['x_values'], mu, sigma)
                ax4.plot(cdf_result['x_values'], theoretical_cdf,
                         'r--', linewidth=2, label=f'Normal CDF (μ={mu:.2f}, σ={sigma:.2f})')
            elif compare_theoretical == 'uniform':
                a, b = np.min(data), np.max(data)
                theoretical_cdf = stats.uniform.cdf(cdf_result['x_values'], a, b-a)
                ax4.plot(cdf_result['x_values'], theoretical_cdf,
                         'g--', linewidth=2, label=f'Uniform CDF (a={a:.2f}, b={b:.2f})')

        ax4.set_title(f'CDF Comparison' + (f' with {compare_theoretical}' if compare_theoretical else ''))
        ax4.set_xlabel('Value')
        ax4.set_ylabel('Cumulative Probability')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.tight_layout()
        return fig


class JointDistributionAnalyzer:
    """
    Class for analyzing joint and marginal probability distributions of vector pairs
    """

    def __init__(self):
        self.joint_results = {}

    def compute_joint_empirical_distribution(self, x_data: Union[pd.Series, np.ndarray],
                                           y_data: Union[pd.Series, np.ndarray],
                                           bins: Union[int, Tuple[int, int]] = 20) -> Dict[str, Any]:
        """
        Compute joint empirical distribution of two variables

        Args:
            x_data: First variable data
            y_data: Second variable data
            bins: Number of bins for each dimension

        Returns:
            Dictionary with joint distribution results
        """
        if isinstance(x_data, pd.Series):
            x_data = x_data.dropna().values
        if isinstance(y_data, pd.Series):
            y_data = y_data.dropna().values

        if len(x_data) != len(y_data):
            min_len = min(len(x_data), len(y_data))
            x_data = x_data[:min_len]
            y_data = y_data[:min_len]
            warnings.warn(f"Data length mismatch. Using first {min_len} observations.")

        # Compute 2D histogram (joint PDF)
        joint_pdf, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=bins, density=True)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        # Compute marginal PDFs
        marginal_x = np.sum(joint_pdf, axis=1) * (y_edges[1] - y_edges[0])
        marginal_y = np.sum(joint_pdf, axis=0) * (x_edges[1] - x_edges[0])

        # Compute joint CDF
        joint_cdf = np.zeros_like(joint_pdf)
        for i in range(len(x_centers)):
            for j in range(len(y_centers)):
                joint_cdf[i, j] = np.sum(joint_pdf[:i+1, :j+1]) * (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])

        result = {
            'joint_pdf': joint_pdf,
            'joint_cdf': joint_cdf,
            'x_centers': x_centers,
            'y_centers': y_centers,
            'x_edges': x_edges,
            'y_edges': y_edges,
            'marginal_x': marginal_x,
            'marginal_y': marginal_y,
            'sample_size': len(x_data),
            'correlation': np.corrcoef(x_data, y_data)[0, 1]
        }

        return result

    def compute_marginal_distributions(self, x_data: Union[pd.Series, np.ndarray],
                                     y_data: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Compute marginal distributions from joint data

        Args:
            x_data: First variable data
            y_data: Second variable data

        Returns:
            Dictionary with marginal distribution results
        """
        if isinstance(x_data, pd.Series):
            x_data = x_data.dropna().values
        if isinstance(y_data, pd.Series):
            y_data = y_data.dropna().values

        # Create empirical distribution analyzer
        emp_analyzer = EmpiricalDistributionAnalyzer()

        # Compute marginal CDFs and PDFs
        marginal_x_cdf = emp_analyzer.compute_empirical_cdf(x_data)
        marginal_y_cdf = emp_analyzer.compute_empirical_cdf(y_data)
        marginal_x_pdf = emp_analyzer.compute_empirical_pdf(x_data)
        marginal_y_pdf = emp_analyzer.compute_empirical_pdf(y_data)

        result = {
            'marginal_x': {
                'cdf': marginal_x_cdf,
                'pdf': marginal_x_pdf,
                'mean': np.mean(x_data),
                'std': np.std(x_data),
                'skewness': stats.skew(x_data),
                'kurtosis': stats.kurtosis(x_data)
            },
            'marginal_y': {
                'cdf': marginal_y_cdf,
                'pdf': marginal_y_pdf,
                'mean': np.mean(y_data),
                'std': np.std(y_data),
                'skewness': stats.skew(y_data),
                'kurtosis': stats.kurtosis(y_data)
            },
            'joint_statistics': {
                'correlation': np.corrcoef(x_data, y_data)[0, 1],
                'covariance': np.cov(x_data, y_data)[0, 1],
                'kendall_tau': stats.kendalltau(x_data, y_data)[0],
                'spearman_rho': stats.spearmanr(x_data, y_data)[0]
            }
        }

        return result

    def plot_joint_distribution(self, x_data: Union[pd.Series, np.ndarray],
                               y_data: Union[pd.Series, np.ndarray],
                               figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Plot joint and marginal distributions

        Args:
            x_data: First variable data
            y_data: Second variable data
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        if isinstance(x_data, pd.Series):
            x_data = x_data.dropna().values
        if isinstance(y_data, pd.Series):
            y_data = y_data.dropna().values

        # Compute joint distribution
        joint_result = self.compute_joint_empirical_distribution(x_data, y_data)

        fig = plt.figure(figsize=figsize)

        # Create grid layout
        gs = fig.add_gridspec(3, 3, width_ratios=[2, 4, 1], height_ratios=[1, 4, 1])

        # Joint distribution heatmap
        ax_joint = fig.add_subplot(gs[1, 1])
        im = ax_joint.imshow(joint_result['joint_pdf'].T, origin='lower', aspect='auto',
                            extent=[joint_result['x_edges'][0], joint_result['x_edges'][-1],
                                   joint_result['y_edges'][0], joint_result['y_edges'][-1]],
                            cmap='Blues')
        ax_joint.set_xlabel('X Variable')
        ax_joint.set_ylabel('Y Variable')
        ax_joint.set_title('Joint Probability Density')

        # Add scatter plot overlay
        ax_joint.scatter(x_data, y_data, alpha=0.5, s=1, color='red')

        # Marginal X distribution
        ax_marg_x = fig.add_subplot(gs[0, 1], sharex=ax_joint)
        ax_marg_x.bar(joint_result['x_centers'], joint_result['marginal_x'],
                      width=np.diff(joint_result['x_edges'])[0], alpha=0.7)
        ax_marg_x.set_title('Marginal Distribution of X')
        ax_marg_x.tick_params(axis='x', labelbottom=False)

        # Marginal Y distribution
        ax_marg_y = fig.add_subplot(gs[1, 2], sharey=ax_joint)
        ax_marg_y.barh(joint_result['y_centers'], joint_result['marginal_y'],
                       height=np.diff(joint_result['y_edges'])[0], alpha=0.7)
        ax_marg_y.set_title('Marginal Distribution of Y', rotation=270, pad=20)
        ax_marg_y.tick_params(axis='y', labelleft=False)

        # Colorbar
        ax_cbar = fig.add_subplot(gs[1, 0])
        fig.colorbar(im, cax=ax_cbar)
        ax_cbar.set_ylabel('Density')

        # Statistics text
        ax_stats = fig.add_subplot(gs[2, :])
        ax_stats.axis('off')
        stats_text = f"Correlation: {joint_result['correlation']:.3f}"
        ax_stats.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12)

        plt.tight_layout()
        return fig


class CopulaAnalyzer:
    """
    Class for copula analysis including elliptical and Archimedean copulas

    Copula Guide:
    =============

    Copulas separate the marginal distributions from the dependence structure, allowing
    flexible modeling of multivariate relationships.

    Types of Copulas:

    1. Elliptical Copulas:
       - Normal (Gaussian): Symmetric dependence, captures linear relationships
       - Student's t: Like Normal but with tail dependence

    2. Archimedean Copulas:
       - Clayton: Lower tail dependence, asymmetric
       - Gumbel: Upper tail dependence, asymmetric
       - Frank: Symmetric, no tail dependence

    Parameters:
    - Normal/t-copula: Correlation matrix (and degrees of freedom for t)
    - Clayton: θ > 0 (higher = stronger lower tail dependence)
    - Gumbel: θ >= 1 (higher = stronger upper tail dependence)

    Kendall's Tau:
    - Non-parametric measure of rank correlation
    - Used for parameter estimation and goodness-of-fit testing
    - Range: [-1, 1], 0 = independence
    """

    def __init__(self):
        self.copula_results = {}
        self.copula_guide = """
        Copula Selection Guide:
        ======================

        Use Normal Copula when:
        - Linear relationships dominate
        - Symmetric dependence expected
        - No extreme tail behavior

        Use Student's t Copula when:
        - Heavy tails in joint distribution
        - Tail dependence present
        - Financial applications with crisis periods

        Use Clayton Copula when:
        - Lower tail dependence important
        - Downside risk modeling
        - Asymmetric relationships

        Use Gumbel Copula when:
        - Upper tail dependence important
        - Extreme positive events cluster
        - Environmental/reliability applications
        """

    def transform_to_uniform_margins(self, x_data: Union[pd.Series, np.ndarray],
                                   y_data: Union[pd.Series, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data to uniform margins using empirical CDF (pseudo-observations)

        Args:
            x_data: First variable data
            y_data: Second variable data

        Returns:
            Tuple of transformed data with uniform margins
        """
        if isinstance(x_data, pd.Series):
            x_data = x_data.dropna().values
        if isinstance(y_data, pd.Series):
            y_data = y_data.dropna().values

        n = len(x_data)

        # Compute empirical CDFs
        x_ranks = stats.rankdata(x_data, method='ordinal')
        y_ranks = stats.rankdata(y_data, method='ordinal')

        # Transform to uniform [0,1] using empirical CDF
        u = x_ranks / (n + 1)
        v = y_ranks / (n + 1)

        return u, v

    def fit_normal_copula(self, x_data: Union[pd.Series, np.ndarray],
                         y_data: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Fit Normal (Gaussian) copula

        Args:
            x_data: First variable data
            y_data: Second variable data

        Returns:
            Dictionary with copula parameters and fit statistics
        """
        # Transform to uniform margins
        u, v = self.transform_to_uniform_margins(x_data, y_data)

        # Transform to standard normal
        z_u = stats.norm.ppf(u)
        z_v = stats.norm.ppf(v)

        # Estimate correlation parameter
        rho = np.corrcoef(z_u, z_v)[0, 1]

        # Compute Kendall's tau
        kendall_tau = stats.kendalltau(x_data, y_data)[0]
        theoretical_tau = (2 / np.pi) * np.arcsin(rho)

        # Log-likelihood
        log_likelihood = np.sum(stats.multivariate_normal.logpdf(
            np.column_stack([z_u, z_v]),
            mean=[0, 0],
            cov=[[1, rho], [rho, 1]]
        )) - np.sum(stats.norm.logpdf(z_u)) - np.sum(stats.norm.logpdf(z_v))

        result = {
            'copula_type': 'Normal',
            'parameters': {'rho': rho},
            'kendall_tau_observed': kendall_tau,
            'kendall_tau_theoretical': theoretical_tau,
            'tau_difference': abs(kendall_tau - theoretical_tau),
            'log_likelihood': log_likelihood,
            'aic': 2 * 1 - 2 * log_likelihood,  # 1 parameter
            'bic': np.log(len(x_data)) * 1 - 2 * log_likelihood,
            'u_values': u,
            'v_values': v
        }

        return result

    def fit_student_t_copula(self, x_data: Union[pd.Series, np.ndarray],
                           y_data: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Fit Student's t copula

        Args:
            x_data: First variable data
            y_data: Second variable data

        Returns:
            Dictionary with copula parameters and fit statistics
        """
        # Transform to uniform margins
        u, v = self.transform_to_uniform_margins(x_data, y_data)

        # Initial parameter estimates
        kendall_tau = stats.kendalltau(x_data, y_data)[0]
        rho_init = np.sin(kendall_tau * np.pi / 2)

        # Fit Student's t copula using MLE (simplified approach)
        # For production use, consider more sophisticated optimization
        def negative_log_likelihood(params):
            rho, nu = params
            if abs(rho) >= 1 or nu <= 2:
                return np.inf

            try:
                # Transform to t-distribution
                t_u = stats.t.ppf(u, df=nu)
                t_v = stats.t.ppf(v, df=nu)

                # Bivariate t log-likelihood
                ll = np.sum(stats.multivariate_t.logpdf(
                    np.column_stack([t_u, t_v]),
                    loc=[0, 0],
                    shape=[[1, rho], [rho, 1]],
                    df=nu
                )) - np.sum(stats.t.logpdf(t_u, df=nu)) - np.sum(stats.t.logpdf(t_v, df=nu))

                return -ll
            except:
                return np.inf

        try:
            from scipy.optimize import minimize
            result_opt = minimize(negative_log_likelihood, [rho_init, 5],
                                bounds=[(-0.99, 0.99), (2.1, 30)],
                                method='L-BFGS-B')
            rho, nu = result_opt.x
            log_likelihood = -result_opt.fun
        except:
            # Fallback if optimization fails
            rho = rho_init
            nu = 5
            log_likelihood = -negative_log_likelihood([rho, nu])

        # Theoretical Kendall's tau for t-copula (same as Normal copula)
        theoretical_tau = (2 / np.pi) * np.arcsin(rho)

        result = {
            'copula_type': "Student's t",
            'parameters': {'rho': rho, 'degrees_of_freedom': nu},
            'kendall_tau_observed': kendall_tau,
            'kendall_tau_theoretical': theoretical_tau,
            'tau_difference': abs(kendall_tau - theoretical_tau),
            'log_likelihood': log_likelihood,
            'aic': 2 * 2 - 2 * log_likelihood,  # 2 parameters
            'bic': np.log(len(x_data)) * 2 - 2 * log_likelihood,
            'u_values': u,
            'v_values': v
        }

        return result

    def fit_clayton_copula(self, x_data: Union[pd.Series, np.ndarray],
                          y_data: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Fit Clayton copula (Archimedean family)

        Args:
            x_data: First variable data
            y_data: Second variable data

        Returns:
            Dictionary with copula parameters and fit statistics
        """
        # Transform to uniform margins
        u, v = self.transform_to_uniform_margins(x_data, y_data)

        # Estimate parameter using Kendall's tau
        kendall_tau = stats.kendalltau(x_data, y_data)[0]

        if kendall_tau <= 0:
            warnings.warn("Clayton copula requires positive dependence. Using theta=0.1")
            theta = 0.1
        else:
            theta = 2 * kendall_tau / (1 - kendall_tau)

        # Clayton copula CDF: C(u,v) = (u^(-θ) + v^(-θ) - 1)^(-1/θ)
        # Log-likelihood for Clayton copula
        def clayton_logpdf(u, v, theta):
            if theta <= 0:
                return -np.inf
            try:
                term1 = np.log(1 + theta)
                term2 = -(1 + 2*theta) * (np.log(u) + np.log(v))
                term3 = -(1/theta + 2) * np.log(u**(-theta) + v**(-theta) - 1)
                return term1 + term2 + term3
            except:
                return -np.inf

        log_likelihood = np.sum([clayton_logpdf(u[i], v[i], theta) for i in range(len(u))])

        # Theoretical Kendall's tau
        theoretical_tau = theta / (theta + 2)

        result = {
            'copula_type': 'Clayton',
            'parameters': {'theta': theta},
            'kendall_tau_observed': kendall_tau,
            'kendall_tau_theoretical': theoretical_tau,
            'tau_difference': abs(kendall_tau - theoretical_tau),
            'log_likelihood': log_likelihood,
            'aic': 2 * 1 - 2 * log_likelihood,  # 1 parameter
            'bic': np.log(len(x_data)) * 1 - 2 * log_likelihood,
            'u_values': u,
            'v_values': v,
            'tail_dependence': {'lower': 2**(-1/theta), 'upper': 0}
        }

        return result

    def fit_gumbel_copula(self, x_data: Union[pd.Series, np.ndarray],
                         y_data: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Fit Gumbel copula (Archimedean family)

        Args:
            x_data: First variable data
            y_data: Second variable data

        Returns:
            Dictionary with copula parameters and fit statistics
        """
        # Transform to uniform margins
        u, v = self.transform_to_uniform_margins(x_data, y_data)

        # Estimate parameter using Kendall's tau
        kendall_tau = stats.kendalltau(x_data, y_data)[0]

        if kendall_tau <= 0:
            warnings.warn("Gumbel copula requires positive dependence. Using theta=1.1")
            theta = 1.1
        else:
            theta = 1 / (1 - kendall_tau)

        if theta < 1:
            theta = 1.1

        # Gumbel copula log-likelihood (simplified)
        def gumbel_logpdf(u, v, theta):
            if theta < 1:
                return -np.inf
            try:
                A = (-np.log(u))**theta + (-np.log(v))**theta
                C = np.exp(-A**(1/theta))

                term1 = np.log(C)
                term2 = np.log(A**(1/theta - 1))
                term3 = np.log(1 + (theta - 1) * A**(-1/theta))
                term4 = (theta - 1) * (np.log(-np.log(u)) + np.log(-np.log(v)))
                term5 = -np.log(u) - np.log(v)

                return term1 + term2 + term3 + term4 + term5
            except:
                return -np.inf

        log_likelihood = np.sum([gumbel_logpdf(u[i], v[i], theta) for i in range(len(u))])

        # Theoretical Kendall's tau
        theoretical_tau = (theta - 1) / theta

        result = {
            'copula_type': 'Gumbel',
            'parameters': {'theta': theta},
            'kendall_tau_observed': kendall_tau,
            'kendall_tau_theoretical': theoretical_tau,
            'tau_difference': abs(kendall_tau - theoretical_tau),
            'log_likelihood': log_likelihood,
            'aic': 2 * 1 - 2 * log_likelihood,  # 1 parameter
            'bic': np.log(len(x_data)) * 1 - 2 * log_likelihood,
            'u_values': u,
            'v_values': v,
            'tail_dependence': {'lower': 0, 'upper': 2 - 2**(1/theta)}
        }

        return result

    def compare_copulas(self, x_data: Union[pd.Series, np.ndarray],
                       y_data: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
        """
        Compare different copula models

        Args:
            x_data: First variable data
            y_data: Second variable data

        Returns:
            DataFrame with comparison results
        """
        copula_fits = []

        # Fit all copulas
        try:
            normal_fit = self.fit_normal_copula(x_data, y_data)
            copula_fits.append(normal_fit)
        except Exception as e:
            print(f"Normal copula failed: {e}")

        try:
            t_fit = self.fit_student_t_copula(x_data, y_data)
            copula_fits.append(t_fit)
        except Exception as e:
            print(f"Student's t copula failed: {e}")

        try:
            clayton_fit = self.fit_clayton_copula(x_data, y_data)
            copula_fits.append(clayton_fit)
        except Exception as e:
            print(f"Clayton copula failed: {e}")

        try:
            gumbel_fit = self.fit_gumbel_copula(x_data, y_data)
            copula_fits.append(gumbel_fit)
        except Exception as e:
            print(f"Gumbel copula failed: {e}")

        # Create comparison DataFrame
        comparison_data = []
        for fit in copula_fits:
            comparison_data.append({
                'Copula': fit['copula_type'],
                'AIC': fit['aic'],
                'BIC': fit['bic'],
                'Log_Likelihood': fit['log_likelihood'],
                'Kendall_Tau_Observed': fit['kendall_tau_observed'],
                'Kendall_Tau_Theoretical': fit['kendall_tau_theoretical'],
                'Tau_Goodness_of_Fit': fit['tau_difference'],
                'Parameters': str(fit['parameters'])
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('AIC')  # Best fit has lowest AIC

        return comparison_df

    def plot_copula_comparison(self, x_data: Union[pd.Series, np.ndarray],
                              y_data: Union[pd.Series, np.ndarray],
                              figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
        """
        Plot copula fits for visual comparison

        Args:
            x_data: First variable data
            y_data: Second variable data
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        # Transform to uniform margins
        u, v = self.transform_to_uniform_margins(x_data, y_data)

        # Fit copulas
        normal_fit = self.fit_normal_copula(x_data, y_data)
        clayton_fit = self.fit_clayton_copula(x_data, y_data)
        gumbel_fit = self.fit_gumbel_copula(x_data, y_data)

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Original data scatter plot
        axes[0, 0].scatter(x_data, y_data, alpha=0.6, s=20)
        axes[0, 0].set_title('Original Data')
        axes[0, 0].set_xlabel('X Variable')
        axes[0, 0].set_ylabel('Y Variable')
        axes[0, 0].grid(True, alpha=0.3)

        # Uniform margins (pseudo-observations)
        axes[0, 1].scatter(u, v, alpha=0.6, s=20)
        axes[0, 1].set_title('Pseudo-observations (Uniform Margins)')
        axes[0, 1].set_xlabel('U (Empirical CDF of X)')
        axes[0, 1].set_ylabel('V (Empirical CDF of Y)')
        axes[0, 1].grid(True, alpha=0.3)

        # Kendall's tau comparison
        copulas = ['Observed', 'Normal', 'Clayton', 'Gumbel']
        tau_values = [
            stats.kendalltau(x_data, y_data)[0],
            normal_fit['kendall_tau_theoretical'],
            clayton_fit['kendall_tau_theoretical'],
            gumbel_fit['kendall_tau_theoretical']
        ]

        axes[0, 2].bar(copulas, tau_values, alpha=0.7)
        axes[0, 2].set_title("Kendall's Tau Comparison")
        axes[0, 2].set_ylabel("Kendall's Tau")
        axes[0, 2].tick_params(axis='x', rotation=45)

        # Model comparison (AIC)
        copula_names = ['Normal', 'Clayton', 'Gumbel']
        aic_values = [normal_fit['aic'], clayton_fit['aic'], gumbel_fit['aic']]

        axes[1, 0].bar(copula_names, aic_values, alpha=0.7)
        axes[1, 0].set_title('Model Comparison (AIC - Lower is Better)')
        axes[1, 0].set_ylabel('AIC')

        # Tail dependence comparison
        tail_dep_data = {
            'Normal': [0, 0],
            'Clayton': [clayton_fit.get('tail_dependence', {}).get('lower', 0), 0],
            'Gumbel': [0, gumbel_fit.get('tail_dependence', {}).get('upper', 0)]
        }

        x_pos = np.arange(len(copula_names))
        lower_tail = [tail_dep_data[name][0] for name in copula_names]
        upper_tail = [tail_dep_data[name][1] for name in copula_names]

        axes[1, 1].bar(x_pos - 0.2, lower_tail, 0.4, label='Lower Tail', alpha=0.7)
        axes[1, 1].bar(x_pos + 0.2, upper_tail, 0.4, label='Upper Tail', alpha=0.7)
        axes[1, 1].set_title('Tail Dependence')
        axes[1, 1].set_ylabel('Tail Dependence Coefficient')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(copula_names)
        axes[1, 1].legend()

        # Goodness of fit (Kendall's tau difference)
        tau_diff = [
            normal_fit['tau_difference'],
            clayton_fit['tau_difference'],
            gumbel_fit['tau_difference']
        ]

        axes[1, 2].bar(copula_names, tau_diff, alpha=0.7)
        axes[1, 2].set_title('Goodness of Fit (Lower is Better)')
        axes[1, 2].set_ylabel('|Observed τ - Theoretical τ|')

        plt.tight_layout()
        return fig

    def get_copula_guide(self) -> str:
        """Return comprehensive copula guide"""
        return self.copula_guide


if __name__ == "__main__":
    print("Probability Distributions Module")
    print("================================")

    # Create sample data for demonstration
    np.random.seed(42)

    # Normal data
    normal_data = np.random.normal(100, 15, 1000)

    # Skewed data
    skewed_data = np.random.exponential(2, 1000)

    # Count data
    count_data = np.random.poisson(5, 1000)

    # Binary data
    binary_data = np.random.binomial(1, 0.3, 1000)

    print("Sample datasets created")
    print(f"Normal data: mean={normal_data.mean():.2f}, std={normal_data.std():.2f}")
    print(f"Skewed data: mean={skewed_data.mean():.2f}, std={skewed_data.std():.2f}")
    print(f"Count data: mean={count_data.mean():.2f}, std={count_data.std():.2f}")
    print(f"Binary data: mean={binary_data.mean():.2f}, std={binary_data.std():.2f}")

    # Test normality
    print("\n=== Normality Testing ===")
    tester = NormalityTester()
    df_test = pd.DataFrame({'normal': normal_data, 'skewed': skewed_data})

    normal_results = tester.comprehensive_normality_test(df_test, 'normal', create_plots=False)
    print(f"Normal data - Overall assessment: {normal_results['overall_assessment']['consensus']}")

    skewed_results = tester.comprehensive_normality_test(df_test, 'skewed', create_plots=False)
    print(f"Skewed data - Overall assessment: {skewed_results['overall_assessment']['consensus']}")

    # Test distribution fitting
    print("\n=== Distribution Modeling ===")
    modeler = DistributionModeler()

    # Fit normal distribution
    normal_fit = modeler.fit_distribution(normal_data, 'normal')
    print(f"Normal fit - AIC: {normal_fit['aic']:.2f}, Parameters: {normal_fit['parameters']}")

    # Fit Poisson distribution
    poisson_fit = modeler.fit_distribution(count_data, 'poisson')
    print(f"Poisson fit - AIC: {poisson_fit['aic']:.2f}, Parameters: {poisson_fit['parameters']}")

    # Compare distributions
    print("\n=== Distribution Comparison ===")
    comparison = modeler.compare_distributions(normal_data, ['normal', 'student_t', 'lognormal'])
    print("Best fitting distributions for normal data:")
    print(comparison.head(3))

    print("\nDistribution modeling module ready for use!")