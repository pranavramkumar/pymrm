"""
Hypothesis Testing Module for Financial and Economic Data

This module provides comprehensive hypothesis testing functionality including:
- Normal distribution CDF and related tests
- Z-tests with significance levels
- T-tests (one-sample, two-sample, paired) with degrees of freedom
- ANOVA tables and F-tests
- Joint hypothesis testing (F-tests) with p-values
- Chi-squared tests of independence and sample size calculation
- Power analysis and effect size calculations

Dependencies: scipy, numpy, pandas, matplotlib, statsmodels
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import Optional, Dict, Any, List, Union, Tuple
from scipy import stats
from scipy.stats import norm, t, f, chi2
import itertools

# Optional imports for enhanced functionality
try:
    import statsmodels.api as sm
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.power import ttest_power, ftest_power
    from statsmodels.stats.contingency_tables import mcnemar
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("Statsmodels not available. Some advanced features will be disabled.")

try:
    from scipy.stats.contingency import chi2_contingency
    CHI2_CONTINGENCY_AVAILABLE = True
except ImportError:
    CHI2_CONTINGENCY_AVAILABLE = False


class NormalDistributionTester:
    """
    Class for normal distribution CDF calculations and related tests
    """

    def __init__(self):
        self.results = {}

    def compute_normal_cdf(self, x: Union[float, np.ndarray, List[float]],
                          mean: float = 0.0, std: float = 1.0) -> Union[float, np.ndarray]:
        """
        Compute cumulative density function for normal distribution

        Args:
            x: Value(s) at which to evaluate CDF
            mean: Mean of the normal distribution
            std: Standard deviation of the normal distribution

        Returns:
            CDF value(s)
        """
        if isinstance(x, (list, tuple)):
            x = np.array(x)

        cdf_values = stats.norm.cdf(x, loc=mean, scale=std)
        return cdf_values

    def compute_normal_pdf(self, x: Union[float, np.ndarray, List[float]],
                          mean: float = 0.0, std: float = 1.0) -> Union[float, np.ndarray]:
        """
        Compute probability density function for normal distribution

        Args:
            x: Value(s) at which to evaluate PDF
            mean: Mean of the normal distribution
            std: Standard deviation of the normal distribution

        Returns:
            PDF value(s)
        """
        if isinstance(x, (list, tuple)):
            x = np.array(x)

        pdf_values = stats.norm.pdf(x, loc=mean, scale=std)
        return pdf_values

    def compute_normal_quantile(self, p: Union[float, np.ndarray, List[float]],
                               mean: float = 0.0, std: float = 1.0) -> Union[float, np.ndarray]:
        """
        Compute quantile (inverse CDF) for normal distribution

        Args:
            p: Probability value(s) (0 < p < 1)
            mean: Mean of the normal distribution
            std: Standard deviation of the normal distribution

        Returns:
            Quantile value(s)
        """
        if isinstance(p, (list, tuple)):
            p = np.array(p)

        quantile_values = stats.norm.ppf(p, loc=mean, scale=std)
        return quantile_values

    def plot_normal_distribution(self, mean: float = 0.0, std: float = 1.0,
                                x_range: Optional[Tuple[float, float]] = None,
                                highlight_area: Optional[Tuple[float, float]] = None,
                                figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot normal distribution with optional highlighted area

        Args:
            mean: Mean of the distribution
            std: Standard deviation of the distribution
            x_range: Range of x values to plot
            highlight_area: Tuple of (lower, upper) bounds to highlight
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        if x_range is None:
            x_range = (mean - 4*std, mean + 4*std)

        x = np.linspace(x_range[0], x_range[1], 1000)
        y = self.compute_normal_pdf(x, mean, std)

        fig, ax = plt.subplots(figsize=figsize)

        # Plot the distribution
        ax.plot(x, y, 'b-', linewidth=2, label=f'N({mean}, {std}²)')
        ax.fill_between(x, y, alpha=0.1)

        # Highlight specific area if requested
        if highlight_area:
            mask = (x >= highlight_area[0]) & (x <= highlight_area[1])
            ax.fill_between(x[mask], y[mask], alpha=0.3, color='red',
                           label=f'P({highlight_area[0]} ≤ X ≤ {highlight_area[1]})')

            # Calculate and display probability
            prob = self.compute_normal_cdf(highlight_area[1], mean, std) - \
                   self.compute_normal_cdf(highlight_area[0], mean, std)
            ax.text(0.02, 0.95, f'Probability = {prob:.4f}',
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.axvline(mean, color='black', linestyle='--', alpha=0.7, label=f'μ = {mean}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Normal Distribution (μ={mean}, σ={std})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class ZTester:
    """
    Class for Z-tests with significance levels
    """

    def __init__(self):
        self.test_results = {}

    def one_sample_z_test(self, sample: Union[List[float], np.ndarray, pd.Series],
                         population_mean: float,
                         population_std: Optional[float] = None,
                         alpha: float = 0.05,
                         alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Perform one-sample Z-test

        Args:
            sample: Sample data
            population_mean: Hypothesized population mean (H0)
            population_std: Known population standard deviation
            alpha: Significance level
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            Dictionary with test results
        """
        if isinstance(sample, pd.Series):
            sample = sample.dropna().values
        elif isinstance(sample, list):
            sample = np.array(sample)

        n = len(sample)
        sample_mean = np.mean(sample)

        # Use population std if provided, otherwise use sample std
        if population_std is None:
            population_std = np.std(sample, ddof=1)
            warnings.warn("Population standard deviation not provided. Using sample standard deviation.")

        # Calculate Z-statistic
        standard_error = population_std / np.sqrt(n)
        z_statistic = (sample_mean - population_mean) / standard_error

        # Calculate p-value based on alternative hypothesis
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
        elif alternative == 'greater':
            p_value = 1 - stats.norm.cdf(z_statistic)
        elif alternative == 'less':
            p_value = stats.norm.cdf(z_statistic)
        else:
            raise ValueError("Alternative must be 'two-sided', 'less', or 'greater'")

        # Critical values
        if alternative == 'two-sided':
            critical_value = stats.norm.ppf(1 - alpha/2)
            reject_h0 = abs(z_statistic) > critical_value
        elif alternative == 'greater':
            critical_value = stats.norm.ppf(1 - alpha)
            reject_h0 = z_statistic > critical_value
        elif alternative == 'less':
            critical_value = stats.norm.ppf(alpha)
            reject_h0 = z_statistic < critical_value

        # Confidence interval
        if alternative == 'two-sided':
            margin_error = critical_value * standard_error
            ci_lower = sample_mean - margin_error
            ci_upper = sample_mean + margin_error
        else:
            ci_lower, ci_upper = None, None

        result = {
            'test_type': 'One-sample Z-test',
            'sample_size': n,
            'sample_mean': sample_mean,
            'population_mean_h0': population_mean,
            'population_std': population_std,
            'standard_error': standard_error,
            'z_statistic': z_statistic,
            'p_value': p_value,
            'alpha': alpha,
            'alternative': alternative,
            'critical_value': critical_value,
            'reject_h0': reject_h0,
            'significant': p_value < alpha,
            'confidence_interval': (ci_lower, ci_upper) if ci_lower is not None else None,
            'effect_size': abs(sample_mean - population_mean) / population_std
        }

        return result

    def two_sample_z_test(self, sample1: Union[List[float], np.ndarray, pd.Series],
                         sample2: Union[List[float], np.ndarray, pd.Series],
                         std1: Optional[float] = None,
                         std2: Optional[float] = None,
                         alpha: float = 0.05,
                         alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Perform two-sample Z-test

        Args:
            sample1: First sample data
            sample2: Second sample data
            std1: Known standard deviation for sample 1
            std2: Known standard deviation for sample 2
            alpha: Significance level
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            Dictionary with test results
        """
        if isinstance(sample1, pd.Series):
            sample1 = sample1.dropna().values
        elif isinstance(sample1, list):
            sample1 = np.array(sample1)

        if isinstance(sample2, pd.Series):
            sample2 = sample2.dropna().values
        elif isinstance(sample2, list):
            sample2 = np.array(sample2)

        n1, n2 = len(sample1), len(sample2)
        mean1, mean2 = np.mean(sample1), np.mean(sample2)

        # Use sample stds if population stds not provided
        if std1 is None:
            std1 = np.std(sample1, ddof=1)
        if std2 is None:
            std2 = np.std(sample2, ddof=1)

        # Calculate Z-statistic
        standard_error = np.sqrt((std1**2 / n1) + (std2**2 / n2))
        z_statistic = (mean1 - mean2) / standard_error

        # Calculate p-value
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
        elif alternative == 'greater':
            p_value = 1 - stats.norm.cdf(z_statistic)
        elif alternative == 'less':
            p_value = stats.norm.cdf(z_statistic)

        # Critical values and decision
        if alternative == 'two-sided':
            critical_value = stats.norm.ppf(1 - alpha/2)
            reject_h0 = abs(z_statistic) > critical_value
        elif alternative == 'greater':
            critical_value = stats.norm.ppf(1 - alpha)
            reject_h0 = z_statistic > critical_value
        elif alternative == 'less':
            critical_value = stats.norm.ppf(alpha)
            reject_h0 = z_statistic < critical_value

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
        cohens_d = (mean1 - mean2) / pooled_std

        result = {
            'test_type': 'Two-sample Z-test',
            'sample1_size': n1,
            'sample2_size': n2,
            'sample1_mean': mean1,
            'sample2_mean': mean2,
            'mean_difference': mean1 - mean2,
            'sample1_std': std1,
            'sample2_std': std2,
            'standard_error': standard_error,
            'z_statistic': z_statistic,
            'p_value': p_value,
            'alpha': alpha,
            'alternative': alternative,
            'critical_value': critical_value,
            'reject_h0': reject_h0,
            'significant': p_value < alpha,
            'cohens_d': cohens_d,
            'pooled_std': pooled_std
        }

        return result


class TTester:
    """
    Class for T-tests with significance levels and degrees of freedom
    """

    def __init__(self):
        self.test_results = {}

    def one_sample_t_test(self, sample: Union[List[float], np.ndarray, pd.Series],
                         population_mean: float,
                         alpha: float = 0.05,
                         alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Perform one-sample t-test

        Args:
            sample: Sample data
            population_mean: Hypothesized population mean (H0)
            alpha: Significance level
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            Dictionary with test results
        """
        if isinstance(sample, pd.Series):
            sample = sample.dropna().values
        elif isinstance(sample, list):
            sample = np.array(sample)

        n = len(sample)
        df = n - 1
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)
        standard_error = sample_std / np.sqrt(n)

        # Calculate t-statistic
        t_statistic = (sample_mean - population_mean) / standard_error

        # Calculate p-value
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
        elif alternative == 'greater':
            p_value = 1 - stats.t.cdf(t_statistic, df)
        elif alternative == 'less':
            p_value = stats.t.cdf(t_statistic, df)

        # Critical values
        if alternative == 'two-sided':
            critical_value = stats.t.ppf(1 - alpha/2, df)
            reject_h0 = abs(t_statistic) > critical_value
        elif alternative == 'greater':
            critical_value = stats.t.ppf(1 - alpha, df)
            reject_h0 = t_statistic > critical_value
        elif alternative == 'less':
            critical_value = stats.t.ppf(alpha, df)
            reject_h0 = t_statistic < critical_value

        # Confidence interval
        if alternative == 'two-sided':
            margin_error = critical_value * standard_error
            ci_lower = sample_mean - margin_error
            ci_upper = sample_mean + margin_error
        else:
            ci_lower, ci_upper = None, None

        result = {
            'test_type': 'One-sample t-test',
            'sample_size': n,
            'degrees_of_freedom': df,
            'sample_mean': sample_mean,
            'population_mean_h0': population_mean,
            'sample_std': sample_std,
            'standard_error': standard_error,
            't_statistic': t_statistic,
            'p_value': p_value,
            'alpha': alpha,
            'alternative': alternative,
            'critical_value': critical_value,
            'reject_h0': reject_h0,
            'significant': p_value < alpha,
            'confidence_interval': (ci_lower, ci_upper) if ci_lower is not None else None,
            'effect_size': abs(sample_mean - population_mean) / sample_std
        }

        return result

    def two_sample_t_test(self, sample1: Union[List[float], np.ndarray, pd.Series],
                         sample2: Union[List[float], np.ndarray, pd.Series],
                         equal_variances: bool = True,
                         alpha: float = 0.05,
                         alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Perform two-sample t-test

        Args:
            sample1: First sample data
            sample2: Second sample data
            equal_variances: Assume equal variances (pooled t-test vs Welch's t-test)
            alpha: Significance level
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            Dictionary with test results
        """
        if isinstance(sample1, pd.Series):
            sample1 = sample1.dropna().values
        elif isinstance(sample1, list):
            sample1 = np.array(sample1)

        if isinstance(sample2, pd.Series):
            sample2 = sample2.dropna().values
        elif isinstance(sample2, list):
            sample2 = np.array(sample2)

        n1, n2 = len(sample1), len(sample2)
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
        std1, std2 = np.sqrt(var1), np.sqrt(var2)

        if equal_variances:
            # Pooled t-test
            pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)
            standard_error = np.sqrt(pooled_var * (1/n1 + 1/n2))
            df = n1 + n2 - 2
            test_type = 'Two-sample t-test (equal variances)'
        else:
            # Welch's t-test (unequal variances)
            standard_error = np.sqrt(var1/n1 + var2/n2)
            df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
            test_type = "Welch's t-test (unequal variances)"

        # Calculate t-statistic
        t_statistic = (mean1 - mean2) / standard_error

        # Calculate p-value
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
        elif alternative == 'greater':
            p_value = 1 - stats.t.cdf(t_statistic, df)
        elif alternative == 'less':
            p_value = stats.t.cdf(t_statistic, df)

        # Critical values and decision
        if alternative == 'two-sided':
            critical_value = stats.t.ppf(1 - alpha/2, df)
            reject_h0 = abs(t_statistic) > critical_value
        elif alternative == 'greater':
            critical_value = stats.t.ppf(1 - alpha, df)
            reject_h0 = t_statistic > critical_value
        elif alternative == 'less':
            critical_value = stats.t.ppf(alpha, df)
            reject_h0 = t_statistic < critical_value

        # Effect size (Cohen's d)
        if equal_variances:
            pooled_std = np.sqrt(pooled_var)
        else:
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

        cohens_d = (mean1 - mean2) / pooled_std

        result = {
            'test_type': test_type,
            'sample1_size': n1,
            'sample2_size': n2,
            'degrees_of_freedom': df,
            'sample1_mean': mean1,
            'sample2_mean': mean2,
            'mean_difference': mean1 - mean2,
            'sample1_std': std1,
            'sample2_std': std2,
            'equal_variances': equal_variances,
            'standard_error': standard_error,
            't_statistic': t_statistic,
            'p_value': p_value,
            'alpha': alpha,
            'alternative': alternative,
            'critical_value': critical_value,
            'reject_h0': reject_h0,
            'significant': p_value < alpha,
            'cohens_d': cohens_d,
            'pooled_std': pooled_std
        }

        return result

    def paired_t_test(self, sample1: Union[List[float], np.ndarray, pd.Series],
                     sample2: Union[List[float], np.ndarray, pd.Series],
                     alpha: float = 0.05,
                     alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Perform paired t-test

        Args:
            sample1: First sample data (before)
            sample2: Second sample data (after)
            alpha: Significance level
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            Dictionary with test results
        """
        if isinstance(sample1, pd.Series):
            sample1 = sample1.values
        elif isinstance(sample1, list):
            sample1 = np.array(sample1)

        if isinstance(sample2, pd.Series):
            sample2 = sample2.values
        elif isinstance(sample2, list):
            sample2 = np.array(sample2)

        if len(sample1) != len(sample2):
            raise ValueError("Sample sizes must be equal for paired t-test")

        # Calculate differences
        differences = sample1 - sample2
        n = len(differences)
        df = n - 1
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        standard_error = std_diff / np.sqrt(n)

        # Calculate t-statistic (testing if mean difference = 0)
        t_statistic = mean_diff / standard_error

        # Calculate p-value
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
        elif alternative == 'greater':
            p_value = 1 - stats.t.cdf(t_statistic, df)
        elif alternative == 'less':
            p_value = stats.t.cdf(t_statistic, df)

        # Critical values and decision
        if alternative == 'two-sided':
            critical_value = stats.t.ppf(1 - alpha/2, df)
            reject_h0 = abs(t_statistic) > critical_value
        elif alternative == 'greater':
            critical_value = stats.t.ppf(1 - alpha, df)
            reject_h0 = t_statistic > critical_value
        elif alternative == 'less':
            critical_value = stats.t.ppf(alpha, df)
            reject_h0 = t_statistic < critical_value

        # Confidence interval for mean difference
        if alternative == 'two-sided':
            margin_error = critical_value * standard_error
            ci_lower = mean_diff - margin_error
            ci_upper = mean_diff + margin_error
        else:
            ci_lower, ci_upper = None, None

        result = {
            'test_type': 'Paired t-test',
            'sample_size': n,
            'degrees_of_freedom': df,
            'sample1_mean': np.mean(sample1),
            'sample2_mean': np.mean(sample2),
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'standard_error': standard_error,
            't_statistic': t_statistic,
            'p_value': p_value,
            'alpha': alpha,
            'alternative': alternative,
            'critical_value': critical_value,
            'reject_h0': reject_h0,
            'significant': p_value < alpha,
            'confidence_interval': (ci_lower, ci_upper) if ci_lower is not None else None,
            'effect_size': abs(mean_diff) / std_diff
        }

        return result


class ANOVATester:
    """
    Class for ANOVA tables and F-tests
    """

    def __init__(self):
        self.anova_results = {}

    def one_way_anova(self, groups: List[Union[List[float], np.ndarray, pd.Series]],
                     group_names: Optional[List[str]] = None,
                     alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform one-way ANOVA

        Args:
            groups: List of groups (each group is a list/array of values)
            group_names: Names for each group
            alpha: Significance level

        Returns:
            Dictionary with ANOVA results including ANOVA table
        """
        # Clean and convert groups
        clean_groups = []
        for i, group in enumerate(groups):
            if isinstance(group, pd.Series):
                clean_group = group.dropna().values
            elif isinstance(group, list):
                clean_group = np.array(group)
            else:
                clean_group = np.array(group)
            clean_groups.append(clean_group)

        if group_names is None:
            group_names = [f'Group_{i+1}' for i in range(len(groups))]

        # Calculate basic statistics
        k = len(clean_groups)  # number of groups
        n_total = sum(len(group) for group in clean_groups)
        n_groups = [len(group) for group in clean_groups]

        # Group means and overall mean
        group_means = [np.mean(group) for group in clean_groups]
        overall_mean = np.mean(np.concatenate(clean_groups))

        # Sum of Squares calculations
        # Between groups sum of squares (SSB)
        ssb = sum(n * (mean - overall_mean)**2 for n, mean in zip(n_groups, group_means))

        # Within groups sum of squares (SSW)
        ssw = sum(np.sum((group - np.mean(group))**2) for group in clean_groups)

        # Total sum of squares (SST)
        sst = ssb + ssw

        # Degrees of freedom
        df_between = k - 1
        df_within = n_total - k
        df_total = n_total - 1

        # Mean squares
        msb = ssb / df_between
        msw = ssw / df_within

        # F-statistic
        f_statistic = msb / msw

        # p-value
        p_value = 1 - stats.f.cdf(f_statistic, df_between, df_within)

        # Critical value
        critical_value = stats.f.ppf(1 - alpha, df_between, df_within)

        # Effect size (eta-squared)
        eta_squared = ssb / sst

        # Create ANOVA table
        anova_table = pd.DataFrame({
            'Source': ['Between Groups', 'Within Groups', 'Total'],
            'SS': [ssb, ssw, sst],
            'df': [df_between, df_within, df_total],
            'MS': [msb, msw, sst/df_total],
            'F': [f_statistic, np.nan, np.nan],
            'p-value': [p_value, np.nan, np.nan]
        })

        # Group statistics
        group_stats = pd.DataFrame({
            'Group': group_names,
            'N': n_groups,
            'Mean': group_means,
            'Std': [np.std(group, ddof=1) for group in clean_groups],
            'Var': [np.var(group, ddof=1) for group in clean_groups]
        })

        result = {
            'test_type': 'One-way ANOVA',
            'anova_table': anova_table,
            'group_statistics': group_stats,
            'f_statistic': f_statistic,
            'p_value': p_value,
            'alpha': alpha,
            'critical_value': critical_value,
            'reject_h0': p_value < alpha,
            'significant': p_value < alpha,
            'effect_size_eta_squared': eta_squared,
            'degrees_of_freedom': (df_between, df_within),
            'sum_of_squares': {'between': ssb, 'within': ssw, 'total': sst},
            'mean_squares': {'between': msb, 'within': msw}
        }

        return result

    def two_way_anova(self, data: pd.DataFrame, dependent_var: str,
                     factor1: str, factor2: str,
                     alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform two-way ANOVA

        Args:
            data: DataFrame containing the data
            dependent_var: Name of dependent variable column
            factor1: Name of first factor column
            factor2: Name of second factor column
            alpha: Significance level

        Returns:
            Dictionary with two-way ANOVA results
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Statsmodels required for two-way ANOVA")

        # Clean data
        clean_data = data[[dependent_var, factor1, factor2]].dropna()

        # Create formula for statsmodels
        formula = f'{dependent_var} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})'

        # Fit the model
        model = sm.formula.ols(formula, data=clean_data).fit()

        # Perform ANOVA
        anova_results = sm.stats.anova_lm(model, typ=2)

        # Extract results
        f_statistics = anova_results['F'].values
        p_values = anova_results['PR(>F)'].values

        result = {
            'test_type': 'Two-way ANOVA',
            'anova_table': anova_results,
            'model_summary': model.summary(),
            'main_effect_factor1': {
                'f_statistic': f_statistics[0],
                'p_value': p_values[0],
                'significant': p_values[0] < alpha
            },
            'main_effect_factor2': {
                'f_statistic': f_statistics[1],
                'p_value': p_values[1],
                'significant': p_values[1] < alpha
            },
            'interaction_effect': {
                'f_statistic': f_statistics[2],
                'p_value': p_values[2],
                'significant': p_values[2] < alpha
            },
            'alpha': alpha,
            'r_squared': model.rsquared,
            'adjusted_r_squared': model.rsquared_adj
        }

        return result


class FTester:
    """
    Class for joint hypothesis testing using F-tests
    """

    def __init__(self):
        self.f_test_results = {}

    def joint_hypothesis_f_test(self, restricted_model_ssr: float,
                               unrestricted_model_ssr: float,
                               num_restrictions: int,
                               degrees_of_freedom: int,
                               alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform joint hypothesis F-test

        Args:
            restricted_model_ssr: Sum of squared residuals from restricted model
            unrestricted_model_ssr: Sum of squared residuals from unrestricted model
            num_restrictions: Number of restrictions (q)
            degrees_of_freedom: Degrees of freedom for unrestricted model (n-k)
            alpha: Significance level

        Returns:
            Dictionary with F-test results
        """
        # Calculate F-statistic
        numerator = (restricted_model_ssr - unrestricted_model_ssr) / num_restrictions
        denominator = unrestricted_model_ssr / degrees_of_freedom
        f_statistic = numerator / denominator

        # Degrees of freedom
        df_numerator = num_restrictions
        df_denominator = degrees_of_freedom

        # Calculate p-value
        p_value = 1 - stats.f.cdf(f_statistic, df_numerator, df_denominator)

        # Critical value
        critical_value = stats.f.ppf(1 - alpha, df_numerator, df_denominator)

        result = {
            'test_type': 'Joint Hypothesis F-test',
            'f_statistic': f_statistic,
            'p_value': p_value,
            'alpha': alpha,
            'critical_value': critical_value,
            'degrees_of_freedom': (df_numerator, df_denominator),
            'reject_h0': p_value < alpha,
            'significant': p_value < alpha,
            'restricted_ssr': restricted_model_ssr,
            'unrestricted_ssr': unrestricted_model_ssr,
            'num_restrictions': num_restrictions
        }

        return result

    def regression_f_test(self, y: np.ndarray, X_unrestricted: np.ndarray,
                         X_restricted: np.ndarray,
                         alpha: float = 0.05) -> Dict[str, Any]:
        """
        F-test for comparing nested regression models

        Args:
            y: Dependent variable
            X_unrestricted: Design matrix for unrestricted model
            X_restricted: Design matrix for restricted model
            alpha: Significance level

        Returns:
            Dictionary with F-test results
        """
        # Fit both models using OLS
        # Unrestricted model
        beta_unr = np.linalg.lstsq(X_unrestricted, y, rcond=None)[0]
        y_pred_unr = X_unrestricted @ beta_unr
        ssr_unrestricted = np.sum((y - y_pred_unr)**2)

        # Restricted model
        beta_res = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
        y_pred_res = X_restricted @ beta_res
        ssr_restricted = np.sum((y - y_pred_res)**2)

        # Parameters
        n = len(y)
        k_unrestricted = X_unrestricted.shape[1]
        k_restricted = X_restricted.shape[1]
        num_restrictions = k_unrestricted - k_restricted
        df = n - k_unrestricted

        # Perform F-test
        f_test_result = self.joint_hypothesis_f_test(
            ssr_restricted, ssr_unrestricted, num_restrictions, df, alpha
        )

        # Add additional information
        f_test_result.update({
            'n_observations': n,
            'k_unrestricted': k_unrestricted,
            'k_restricted': k_restricted,
            'r_squared_unrestricted': 1 - ssr_unrestricted / np.sum((y - np.mean(y))**2),
            'r_squared_restricted': 1 - ssr_restricted / np.sum((y - np.mean(y))**2)
        })

        return f_test_result


class ChiSquaredTester:
    """
    Class for Chi-squared tests and sample size calculations
    """

    def __init__(self):
        self.chi2_results = {}

    def chi2_independence_test(self, contingency_table: Union[pd.DataFrame, np.ndarray],
                              alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform Chi-squared test of independence

        Args:
            contingency_table: Contingency table (observed frequencies)
            alpha: Significance level

        Returns:
            Dictionary with test results
        """
        if isinstance(contingency_table, pd.DataFrame):
            observed = contingency_table.values
            row_labels = contingency_table.index.tolist()
            col_labels = contingency_table.columns.tolist()
        else:
            observed = np.array(contingency_table)
            row_labels = [f'Row_{i+1}' for i in range(observed.shape[0])]
            col_labels = [f'Col_{i+1}' for i in range(observed.shape[1])]

        # Perform chi-squared test
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)

        # Critical value
        critical_value = stats.chi2.ppf(1 - alpha, dof)

        # Effect size measures
        n = np.sum(observed)
        cramers_v = np.sqrt(chi2_stat / (n * (min(observed.shape) - 1)))

        # Standardized residuals
        std_residuals = (observed - expected) / np.sqrt(expected)

        result = {
            'test_type': 'Chi-squared test of independence',
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'alpha': alpha,
            'critical_value': critical_value,
            'reject_h0': p_value < alpha,
            'significant': p_value < alpha,
            'observed_frequencies': pd.DataFrame(observed, index=row_labels, columns=col_labels),
            'expected_frequencies': pd.DataFrame(expected, index=row_labels, columns=col_labels),
            'standardized_residuals': pd.DataFrame(std_residuals, index=row_labels, columns=col_labels),
            'cramers_v': cramers_v,
            'sample_size': n,
            'effect_size_interpretation': self._interpret_cramers_v(cramers_v)
        }

        return result

    def chi2_goodness_of_fit(self, observed: Union[List[float], np.ndarray],
                            expected: Union[List[float], np.ndarray],
                            alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform Chi-squared goodness of fit test

        Args:
            observed: Observed frequencies
            expected: Expected frequencies
            alpha: Significance level

        Returns:
            Dictionary with test results
        """
        observed = np.array(observed)
        expected = np.array(expected)

        if len(observed) != len(expected):
            raise ValueError("Observed and expected arrays must have same length")

        # Calculate chi-squared statistic
        chi2_stat = np.sum((observed - expected)**2 / expected)

        # Degrees of freedom
        dof = len(observed) - 1

        # p-value
        p_value = 1 - stats.chi2.cdf(chi2_stat, dof)

        # Critical value
        critical_value = stats.chi2.ppf(1 - alpha, dof)

        result = {
            'test_type': 'Chi-squared goodness of fit test',
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'alpha': alpha,
            'critical_value': critical_value,
            'reject_h0': p_value < alpha,
            'significant': p_value < alpha,
            'observed_frequencies': observed,
            'expected_frequencies': expected,
            'residuals': observed - expected,
            'standardized_residuals': (observed - expected) / np.sqrt(expected)
        }

        return result

    def calculate_chi2_sample_size(self, effect_size: float,
                                  alpha: float = 0.05,
                                  power: float = 0.80,
                                  df: Optional[int] = None,
                                  contingency_table_shape: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Calculate required sample size for Chi-squared test

        Args:
            effect_size: Effect size (w for independence test, or custom)
            alpha: Significance level
            power: Desired statistical power
            df: Degrees of freedom (if known)
            contingency_table_shape: Shape of contingency table (rows, cols)

        Returns:
            Dictionary with sample size calculation results
        """
        # Calculate degrees of freedom if not provided
        if df is None and contingency_table_shape is not None:
            df = (contingency_table_shape[0] - 1) * (contingency_table_shape[1] - 1)
        elif df is None:
            raise ValueError("Either df or contingency_table_shape must be provided")

        # Critical values
        critical_value = stats.chi2.ppf(1 - alpha, df)

        # Non-centrality parameter
        ncp = effect_size**2  # For effect size w

        # Calculate sample size iteratively
        def power_function(n):
            ncp_actual = n * effect_size**2
            # Power = P(Chi2(df, ncp) > critical_value)
            return 1 - stats.ncx2.cdf(critical_value, df, ncp_actual)

        # Binary search for sample size
        n_low, n_high = 10, 10000
        while n_high - n_low > 1:
            n_mid = (n_low + n_high) // 2
            if power_function(n_mid) < power:
                n_low = n_mid
            else:
                n_high = n_mid

        required_n = n_high

        # Effect size interpretation
        effect_size_interp = self._interpret_effect_size_w(effect_size)

        result = {
            'test_type': 'Chi-squared sample size calculation',
            'required_sample_size': required_n,
            'effect_size': effect_size,
            'effect_size_interpretation': effect_size_interp,
            'alpha': alpha,
            'power': power,
            'degrees_of_freedom': df,
            'achieved_power': power_function(required_n),
            'critical_value': critical_value,
            'contingency_table_shape': contingency_table_shape
        }

        return result

    def _interpret_cramers_v(self, cramers_v: float) -> str:
        """Interpret Cramer's V effect size"""
        if cramers_v < 0.1:
            return "Negligible association"
        elif cramers_v < 0.3:
            return "Weak association"
        elif cramers_v < 0.5:
            return "Moderate association"
        else:
            return "Strong association"

    def _interpret_effect_size_w(self, w: float) -> str:
        """Interpret Cohen's w effect size"""
        if w < 0.1:
            return "Small effect"
        elif w < 0.3:
            return "Medium effect"
        elif w < 0.5:
            return "Large effect"
        else:
            return "Very large effect"


class HypothesisTestingSuite:
    """
    Comprehensive hypothesis testing suite combining all testing methods
    """

    def __init__(self):
        self.normal_tester = NormalDistributionTester()
        self.z_tester = ZTester()
        self.t_tester = TTester()
        self.anova_tester = ANOVATester()
        self.f_tester = FTester()
        self.chi2_tester = ChiSquaredTester()
        self.all_results = {}

    def comprehensive_test_report(self, data: pd.DataFrame,
                                 test_types: List[str],
                                 **kwargs) -> Dict[str, Any]:
        """
        Generate comprehensive hypothesis testing report

        Args:
            data: DataFrame with data for testing
            test_types: List of tests to perform
            **kwargs: Additional parameters for specific tests

        Returns:
            Dictionary with all test results
        """
        report = {
            'data_info': {
                'shape': data.shape,
                'columns': data.columns.tolist(),
                'missing_values': data.isnull().sum().to_dict()
            },
            'test_results': {}
        }

        # Available tests
        available_tests = {
            'normality': self._test_normality,
            'one_sample_t': self._test_one_sample_t,
            'two_sample_t': self._test_two_sample_t,
            'paired_t': self._test_paired_t,
            'anova': self._test_anova,
            'chi2_independence': self._test_chi2_independence
        }

        # Perform requested tests
        for test_type in test_types:
            if test_type in available_tests:
                try:
                    test_result = available_tests[test_type](data, **kwargs)
                    report['test_results'][test_type] = test_result
                except Exception as e:
                    report['test_results'][test_type] = {'error': str(e)}

        # Store comprehensive results
        self.all_results = report
        return report

    def _test_normality(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Test normality for numeric columns"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        results = {}

        for col in numeric_cols:
            sample_data = data[col].dropna()
            if len(sample_data) > 3:
                # Shapiro-Wilk test
                shapiro_stat, shapiro_p = stats.shapiro(sample_data[:5000])  # Limit for efficiency
                results[col] = {
                    'shapiro_wilk_statistic': shapiro_stat,
                    'shapiro_wilk_p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                }

        return results

    def _test_one_sample_t(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Perform one-sample t-tests"""
        column = kwargs.get('column')
        population_mean = kwargs.get('population_mean', 0)
        alpha = kwargs.get('alpha', 0.05)

        if column and column in data.columns:
            return self.t_tester.one_sample_t_test(data[column], population_mean, alpha)
        else:
            return {'error': 'Column not specified or not found'}

    def _test_two_sample_t(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Perform two-sample t-tests"""
        col1 = kwargs.get('column1')
        col2 = kwargs.get('column2')
        alpha = kwargs.get('alpha', 0.05)

        if col1 and col2 and col1 in data.columns and col2 in data.columns:
            return self.t_tester.two_sample_t_test(data[col1], data[col2], alpha=alpha)
        else:
            return {'error': 'Columns not specified or not found'}

    def _test_paired_t(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Perform paired t-tests"""
        col1 = kwargs.get('column1')
        col2 = kwargs.get('column2')
        alpha = kwargs.get('alpha', 0.05)

        if col1 and col2 and col1 in data.columns and col2 in data.columns:
            return self.t_tester.paired_t_test(data[col1], data[col2], alpha)
        else:
            return {'error': 'Columns not specified or not found'}

    def _test_anova(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Perform ANOVA"""
        group_column = kwargs.get('group_column')
        value_column = kwargs.get('value_column')

        if group_column and value_column and both in data.columns:
            groups = [group.values for name, group in data.groupby(group_column)[value_column]]
            group_names = [str(name) for name in data[group_column].unique()]
            return self.anova_tester.one_way_anova(groups, group_names)
        else:
            return {'error': 'Group or value column not specified or not found'}

    def _test_chi2_independence(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Perform Chi-squared test of independence"""
        col1 = kwargs.get('column1')
        col2 = kwargs.get('column2')

        if col1 and col2 and col1 in data.columns and col2 in data.columns:
            contingency_table = pd.crosstab(data[col1], data[col2])
            return self.chi2_tester.chi2_independence_test(contingency_table)
        else:
            return {'error': 'Columns not specified or not found'}


# Utility functions
def quick_t_test(sample1: Union[List[float], np.ndarray, pd.Series],
                sample2: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
                population_mean: float = 0,
                alpha: float = 0.05) -> Dict[str, Any]:
    """
    Quick t-test function for common use cases

    Args:
        sample1: First sample or single sample
        sample2: Second sample (if None, performs one-sample test)
        population_mean: Population mean for one-sample test
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    tester = TTester()

    if sample2 is None:
        # One-sample t-test
        return tester.one_sample_t_test(sample1, population_mean, alpha)
    else:
        # Two-sample t-test
        return tester.two_sample_t_test(sample1, sample2, alpha=alpha)


if __name__ == "__main__":
    print("Hypothesis Testing Module")
    print("=========================")

    # Create sample data for demonstration
    np.random.seed(42)

    # Sample data for various tests
    normal_sample = np.random.normal(100, 15, 50)
    sample_a = np.random.normal(20, 5, 30)
    sample_b = np.random.normal(22, 4, 35)

    print("Sample data created for demonstration")

    # Quick t-test example
    print("\n=== Quick T-test Example ===")
    t_result = quick_t_test(sample_a, sample_b)
    print(f"Two-sample t-test p-value: {t_result['p_value']:.4f}")
    print(f"Significant difference: {t_result['significant']}")

    # Normal distribution CDF example
    print("\n=== Normal Distribution CDF ===")
    normal_tester = NormalDistributionTester()
    cdf_value = normal_tester.compute_normal_cdf(1.96)
    print(f"P(Z ≤ 1.96) = {cdf_value:.4f}")

    # ANOVA example
    print("\n=== ANOVA Example ===")
    anova_tester = ANOVATester()
    group1 = np.random.normal(20, 3, 15)
    group2 = np.random.normal(22, 3, 15)
    group3 = np.random.normal(25, 3, 15)

    anova_result = anova_tester.one_way_anova([group1, group2, group3], ['A', 'B', 'C'])
    print(f"ANOVA F-statistic: {anova_result['f_statistic']:.4f}")
    print(f"ANOVA p-value: {anova_result['p_value']:.4f}")

    # Chi-squared test example
    print("\n=== Chi-squared Test Example ===")
    chi2_tester = ChiSquaredTester()

    # Create sample contingency table
    contingency = np.array([[20, 30, 25],
                           [15, 35, 30],
                           [25, 20, 35]])

    chi2_result = chi2_tester.chi2_independence_test(contingency)
    print(f"Chi-squared statistic: {chi2_result['chi2_statistic']:.4f}")
    print(f"Chi-squared p-value: {chi2_result['p_value']:.4f}")

    print("\nHypothesis Testing module ready for use!")