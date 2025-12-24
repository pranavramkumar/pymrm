"""
Sample Statistics Module for Financial and Economic Data

This module provides comprehensive sample statistics computation and visualization for:
- Descriptive statistics (central tendency, dispersion, shape)
- Risk measures (variance, semi-variance, drawdowns)
- Correlation and covariance analysis
- Financial returns analysis (percentage, log, basket, index returns)
- Advanced statistical measures with visualizations

Dependencies: pandas, numpy, matplotlib, seaborn, scipy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Optional, Dict, Any, List, Union, Tuple
from scipy import stats
import itertools

# Optional imports for enhanced functionality
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Some interactive features will be disabled.")


class SampleStatisticsCalculator:
    """
    Main class for computing comprehensive sample statistics
    """

    def __init__(self):
        self.results = {}
        self.cached_stats = {}

    def compute_basic_statistics(self, data: Union[pd.Series, np.ndarray, pd.DataFrame],
                                column: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute basic descriptive statistics

        Args:
            data: Data vector, series, or DataFrame
            column: Column name if data is DataFrame

        Returns:
            Dictionary with basic statistics
        """
        # Extract series if DataFrame
        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError("Column name required for DataFrame input")
            series = data[column].dropna()
        else:
            series = pd.Series(data).dropna() if isinstance(data, np.ndarray) else data.dropna()

        values = series.values

        # Basic statistics
        basic_stats = {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values, ddof=1),
            'variance': np.var(values, ddof=1),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values),
            'median': np.median(values),
            'mode': stats.mode(values)[0] if len(values) > 0 else np.nan
        }

        # Quartiles and percentiles
        quartiles = {
            'q1': np.percentile(values, 25),
            'q2': np.percentile(values, 50),  # Same as median
            'q3': np.percentile(values, 75),
            'iqr': np.percentile(values, 75) - np.percentile(values, 25)
        }

        # Common percentiles
        percentiles = {}
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            percentiles[f'p{p}'] = np.percentile(values, p)

        # Shape statistics
        shape_stats = {
            'skewness': stats.skew(values),
            'kurtosis': stats.kurtosis(values),
            'excess_kurtosis': stats.kurtosis(values),  # scipy returns excess kurtosis
            'fisher_pearson_skewness': stats.skew(values),
            'jarque_bera_stat': self._jarque_bera_test(values)[0],
            'jarque_bera_pvalue': self._jarque_bera_test(values)[1]
        }

        result = {
            'basic': basic_stats,
            'quartiles': quartiles,
            'percentiles': percentiles,
            'shape': shape_stats,
            'column_name': column if column else 'data'
        }

        # Cache results
        cache_key = f"{column if column else 'data'}_basic"
        self.cached_stats[cache_key] = result

        return result

    def compute_variance_measures(self, data: Union[pd.Series, np.ndarray, pd.DataFrame],
                                column: Optional[str] = None,
                                trim_lower: float = 0.0,
                                trim_upper: float = 0.0) -> Dict[str, Any]:
        """
        Compute various variance measures including semi-variance and trimmed variance

        Args:
            data: Data vector, series, or DataFrame
            column: Column name if data is DataFrame
            trim_lower: Lower trimming percentage (0-100)
            trim_upper: Upper trimming percentage (0-100)

        Returns:
            Dictionary with variance measures
        """
        # Extract series if DataFrame
        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError("Column name required for DataFrame input")
            series = data[column].dropna()
        else:
            series = pd.Series(data).dropna() if isinstance(data, np.ndarray) else data.dropna()

        values = series.values
        mean_val = np.mean(values)

        # Standard variance
        variance = np.var(values, ddof=1)
        std_dev = np.std(values, ddof=1)

        # Semi-variance (downside variance)
        downside_values = values[values < mean_val]
        semi_variance = np.var(downside_values, ddof=1) if len(downside_values) > 1 else 0
        downside_std = np.sqrt(semi_variance) if semi_variance > 0 else 0

        # Upside variance
        upside_values = values[values > mean_val]
        upside_variance = np.var(upside_values, ddof=1) if len(upside_values) > 1 else 0
        upside_std = np.sqrt(upside_variance) if upside_variance > 0 else 0

        # Trimmed variance
        if trim_lower > 0 or trim_upper > 0:
            lower_percentile = trim_lower
            upper_percentile = 100 - trim_upper
            trimmed_values = values[
                (values >= np.percentile(values, lower_percentile)) &
                (values <= np.percentile(values, upper_percentile))
            ]
            trimmed_variance = np.var(trimmed_values, ddof=1) if len(trimmed_values) > 1 else 0
            trimmed_std = np.sqrt(trimmed_variance) if trimmed_variance > 0 else 0
        else:
            trimmed_variance = variance
            trimmed_std = std_dev

        # Mean absolute deviation
        mad = np.mean(np.abs(values - mean_val))

        # Median absolute deviation
        median_val = np.median(values)
        median_mad = np.median(np.abs(values - median_val))

        result = {
            'variance': variance,
            'standard_deviation': std_dev,
            'semi_variance': semi_variance,
            'downside_deviation': downside_std,
            'upside_variance': upside_variance,
            'upside_deviation': upside_std,
            'trimmed_variance': trimmed_variance,
            'trimmed_std': trimmed_std,
            'trimming_lower': trim_lower,
            'trimming_upper': trim_upper,
            'mean_absolute_deviation': mad,
            'median_absolute_deviation': median_mad,
            'coefficient_of_variation': std_dev / abs(mean_val) if mean_val != 0 else np.inf
        }

        return result

    def compute_quantiles(self, data: Union[pd.Series, np.ndarray, pd.DataFrame],
                         column: Optional[str] = None,
                         quantiles: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Compute custom quantiles and percentiles

        Args:
            data: Data vector, series, or DataFrame
            column: Column name if data is DataFrame
            quantiles: List of quantiles (0-1) to compute

        Returns:
            Dictionary with quantile results
        """
        # Extract series if DataFrame
        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError("Column name required for DataFrame input")
            series = data[column].dropna()
        else:
            series = pd.Series(data).dropna() if isinstance(data, np.ndarray) else data.dropna()

        values = series.values

        if quantiles is None:
            quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

        # Compute quantiles
        quantile_results = {}
        for q in quantiles:
            quantile_results[f'q{q:.3f}'] = np.quantile(values, q)

        # Additional quantile statistics
        result = {
            'quantiles': quantile_results,
            'quantile_list': quantiles,
            'interquartile_range': np.quantile(values, 0.75) - np.quantile(values, 0.25),
            'interdecile_range': np.quantile(values, 0.9) - np.quantile(values, 0.1),
            'range_90_10': np.quantile(values, 0.9) - np.quantile(values, 0.1),
            'range_95_5': np.quantile(values, 0.95) - np.quantile(values, 0.05),
            'range_99_1': np.quantile(values, 0.99) - np.quantile(values, 0.01)
        }

        return result

    def compute_correlation_matrix(self, data: pd.DataFrame,
                                 columns: Optional[List[str]] = None,
                                 method: str = 'pearson') -> Dict[str, Any]:
        """
        Compute correlation matrix with multiple methods

        Args:
            data: DataFrame with numeric data
            columns: List of columns to include
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            Dictionary with correlation results
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        df_subset = data[columns].dropna()

        # Compute correlations
        if method == 'pearson':
            corr_matrix = df_subset.corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = df_subset.corr(method='spearman')
        elif method == 'kendall':
            corr_matrix = df_subset.corr(method='kendall')
        else:
            raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")

        # Extract correlation pairs
        correlation_pairs = []
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i < j:  # Only upper triangle
                    correlation_pairs.append({
                        'variable1': col1,
                        'variable2': col2,
                        'correlation': corr_matrix.loc[col1, col2],
                        'abs_correlation': abs(corr_matrix.loc[col1, col2])
                    })

        # Sort by absolute correlation
        correlation_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)

        # Concordant correlation (using Kendall's tau)
        if method == 'kendall':
            concordant_pairs = []
            for pair in correlation_pairs:
                var1, var2 = pair['variable1'], pair['variable2']
                tau_stat, tau_p = stats.kendalltau(df_subset[var1], df_subset[var2])
                concordant_pairs.append({
                    'variable1': var1,
                    'variable2': var2,
                    'kendall_tau': tau_stat,
                    'p_value': tau_p,
                    'concordant': tau_stat > 0
                })
        else:
            concordant_pairs = None

        result = {
            'correlation_matrix': corr_matrix,
            'method': method,
            'correlation_pairs': correlation_pairs,
            'concordant_pairs': concordant_pairs,
            'strongest_correlation': correlation_pairs[0] if correlation_pairs else None,
            'mean_correlation': np.mean([pair['abs_correlation'] for pair in correlation_pairs]),
            'columns': columns
        }

        return result

    def compute_covariance_matrix(self, data: pd.DataFrame,
                                columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compute covariance matrix

        Args:
            data: DataFrame with numeric data
            columns: List of columns to include

        Returns:
            Dictionary with covariance results
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        df_subset = data[columns].dropna()

        # Compute covariance matrix
        cov_matrix = df_subset.cov()

        # Extract covariance pairs
        covariance_pairs = []
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i < j:  # Only upper triangle
                    covariance_pairs.append({
                        'variable1': col1,
                        'variable2': col2,
                        'covariance': cov_matrix.loc[col1, col2],
                        'abs_covariance': abs(cov_matrix.loc[col1, col2])
                    })

        # Sort by absolute covariance
        covariance_pairs.sort(key=lambda x: x['abs_covariance'], reverse=True)

        result = {
            'covariance_matrix': cov_matrix,
            'covariance_pairs': covariance_pairs,
            'strongest_covariance': covariance_pairs[0] if covariance_pairs else None,
            'columns': columns
        }

        return result

    def compute_pairwise_correlations(self, data: pd.DataFrame,
                                    columns: Optional[List[str]] = None,
                                    methods: List[str] = ['pearson', 'spearman', 'kendall']) -> Dict[str, Any]:
        """
        Compute pairwise correlations using multiple methods

        Args:
            data: DataFrame with numeric data
            columns: List of columns to include
            methods: List of correlation methods to compute

        Returns:
            Dictionary with pairwise correlation results
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        df_subset = data[columns].dropna()
        results = {}

        for method in methods:
            corr_result = self.compute_correlation_matrix(data, columns, method)
            results[method] = corr_result

        # Compare methods
        if len(methods) > 1:
            comparison_data = []
            for i, col1 in enumerate(columns):
                for j, col2 in enumerate(columns):
                    if i < j:
                        row = {'variable1': col1, 'variable2': col2}
                        for method in methods:
                            row[f'{method}_correlation'] = results[method]['correlation_matrix'].loc[col1, col2]
                        comparison_data.append(row)

            results['comparison'] = pd.DataFrame(comparison_data)

        return results

    def _jarque_bera_test(self, values: np.ndarray) -> Tuple[float, float]:
        """Compute Jarque-Bera test statistic and p-value"""
        try:
            from scipy.stats import jarque_bera
            return jarque_bera(values)
        except ImportError:
            # Fallback implementation
            n = len(values)
            skewness = stats.skew(values)
            kurtosis_val = stats.kurtosis(values)
            jb_stat = n * (skewness**2 / 6 + kurtosis_val**2 / 24)
            p_value = 1 - stats.chi2.cdf(jb_stat, df=2)
            return jb_stat, p_value


class FinancialReturnsCalculator:
    """
    Class for computing various types of financial returns
    """

    def __init__(self):
        self.returns_cache = {}

    def compute_percentage_returns(self, data: Union[pd.Series, pd.DataFrame],
                                 column: Optional[str] = None,
                                 periods: int = 1) -> Union[pd.Series, pd.DataFrame]:
        """
        Compute percentage returns

        Args:
            data: Price data (Series or DataFrame)
            column: Column name if DataFrame
            periods: Number of periods for return calculation

        Returns:
            Percentage returns
        """
        if isinstance(data, pd.DataFrame):
            if column:
                prices = data[column]
            else:
                return data.pct_change(periods=periods) * 100
        else:
            prices = data

        returns = prices.pct_change(periods=periods) * 100
        return returns.dropna()

    def compute_log_returns(self, data: Union[pd.Series, pd.DataFrame],
                          column: Optional[str] = None,
                          periods: int = 1) -> Union[pd.Series, pd.DataFrame]:
        """
        Compute logarithmic returns

        Args:
            data: Price data (Series or DataFrame)
            column: Column name if DataFrame
            periods: Number of periods for return calculation

        Returns:
            Log returns
        """
        if isinstance(data, pd.DataFrame):
            if column:
                prices = data[column]
            else:
                return np.log(data / data.shift(periods)) * 100
        else:
            prices = data

        log_returns = np.log(prices / prices.shift(periods)) * 100
        return log_returns.dropna()

    def compute_drawdown(self, data: Union[pd.Series, pd.DataFrame],
                        column: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute drawdown statistics

        Args:
            data: Price or cumulative return data
            column: Column name if DataFrame

        Returns:
            Dictionary with drawdown statistics
        """
        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError("Column name required for DataFrame input")
            prices = data[column]
        else:
            prices = data

        # Calculate cumulative returns (assuming prices)
        cumulative_returns = prices / prices.iloc[0]

        # Calculate running maximum
        running_max = cumulative_returns.expanding().max()

        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max * 100

        # Drawdown statistics
        max_drawdown = drawdown.min()
        max_drawdown_idx = drawdown.idxmin()

        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_date = None

        for date, dd in drawdown.items():
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_date = date
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                if start_date:
                    drawdown_periods.append({
                        'start': start_date,
                        'end': date,
                        'duration': (date - start_date).days if hasattr(date, 'days') else len(drawdown.loc[start_date:date]) - 1,
                        'max_drawdown': drawdown.loc[start_date:date].min()
                    })

        result = {
            'drawdown_series': drawdown,
            'max_drawdown': max_drawdown,
            'max_drawdown_date': max_drawdown_idx,
            'average_drawdown': drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0,
            'drawdown_periods': drawdown_periods,
            'current_drawdown': drawdown.iloc[-1],
            'recovery_factor': abs(cumulative_returns.iloc[-1] - 1) / abs(max_drawdown / 100) if max_drawdown != 0 else np.inf
        }

        return result

    def compute_leveraged_returns(self, returns: Union[pd.Series, pd.DataFrame],
                                leverage: float = 2.0,
                                column: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
        """
        Compute leveraged returns

        Args:
            returns: Return data (Series or DataFrame)
            leverage: Leverage factor
            column: Column name if DataFrame

        Returns:
            Leveraged returns
        """
        if isinstance(returns, pd.DataFrame):
            if column:
                return returns[column] * leverage
            else:
                return returns * leverage
        else:
            return returns * leverage

    def compute_basket_return(self, data: pd.DataFrame,
                            weights: Optional[Dict[str, float]] = None,
                            columns: Optional[List[str]] = None) -> pd.Series:
        """
        Compute basket (portfolio) returns

        Args:
            data: DataFrame with return data
            weights: Dictionary of weights for each asset
            columns: List of columns to include

        Returns:
            Basket return series
        """
        if columns is None:
            columns = data.columns.tolist()

        if weights is None:
            # Equal weights
            weights = {col: 1/len(columns) for col in columns}

        # Ensure weights sum to 1
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            weights = {k: v/total_weight for k, v in weights.items()}

        # Compute weighted returns
        basket_returns = pd.Series(0.0, index=data.index)
        for col in columns:
            if col in weights and col in data.columns:
                basket_returns += data[col] * weights[col]

        return basket_returns

    def compute_index_return(self, data: pd.DataFrame,
                           base_value: float = 100.0,
                           columns: Optional[List[str]] = None) -> pd.Series:
        """
        Compute index return (cumulative performance)

        Args:
            data: DataFrame with return data
            base_value: Starting index value
            columns: List of columns to include

        Returns:
            Index value series
        """
        if columns is None:
            columns = data.columns.tolist()

        # Calculate equal-weighted basket returns
        basket_returns = self.compute_basket_return(data, columns=columns)

        # Convert to index values
        index_values = (1 + basket_returns / 100).cumprod() * base_value

        return index_values


class StatisticsVisualizer:
    """
    Class for creating visualizations of sample statistics
    """

    def __init__(self):
        self.figures = {}

    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                               title: str = "Correlation Matrix",
                               figsize: Tuple[int, int] = (10, 8),
                               annot: bool = True) -> plt.Figure:
        """
        Create correlation matrix heatmap

        Args:
            correlation_matrix: Correlation matrix DataFrame
            title: Plot title
            figsize: Figure size
            annot: Whether to annotate cells with values

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        sns.heatmap(correlation_matrix,
                   annot=annot,
                   cmap='RdBu_r',
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8},
                   ax=ax)

        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig

    def plot_covariance_heatmap(self, covariance_matrix: pd.DataFrame,
                              title: str = "Covariance Matrix",
                              figsize: Tuple[int, int] = (10, 8),
                              annot: bool = True) -> plt.Figure:
        """
        Create covariance matrix heatmap

        Args:
            covariance_matrix: Covariance matrix DataFrame
            title: Plot title
            figsize: Figure size
            annot: Whether to annotate cells with values

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        sns.heatmap(covariance_matrix,
                   annot=annot,
                   cmap='viridis',
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8},
                   ax=ax)

        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig

    def plot_statistics_summary(self, stats_dict: Dict[str, Any],
                              title: str = "Statistics Summary",
                              figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Create comprehensive statistics summary plot

        Args:
            stats_dict: Dictionary with statistics
            title: Plot title
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Basic statistics bar plot
        basic_stats = stats_dict.get('basic', {})
        if basic_stats:
            metrics = ['mean', 'std', 'min', 'max', 'skewness', 'kurtosis']
            values = [basic_stats.get(metric, 0) for metric in metrics]

            axes[0, 0].bar(metrics, values)
            axes[0, 0].set_title('Basic Statistics')
            axes[0, 0].tick_params(axis='x', rotation=45)

        # Percentiles plot
        percentiles = stats_dict.get('percentiles', {})
        if percentiles:
            perc_keys = sorted([k for k in percentiles.keys() if k.startswith('p')])
            perc_values = [percentiles[k] for k in perc_keys]
            perc_labels = [k[1:] + '%' for k in perc_keys]

            axes[0, 1].plot(perc_labels, perc_values, marker='o')
            axes[0, 1].set_title('Percentiles')
            axes[0, 1].tick_params(axis='x', rotation=45)

        # Quartiles box-like plot
        quartiles = stats_dict.get('quartiles', {})
        if quartiles:
            q_values = [quartiles.get(f'q{i}', 0) for i in [1, 2, 3]]
            axes[0, 2].bar(['Q1', 'Q2 (Median)', 'Q3'], q_values)
            axes[0, 2].set_title('Quartiles')

        # Variance measures
        variance_data = stats_dict.get('variance', {})
        if variance_data:
            var_metrics = ['variance', 'semi_variance', 'trimmed_variance']
            var_values = [variance_data.get(metric, 0) for metric in var_metrics]

            axes[1, 0].bar(['Total Var', 'Semi Var', 'Trimmed Var'], var_values)
            axes[1, 0].set_title('Variance Measures')

        # Risk measures (if available)
        if 'drawdown' in stats_dict:
            dd_stats = stats_dict['drawdown']
            dd_metrics = ['max_drawdown', 'average_drawdown', 'current_drawdown']
            dd_values = [dd_stats.get(metric, 0) for metric in dd_metrics]

            axes[1, 1].bar(['Max DD', 'Avg DD', 'Current DD'], dd_values)
            axes[1, 1].set_title('Drawdown Statistics')

        # Distribution shape
        shape_data = stats_dict.get('shape', {})
        if shape_data:
            axes[1, 2].bar(['Skewness'], [shape_data.get('skewness', 0)], color='orange', alpha=0.7)
            ax_twin = axes[1, 2].twinx()
            ax_twin.bar(['Kurtosis'], [shape_data.get('kurtosis', 0)], color='red', alpha=0.7)
            axes[1, 2].set_title('Distribution Shape')
            axes[1, 2].set_ylabel('Skewness', color='orange')
            ax_twin.set_ylabel('Kurtosis', color='red')

        plt.tight_layout()
        return fig

    def plot_returns_analysis(self, returns_data: Dict[str, Any],
                            figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create comprehensive returns analysis plot

        Args:
            returns_data: Dictionary with returns data
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Returns Analysis', fontsize=16, fontweight='bold')

        # Returns time series
        if 'returns' in returns_data:
            returns = returns_data['returns']
            axes[0, 0].plot(returns.index, returns.values)
            axes[0, 0].set_title('Returns Time Series')
            axes[0, 0].set_ylabel('Return (%)')

        # Returns histogram
        if 'returns' in returns_data:
            axes[0, 1].hist(returns.values, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Returns Distribution')
            axes[0, 1].set_xlabel('Return (%)')
            axes[0, 1].set_ylabel('Frequency')

        # Cumulative returns
        if 'cumulative_returns' in returns_data:
            cum_returns = returns_data['cumulative_returns']
            axes[0, 2].plot(cum_returns.index, cum_returns.values)
            axes[0, 2].set_title('Cumulative Returns')
            axes[0, 2].set_ylabel('Cumulative Return')

        # Drawdown
        if 'drawdown' in returns_data:
            drawdown = returns_data['drawdown']['drawdown_series']
            axes[1, 0].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
            axes[1, 0].plot(drawdown.index, drawdown.values, color='red')
            axes[1, 0].set_title('Drawdown')
            axes[1, 0].set_ylabel('Drawdown (%)')

        # Rolling volatility
        if 'returns' in returns_data:
            returns = returns_data['returns']
            rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)  # Annualized
            axes[1, 1].plot(rolling_vol.index, rolling_vol.values)
            axes[1, 1].set_title('Rolling Volatility (30-day)')
            axes[1, 1].set_ylabel('Annualized Volatility (%)')

        # Risk-return scatter (if multiple assets)
        if 'risk_return' in returns_data:
            risk_return = returns_data['risk_return']
            axes[1, 2].scatter(risk_return['risk'], risk_return['return'])
            for i, txt in enumerate(risk_return['assets']):
                axes[1, 2].annotate(txt, (risk_return['risk'][i], risk_return['return'][i]))
            axes[1, 2].set_title('Risk-Return Profile')
            axes[1, 2].set_xlabel('Risk (Volatility)')
            axes[1, 2].set_ylabel('Return')

        plt.tight_layout()
        return fig


class ComprehensiveStatistics:
    """
    Main class that combines all statistical analysis functionality
    """

    def __init__(self):
        self.calculator = SampleStatisticsCalculator()
        self.returns_calculator = FinancialReturnsCalculator()
        self.visualizer = StatisticsVisualizer()
        self.results = {}

    def compute_comprehensive_statistics(self, data: pd.DataFrame,
                                       target_columns: Optional[List[str]] = None,
                                       price_columns: Optional[List[str]] = None,
                                       compute_returns: bool = True) -> Dict[str, Any]:
        """
        Compute comprehensive statistics for a dataset

        Args:
            data: DataFrame with data
            target_columns: Columns for statistical analysis
            price_columns: Columns containing price data for returns analysis
            compute_returns: Whether to compute returns analysis

        Returns:
            Dictionary with comprehensive results
        """
        if target_columns is None:
            target_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        results = {
            'dataset_info': {
                'shape': data.shape,
                'columns': data.columns.tolist(),
                'target_columns': target_columns,
                'missing_values': data.isnull().sum().to_dict()
            }
        }

        # Basic statistics for each column
        results['column_statistics'] = {}
        for col in target_columns:
            try:
                basic_stats = self.calculator.compute_basic_statistics(data, col)
                variance_stats = self.calculator.compute_variance_measures(data, col, trim_lower=5, trim_upper=5)
                quantile_stats = self.calculator.compute_quantiles(data, col)

                results['column_statistics'][col] = {
                    'basic': basic_stats,
                    'variance': variance_stats,
                    'quantiles': quantile_stats
                }
            except Exception as e:
                results['column_statistics'][col] = {'error': str(e)}

        # Correlation analysis
        try:
            corr_results = self.calculator.compute_pairwise_correlations(data, target_columns)
            results['correlation_analysis'] = corr_results
        except Exception as e:
            results['correlation_analysis'] = {'error': str(e)}

        # Covariance analysis
        try:
            cov_results = self.calculator.compute_covariance_matrix(data, target_columns)
            results['covariance_analysis'] = cov_results
        except Exception as e:
            results['covariance_analysis'] = {'error': str(e)}

        # Returns analysis
        if compute_returns and price_columns:
            results['returns_analysis'] = {}
            for col in price_columns:
                try:
                    # Compute different types of returns
                    pct_returns = self.returns_calculator.compute_percentage_returns(data, col)
                    log_returns = self.returns_calculator.compute_log_returns(data, col)
                    drawdown_stats = self.returns_calculator.compute_drawdown(data, col)

                    # Returns statistics
                    returns_stats = self.calculator.compute_basic_statistics(pct_returns)
                    returns_variance = self.calculator.compute_variance_measures(pct_returns)

                    results['returns_analysis'][col] = {
                        'percentage_returns': pct_returns,
                        'log_returns': log_returns,
                        'drawdown': drawdown_stats,
                        'returns_statistics': returns_stats,
                        'returns_variance': returns_variance
                    }
                except Exception as e:
                    results['returns_analysis'][col] = {'error': str(e)}

        # Store comprehensive results
        self.results = results
        return results

    def generate_statistics_report(self, data: pd.DataFrame,
                                 target_columns: Optional[List[str]] = None,
                                 create_visualizations: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive statistics report

        Args:
            data: DataFrame with data
            target_columns: Columns for analysis
            create_visualizations: Whether to create plots

        Returns:
            Dictionary with report results
        """
        # Compute comprehensive statistics
        stats_results = self.compute_comprehensive_statistics(data, target_columns)

        # Create visualizations if requested
        if create_visualizations:
            visualizations = {}

            # Correlation heatmap
            try:
                corr_matrix = stats_results['correlation_analysis']['pearson']['correlation_matrix']
                corr_fig = self.visualizer.plot_correlation_heatmap(corr_matrix)
                visualizations['correlation_heatmap'] = corr_fig
            except:
                pass

            # Covariance heatmap
            try:
                cov_matrix = stats_results['covariance_analysis']['covariance_matrix']
                cov_fig = self.visualizer.plot_covariance_heatmap(cov_matrix)
                visualizations['covariance_heatmap'] = cov_fig
            except:
                pass

            # Statistics summary for first column
            try:
                first_col = target_columns[0] if target_columns else data.columns[0]
                col_stats = stats_results['column_statistics'][first_col]
                summary_fig = self.visualizer.plot_statistics_summary(col_stats)
                visualizations['statistics_summary'] = summary_fig
            except:
                pass

            stats_results['visualizations'] = visualizations

        return stats_results

    def export_results_to_excel(self, results: Dict[str, Any], filename: str):
        """
        Export results to Excel file

        Args:
            results: Results dictionary
            filename: Output filename
        """
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Basic statistics
            if 'column_statistics' in results:
                basic_stats_data = []
                for col, stats in results['column_statistics'].items():
                    if 'error' not in stats:
                        row = {'Column': col}
                        row.update(stats['basic']['basic'])
                        row.update(stats['variance'])
                        basic_stats_data.append(row)

                if basic_stats_data:
                    pd.DataFrame(basic_stats_data).to_excel(writer, sheet_name='Basic Statistics', index=False)

            # Correlation matrix
            if 'correlation_analysis' in results:
                corr_data = results['correlation_analysis']
                if 'pearson' in corr_data:
                    corr_data['pearson']['correlation_matrix'].to_excel(writer, sheet_name='Correlation Matrix')

            # Covariance matrix
            if 'covariance_analysis' in results:
                cov_data = results['covariance_analysis']
                if 'covariance_matrix' in cov_data:
                    cov_data['covariance_matrix'].to_excel(writer, sheet_name='Covariance Matrix')

        print(f"Results exported to {filename}")


# Utility functions
def quick_stats(data: Union[pd.DataFrame, pd.Series], column: Optional[str] = None) -> pd.DataFrame:
    """
    Quick statistics summary for a single variable

    Args:
        data: Data (DataFrame or Series)
        column: Column name if DataFrame

    Returns:
        DataFrame with statistics summary
    """
    calculator = SampleStatisticsCalculator()

    # Get basic statistics
    basic_stats = calculator.compute_basic_statistics(data, column)
    variance_stats = calculator.compute_variance_measures(data, column)
    quantile_stats = calculator.compute_quantiles(data, column)

    # Combine results
    summary_data = []

    # Basic statistics
    for key, value in basic_stats['basic'].items():
        summary_data.append({'Statistic': key.replace('_', ' ').title(), 'Value': value})

    # Shape statistics
    for key, value in basic_stats['shape'].items():
        summary_data.append({'Statistic': key.replace('_', ' ').title(), 'Value': value})

    # Variance statistics
    for key, value in variance_stats.items():
        if key not in ['trimming_lower', 'trimming_upper']:
            summary_data.append({'Statistic': key.replace('_', ' ').title(), 'Value': value})

    # Quartiles
    for key, value in basic_stats['quartiles'].items():
        summary_data.append({'Statistic': key.upper(), 'Value': value})

    return pd.DataFrame(summary_data)


if __name__ == "__main__":
    print("Sample Statistics Module")
    print("========================")

    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000

    # Generate sample financial data
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    returns_a = np.random.normal(0.05, 0.15, n_samples)
    returns_b = np.random.normal(0.03, 0.12, n_samples)
    returns_c = 0.5 * returns_a + 0.5 * np.random.normal(0.02, 0.10, n_samples)

    # Create prices from returns
    prices_a = 100 * np.cumprod(1 + returns_a/100)
    prices_b = 100 * np.cumprod(1 + returns_b/100)
    prices_c = 100 * np.cumprod(1 + returns_c/100)

    # Create DataFrame
    sample_data = pd.DataFrame({
        'Date': dates,
        'Asset_A_Price': prices_a,
        'Asset_B_Price': prices_b,
        'Asset_C_Price': prices_c,
        'Asset_A_Returns': returns_a,
        'Asset_B_Returns': returns_b,
        'Asset_C_Returns': returns_c
    })

    print(f"Sample dataset created with shape: {sample_data.shape}")

    # Initialize comprehensive statistics
    comp_stats = ComprehensiveStatistics()

    # Generate comprehensive report
    print("\nGenerating comprehensive statistics report...")
    report = comp_stats.generate_statistics_report(
        sample_data,
        target_columns=['Asset_A_Returns', 'Asset_B_Returns', 'Asset_C_Returns'],
        create_visualizations=True
    )

    print(f"Report generated with {len(report)} main sections")

    # Quick stats example
    print("\nQuick statistics for Asset A Returns:")
    quick_summary = quick_stats(sample_data, 'Asset_A_Returns')
    print(quick_summary.head(10))

    print("\nSample Statistics module ready for use!")