"""
Exploratory Data Analysis Module for Financial and Economic Data

This module provides comprehensive exploratory data analysis functions for:
- Categorical variable analysis (encoding, frequency counts, cross-tabulation)
- Quantitative variable analysis (distributions, histograms, scatter plots)
- Spatial analysis (geographic scatter plots, 3D spatial visualization)
- Statistical testing and correlation analysis
- Interactive visualizations and reports

Dependencies: pandas, numpy, matplotlib, seaborn, plotly, scipy, sklearn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Optional, List, Dict, Any, Union, Tuple
from scipy import stats
from scipy.stats import chi2_contingency
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

try:
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Some encoding features will be disabled.")


class CategoricalAnalyzer:
    """
    Class for analyzing categorical variables
    """

    def __init__(self):
        self.analysis_results = {}
        self.encoders = {}

    def get_frequency_counts(self, df: pd.DataFrame, column: str,
                           normalize: bool = False, dropna: bool = True,
                           sort_by: str = 'count') -> pd.DataFrame:
        """
        Get frequency counts for a categorical variable

        Args:
            df: DataFrame containing the column
            column: Name of categorical column
            normalize: If True, return proportions instead of counts
            dropna: If True, exclude NaN values
            sort_by: Sort by 'count', 'index', or 'value'

        Returns:
            DataFrame with frequency counts
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        counts = df[column].value_counts(normalize=normalize, dropna=dropna)

        result_df = pd.DataFrame({
            'Category': counts.index,
            'Count' if not normalize else 'Proportion': counts.values
        })

        if normalize:
            result_df['Percentage'] = result_df['Proportion'] * 100

        # Sort results
        if sort_by == 'count':
            result_df = result_df.sort_values('Count' if not normalize else 'Proportion', ascending=False)
        elif sort_by == 'index':
            result_df = result_df.sort_values('Category')

        result_df = result_df.reset_index(drop=True)

        # Store results
        self.analysis_results[f'{column}_frequency'] = result_df

        return result_df

    def encode_categorical(self, df: pd.DataFrame, column: str,
                         method: str = 'label', **kwargs) -> pd.DataFrame:
        """
        Encode categorical variables using various methods

        Args:
            df: DataFrame containing the column
            column: Name of categorical column
            method: Encoding method ('label', 'ordinal', 'onehot', 'target')
            **kwargs: Additional parameters for encoding methods

        Returns:
            DataFrame with encoded column(s)
        """
        if not SKLEARN_AVAILABLE and method in ['label', 'ordinal', 'onehot']:
            raise ImportError("Scikit-learn required for advanced encoding methods")

        df_encoded = df.copy()

        if method == 'label':
            encoder = LabelEncoder()
            df_encoded[f'{column}_encoded'] = encoder.fit_transform(df_encoded[column].astype(str))
            self.encoders[f'{column}_label'] = encoder

        elif method == 'ordinal':
            encoder = OrdinalEncoder()
            df_encoded[f'{column}_encoded'] = encoder.fit_transform(df_encoded[[column]])
            self.encoders[f'{column}_ordinal'] = encoder

        elif method == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, drop='first', **kwargs)
            encoded_array = encoder.fit_transform(df_encoded[[column]])
            feature_names = [f'{column}_{cat}' for cat in encoder.categories_[0][1:]]

            for i, name in enumerate(feature_names):
                df_encoded[name] = encoded_array[:, i]

            self.encoders[f'{column}_onehot'] = encoder

        elif method == 'dummy':
            dummies = pd.get_dummies(df_encoded[column], prefix=column, drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)

        elif method == 'frequency':
            freq_map = df_encoded[column].value_counts().to_dict()
            df_encoded[f'{column}_freq_encoded'] = df_encoded[column].map(freq_map)

        elif method == 'target':
            # Target encoding requires a target variable
            target = kwargs.get('target')
            if target is None:
                raise ValueError("Target variable required for target encoding")

            target_means = df_encoded.groupby(column)[target].mean()
            df_encoded[f'{column}_target_encoded'] = df_encoded[column].map(target_means)

        return df_encoded

    def cross_tabulation(self, df: pd.DataFrame, col1: str, col2: str,
                        normalize: Optional[str] = None, margins: bool = True,
                        chi2_test: bool = True) -> Dict[str, Any]:
        """
        Create cross-tabulation table for two categorical variables

        Args:
            df: DataFrame containing the columns
            col1: First categorical column
            col2: Second categorical column
            normalize: Normalize by 'index', 'columns', 'all', or None
            margins: Add margin totals
            chi2_test: Perform chi-square test of independence

        Returns:
            Dictionary with cross-tabulation results and statistics
        """
        if col1 not in df.columns or col2 not in df.columns:
            raise ValueError(f"One or both columns not found in DataFrame")

        # Create cross-tabulation
        crosstab = pd.crosstab(df[col1], df[col2], normalize=normalize, margins=margins)

        results = {
            'crosstab': crosstab,
            'col1': col1,
            'col2': col2,
            'normalize': normalize
        }

        # Chi-square test of independence
        if chi2_test:
            # Create contingency table without margins for statistical test
            contingency = pd.crosstab(df[col1], df[col2])
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency)

            results['chi2_test'] = {
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'expected_frequencies': pd.DataFrame(expected,
                                                   index=contingency.index,
                                                   columns=contingency.columns),
                'significant': p_value < 0.05
            }

        # Store results
        self.analysis_results[f'{col1}_vs_{col2}_crosstab'] = results

        return results

    def contingency_table_with_probabilities(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, pd.DataFrame]:
        """
        Create contingency table with fitted probabilities

        Args:
            df: DataFrame containing the columns
            col1: First categorical column
            col2: Second categorical column

        Returns:
            Dictionary with observed, expected, and probability tables
        """
        # Observed frequencies
        observed = pd.crosstab(df[col1], df[col2])

        # Chi-square test to get expected frequencies
        chi2_stat, p_value, dof, expected = chi2_contingency(observed)
        expected_df = pd.DataFrame(expected, index=observed.index, columns=observed.columns)

        # Calculate probabilities
        total = observed.sum().sum()
        prob_observed = observed / total
        prob_expected = expected_df / total

        # Row and column probabilities
        row_probs = observed.div(observed.sum(axis=1), axis=0)
        col_probs = observed.div(observed.sum(axis=0), axis=1)

        results = {
            'observed_frequencies': observed,
            'expected_frequencies': expected_df,
            'observed_probabilities': prob_observed,
            'expected_probabilities': prob_expected,
            'row_conditional_probabilities': row_probs,
            'column_conditional_probabilities': col_probs,
            'chi2_statistic': chi2_stat,
            'p_value': p_value
        }

        return results

    def plot_frequency_distribution(self, df: pd.DataFrame, column: str,
                                  figsize: Tuple[int, int] = (10, 6),
                                  top_n: Optional[int] = None) -> plt.Figure:
        """
        Plot frequency distribution of categorical variable

        Args:
            df: DataFrame containing the column
            column: Name of categorical column
            figsize: Figure size
            top_n: Show only top N categories

        Returns:
            Matplotlib figure object
        """
        freq_data = self.get_frequency_counts(df, column)

        if top_n:
            freq_data = freq_data.head(top_n)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Bar plot
        bars = ax1.bar(range(len(freq_data)), freq_data['Count'])
        ax1.set_xlabel('Categories')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Frequency Distribution of {column}')
        ax1.set_xticks(range(len(freq_data)))
        ax1.set_xticklabels(freq_data['Category'], rotation=45, ha='right')

        # Pie chart
        ax2.pie(freq_data['Count'], labels=freq_data['Category'], autopct='%1.1f%%')
        ax2.set_title(f'Proportion Distribution of {column}')

        plt.tight_layout()
        return fig

    def plot_crosstab_heatmap(self, df: pd.DataFrame, col1: str, col2: str,
                             normalize: Optional[str] = None,
                             figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot heatmap of cross-tabulation

        Args:
            df: DataFrame containing the columns
            col1: First categorical column
            col2: Second categorical column
            normalize: Normalization method
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        crosstab_results = self.cross_tabulation(df, col1, col2, normalize=normalize, margins=False)
        crosstab = crosstab_results['crosstab']

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(crosstab, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', ax=ax)
        ax.set_title(f'Cross-tabulation Heatmap: {col1} vs {col2}')

        return fig


class QuantitativeAnalyzer:
    """
    Class for analyzing quantitative variables
    """

    def __init__(self):
        self.analysis_results = {}

    def get_distribution_summary(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Get comprehensive distribution summary for quantitative variable

        Args:
            df: DataFrame containing the column
            column: Name of quantitative column

        Returns:
            Dictionary with distribution statistics
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        data = df[column].dropna()

        summary = {
            'basic_stats': {
                'count': len(data),
                'mean': data.mean(),
                'median': data.median(),
                'mode': data.mode().iloc[0] if len(data.mode()) > 0 else np.nan,
                'std': data.std(),
                'var': data.var(),
                'min': data.min(),
                'max': data.max(),
                'range': data.max() - data.min()
            },
            'percentiles': {
                'q25': data.quantile(0.25),
                'q50': data.quantile(0.50),
                'q75': data.quantile(0.75),
                'iqr': data.quantile(0.75) - data.quantile(0.25)
            },
            'shape': {
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data)
            },
            'missing_values': df[column].isnull().sum(),
            'unique_values': data.nunique(),
            'zeros': (data == 0).sum()
        }

        # Normality tests
        if len(data) >= 3:
            shapiro_stat, shapiro_p = stats.shapiro(data[:5000])  # Limit for computational efficiency
            summary['normality_tests'] = {
                'shapiro_wilk': {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                }
            }

        # Outlier detection using IQR method
        q1, q3 = summary['percentiles']['q25'], summary['percentiles']['q75']
        iqr = summary['percentiles']['iqr']
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = data[(data < lower_bound) | (data > upper_bound)]
        summary['outliers'] = {
            'count': len(outliers),
            'percentage': (len(outliers) / len(data)) * 100,
            'values': outliers.tolist()[:20]  # Show first 20 outliers
        }

        # Store results
        self.analysis_results[f'{column}_distribution'] = summary

        return summary

    def plot_distribution(self, df: pd.DataFrame, column: str,
                         figsize: Tuple[int, int] = (15, 10),
                         bins: int = 30) -> plt.Figure:
        """
        Create comprehensive distribution plots for quantitative variable

        Args:
            df: DataFrame containing the column
            column: Name of quantitative column
            figsize: Figure size
            bins: Number of bins for histogram

        Returns:
            Matplotlib figure object
        """
        data = df[column].dropna()

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Distribution Analysis: {column}', fontsize=16)

        # Histogram
        axes[0, 0].hist(data, bins=bins, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
        axes[0, 0].axvline(data.median(), color='green', linestyle='--', label=f'Median: {data.median():.2f}')
        axes[0, 0].set_title('Histogram')
        axes[0, 0].set_xlabel(column)
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()

        # Density plot
        data.plot.density(ax=axes[0, 1])
        axes[0, 1].set_title('Density Plot')
        axes[0, 1].set_xlabel(column)

        # Box plot
        axes[0, 2].boxplot(data, vert=True)
        axes[0, 2].set_title('Box Plot')
        axes[0, 2].set_ylabel(column)

        # Q-Q plot
        stats.probplot(data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal)')

        # Violin plot
        axes[1, 1].violinplot([data], positions=[0], showmeans=True, showmedians=True)
        axes[1, 1].set_title('Violin Plot')
        axes[1, 1].set_ylabel(column)

        # Cumulative distribution
        sorted_data = np.sort(data)
        p = np.arange(len(sorted_data)) / (len(sorted_data) - 1)
        axes[1, 2].plot(sorted_data, p)
        axes[1, 2].set_title('Cumulative Distribution')
        axes[1, 2].set_xlabel(column)
        axes[1, 2].set_ylabel('Cumulative Probability')

        plt.tight_layout()
        return fig

    def create_pairwise_scatter_plots(self, df: pd.DataFrame,
                                    columns: Optional[List[str]] = None,
                                    sample_size: Optional[int] = None,
                                    figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Create pairwise scatter plots for quantitative variables

        Args:
            df: DataFrame containing the columns
            columns: List of columns to include (if None, uses all numeric columns)
            sample_size: Sample size for large datasets
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Sample data if needed
        df_plot = df[columns].dropna()
        if sample_size and len(df_plot) > sample_size:
            df_plot = df_plot.sample(sample_size, random_state=42)

        # Create pair plot
        fig = plt.figure(figsize=figsize)
        pair_plot = sns.pairplot(df_plot, diag_kind='hist')
        plt.suptitle('Pairwise Scatter Plots', y=1.02, fontsize=16)

        return pair_plot.fig

    def correlation_analysis(self, df: pd.DataFrame,
                           columns: Optional[List[str]] = None,
                           method: str = 'pearson') -> Dict[str, Any]:
        """
        Perform correlation analysis between quantitative variables

        Args:
            df: DataFrame containing the columns
            columns: List of columns to include
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            Dictionary with correlation results
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        corr_matrix = df[columns].corr(method=method)

        # Find strongest correlations
        corr_pairs = []
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                corr_pairs.append({
                    'variable1': columns[i],
                    'variable2': columns[j],
                    'correlation': corr_matrix.iloc[i, j],
                    'abs_correlation': abs(corr_matrix.iloc[i, j])
                })

        corr_pairs_df = pd.DataFrame(corr_pairs).sort_values('abs_correlation', ascending=False)

        results = {
            'correlation_matrix': corr_matrix,
            'correlation_pairs': corr_pairs_df,
            'strongest_positive': corr_pairs_df.iloc[0] if len(corr_pairs_df) > 0 else None,
            'method': method
        }

        return results

    def plot_correlation_heatmap(self, df: pd.DataFrame,
                               columns: Optional[List[str]] = None,
                               method: str = 'pearson',
                               figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot correlation heatmap

        Args:
            df: DataFrame containing the columns
            columns: List of columns to include
            method: Correlation method
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        corr_results = self.correlation_analysis(df, columns, method)
        corr_matrix = corr_results['correlation_matrix']

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, ax=ax)
        ax.set_title(f'{method.capitalize()} Correlation Matrix')

        return fig

    def describe_quantitative_vector(self, df: pd.DataFrame, column: str,
                                   percentiles: Optional[List[float]] = None,
                                   include_mode: bool = True) -> pd.DataFrame:
        """
        Generate descriptive statistics for quantitative variable

        Args:
            df: DataFrame containing the column
            column: Name of quantitative column
            percentiles: List of percentiles to include (default: [0.25, 0.5, 0.75])
            include_mode: Whether to include mode in description

        Returns:
            DataFrame with descriptive statistics
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        if percentiles is None:
            percentiles = [0.25, 0.5, 0.75]

        data = df[column].dropna()

        # Basic describe
        desc = data.describe(percentiles=percentiles)

        # Additional statistics
        desc_dict = desc.to_dict()

        # Add mode if requested
        if include_mode:
            mode_values = data.mode()
            desc_dict['mode'] = mode_values.iloc[0] if len(mode_values) > 0 else np.nan

        # Add additional statistics
        desc_dict['variance'] = data.var()
        desc_dict['skewness'] = stats.skew(data)
        desc_dict['kurtosis'] = stats.kurtosis(data)
        desc_dict['range'] = data.max() - data.min()
        desc_dict['iqr'] = data.quantile(0.75) - data.quantile(0.25)
        desc_dict['coefficient_of_variation'] = data.std() / data.mean() if data.mean() != 0 else np.nan

        # Convert to DataFrame for better display
        result_df = pd.DataFrame({
            'Statistic': list(desc_dict.keys()),
            'Value': list(desc_dict.values())
        })

        # Store results
        self.analysis_results[f'{column}_description'] = result_df

        return result_df

    def create_boxplot_analysis(self, df: pd.DataFrame,
                              columns: Union[str, List[str]],
                              by_column: Optional[str] = None,
                              clipped: bool = False,
                              clip_percentiles: Tuple[float, float] = (0.05, 0.95),
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create boxplots for quantitative variables with optional clipping

        Args:
            df: DataFrame containing the columns
            columns: Column name(s) to create boxplots for
            by_column: Optional categorical column to group by
            clipped: Whether to clip outliers based on percentiles
            clip_percentiles: Percentiles to use for clipping (lower, upper)
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        if isinstance(columns, str):
            columns = [columns]

        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

        df_plot = df.copy()

        # Apply clipping if requested
        if clipped:
            for col in columns:
                lower_clip = df_plot[col].quantile(clip_percentiles[0])
                upper_clip = df_plot[col].quantile(clip_percentiles[1])
                df_plot[col] = df_plot[col].clip(lower_clip, upper_clip)

        # Create figure
        if by_column:
            # Grouped boxplots
            n_cols = len(columns)
            fig, axes = plt.subplots(1, n_cols, figsize=(figsize[0] * n_cols / 2, figsize[1]))

            if n_cols == 1:
                axes = [axes]

            for i, col in enumerate(columns):
                # Create boxplot by category
                categories = df_plot[by_column].unique()
                box_data = [df_plot[df_plot[by_column] == cat][col].dropna() for cat in categories]

                axes[i].boxplot(box_data, labels=categories)
                axes[i].set_title(f'{col} by {by_column}' + (' (Clipped)' if clipped else ''))
                axes[i].set_ylabel(col)
                axes[i].tick_params(axis='x', rotation=45)

                # Add mean markers
                means = [data.mean() for data in box_data]
                axes[i].scatter(range(1, len(categories) + 1), means,
                               marker='x', color='red', s=50, label='Mean')
                axes[i].legend()
        else:
            # Simple boxplots
            fig, ax = plt.subplots(figsize=figsize)

            # Prepare data for boxplot
            box_data = [df_plot[col].dropna() for col in columns]

            bp = ax.boxplot(box_data, labels=columns, patch_artist=True)

            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(columns)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Add mean markers
            means = [data.mean() for data in box_data]
            ax.scatter(range(1, len(columns) + 1), means,
                      marker='x', color='red', s=100, label='Mean', zorder=5)

            ax.set_title('Boxplot Analysis' + (' (Clipped)' if clipped else ''))
            ax.tick_params(axis='x', rotation=45)
            ax.legend()

            # Add grid
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_clipped_boxplot(self, df: pd.DataFrame,
                             columns: Union[str, List[str]],
                             clip_method: str = 'percentile',
                             clip_values: Union[Tuple[float, float], float] = (0.05, 0.95),
                             by_column: Optional[str] = None,
                             figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create boxplots with various clipping methods

        Args:
            df: DataFrame containing the columns
            columns: Column name(s) to create boxplots for
            clip_method: Method for clipping ('percentile', 'iqr', 'std')
            clip_values: Values for clipping based on method
            by_column: Optional categorical column to group by
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        if isinstance(columns, str):
            columns = [columns]

        df_plot = df.copy()

        # Apply different clipping methods
        for col in columns:
            if clip_method == 'percentile':
                lower_clip = df_plot[col].quantile(clip_values[0])
                upper_clip = df_plot[col].quantile(clip_values[1])
            elif clip_method == 'iqr':
                Q1 = df_plot[col].quantile(0.25)
                Q3 = df_plot[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_clip = Q1 - clip_values * IQR
                upper_clip = Q3 + clip_values * IQR
            elif clip_method == 'std':
                mean_val = df_plot[col].mean()
                std_val = df_plot[col].std()
                lower_clip = mean_val - clip_values * std_val
                upper_clip = mean_val + clip_values * std_val
            else:
                raise ValueError("clip_method must be 'percentile', 'iqr', or 'std'")

            df_plot[col] = df_plot[col].clip(lower_clip, upper_clip)

        return self.create_boxplot_analysis(df_plot, columns, by_column,
                                          clipped=True, figsize=figsize)


class PivotAnalyzer:
    """
    Class for pivot tables, group by operations, and pivot charts
    """

    def __init__(self):
        self.pivot_results = {}

    def quick_pivot(self, df: pd.DataFrame,
                   index_cols: Union[str, List[str]],
                   value_cols: Union[str, List[str]],
                   agg_funcs: Union[str, List[str]] = 'mean',
                   columns_col: Optional[str] = None,
                   normalize: bool = False) -> pd.DataFrame:
        """
        Create pivot table with intuitive interface

        Args:
            df: DataFrame to pivot
            index_cols: Column(s) to use as index
            value_cols: Column(s) to aggregate
            agg_funcs: Aggregation function(s) ('mean', 'sum', 'count', 'std', etc.)
            columns_col: Optional column to use as pivot columns
            normalize: Whether to normalize values as percentages

        Returns:
            Pivot table DataFrame
        """
        if isinstance(index_cols, str):
            index_cols = [index_cols]
        if isinstance(value_cols, str):
            value_cols = [value_cols]
        if isinstance(agg_funcs, str):
            agg_funcs = [agg_funcs]

        # Create pivot table
        pivot_table = pd.pivot_table(
            df,
            index=index_cols,
            values=value_cols,
            aggfunc=agg_funcs[0] if len(agg_funcs) == 1 else agg_funcs,
            columns=columns_col,
            fill_value=0
        )

        # Normalize if requested
        if normalize:
            if columns_col:
                # Normalize across columns (rows sum to 100%)
                pivot_table = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100
            else:
                # Normalize entire table
                pivot_table = (pivot_table / pivot_table.sum().sum()) * 100

        # Store results
        result_key = f"pivot_{'-'.join(index_cols)}_by_{'-'.join(value_cols)}"
        self.pivot_results[result_key] = pivot_table

        return pivot_table

    def group_by_analysis(self, df: pd.DataFrame,
                         group_cols: Union[str, List[str]],
                         agg_dict: Optional[Dict[str, Union[str, List[str]]]] = None,
                         include_percentages: bool = True) -> pd.DataFrame:
        """
        Perform group by operations with multiple aggregations

        Args:
            df: DataFrame to group
            group_cols: Column(s) to group by
            agg_dict: Dictionary mapping columns to aggregation functions
            include_percentages: Whether to include percentage calculations

        Returns:
            Grouped DataFrame with aggregations
        """
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        # Default aggregation if none provided
        if agg_dict is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            agg_dict = {col: ['count', 'mean', 'std', 'min', 'max'] for col in numeric_cols}

        # Perform group by
        grouped = df.groupby(group_cols).agg(agg_dict)

        # Flatten column names if multi-level
        if isinstance(grouped.columns, pd.MultiIndex):
            grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

        # Add percentages if requested
        if include_percentages:
            # Add count percentages
            count_cols = [col for col in grouped.columns if 'count' in col.lower()]
            for count_col in count_cols:
                total_count = grouped[count_col].sum()
                grouped[f'{count_col}_percentage'] = (grouped[count_col] / total_count) * 100

        # Reset index for easier use
        grouped = grouped.reset_index()

        # Store results
        result_key = f"groupby_{'-'.join(group_cols)}"
        self.pivot_results[result_key] = grouped

        return grouped

    def create_pivot_chart(self, df: pd.DataFrame,
                          index_col: str,
                          value_col: str,
                          columns_col: Optional[str] = None,
                          agg_func: str = 'mean',
                          chart_type: str = 'bar',
                          normalize: bool = False,
                          figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create pivot chart visualization

        Args:
            df: DataFrame to pivot and chart
            index_col: Column to use as index
            value_col: Column to aggregate
            columns_col: Optional column to use as pivot columns
            agg_func: Aggregation function
            chart_type: Type of chart ('bar', 'line', 'area', 'pie')
            normalize: Whether to normalize values
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        # Create pivot table
        pivot_data = self.quick_pivot(df, index_col, value_col, agg_func, columns_col, normalize)

        # Create chart
        fig, ax = plt.subplots(figsize=figsize)

        if chart_type == 'bar':
            pivot_data.plot(kind='bar', ax=ax, rot=45)
        elif chart_type == 'line':
            pivot_data.plot(kind='line', ax=ax, marker='o')
        elif chart_type == 'area':
            pivot_data.plot(kind='area', ax=ax, alpha=0.7)
        elif chart_type == 'pie' and columns_col is None:
            # Pie chart only works for single series
            pivot_data.plot(kind='pie', ax=ax, autopct='%1.1f%%')
        else:
            pivot_data.plot(ax=ax)

        title = f'{agg_func.capitalize()} of {value_col} by {index_col}'
        if columns_col:
            title += f' (grouped by {columns_col})'
        if normalize:
            title += ' - Normalized'

        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        return fig

    def multi_metric_pivot(self, df: pd.DataFrame,
                          index_cols: Union[str, List[str]],
                          metric_cols: List[str],
                          agg_funcs: List[str] = ['mean', 'sum', 'count'],
                          percentile_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create comprehensive pivot with multiple metrics and aggregations

        Args:
            df: DataFrame to pivot
            index_cols: Column(s) to use as index
            metric_cols: List of numeric columns to aggregate
            agg_funcs: List of aggregation functions
            percentile_cols: Columns to calculate percentiles for

        Returns:
            Multi-metric pivot table
        """
        if isinstance(index_cols, str):
            index_cols = [index_cols]

        # Build aggregation dictionary
        agg_dict = {}
        for col in metric_cols:
            agg_dict[col] = agg_funcs.copy()

            # Add percentiles if specified
            if percentile_cols and col in percentile_cols:
                agg_dict[col].extend([
                    lambda x: x.quantile(0.25),
                    lambda x: x.quantile(0.5),
                    lambda x: x.quantile(0.75)
                ])

        # Perform aggregation
        result = df.groupby(index_cols).agg(agg_dict)

        # Flatten column names
        if isinstance(result.columns, pd.MultiIndex):
            new_columns = []
            for col in result.columns:
                if '<lambda>' in str(col[1]):
                    # Handle lambda functions for percentiles
                    if 'quantile(0.25)' in str(col[1]):
                        new_columns.append(f'{col[0]}_25th_percentile')
                    elif 'quantile(0.5)' in str(col[1]):
                        new_columns.append(f'{col[0]}_median')
                    elif 'quantile(0.75)' in str(col[1]):
                        new_columns.append(f'{col[0]}_75th_percentile')
                    else:
                        new_columns.append(f'{col[0]}_{col[1]}')
                else:
                    new_columns.append(f'{col[0]}_{col[1]}')

            result.columns = new_columns

        # Reset index
        result = result.reset_index()

        return result

    def percentage_breakdown(self, df: pd.DataFrame,
                           category_col: str,
                           value_col: Optional[str] = None,
                           normalize_by: str = 'total') -> pd.DataFrame:
        """
        Create percentage breakdown of categorical variables

        Args:
            df: DataFrame to analyze
            category_col: Categorical column to break down
            value_col: Optional numeric column to weight by
            normalize_by: How to normalize ('total', 'category', 'value')

        Returns:
            DataFrame with percentage breakdown
        """
        if value_col:
            # Weighted percentage breakdown
            if normalize_by == 'total':
                total = df[value_col].sum()
                result = df.groupby(category_col)[value_col].sum()
                percentages = (result / total) * 100
            elif normalize_by == 'category':
                result = df.groupby(category_col)[value_col].agg(['sum', 'mean', 'count'])
                total_sum = result['sum'].sum()
                result['percentage_of_total'] = (result['sum'] / total_sum) * 100
                return result
        else:
            # Count-based percentage breakdown
            counts = df[category_col].value_counts()
            total = len(df)
            percentages = (counts / total) * 100

        # Create result DataFrame
        result_df = pd.DataFrame({
            'Category': percentages.index,
            'Count': df[category_col].value_counts()[percentages.index] if not value_col else df.groupby(category_col)[value_col].sum()[percentages.index],
            'Percentage': percentages.values
        })

        # Sort by percentage
        result_df = result_df.sort_values('Percentage', ascending=False)

        return result_df

    def normalized_quantitative_summary(self, df: pd.DataFrame,
                                      numeric_cols: Optional[List[str]] = None,
                                      group_by_col: Optional[str] = None,
                                      normalization_method: str = 'zscore') -> pd.DataFrame:
        """
        Create normalized summary of quantitative variables

        Args:
            df: DataFrame to analyze
            numeric_cols: List of numeric columns (if None, uses all numeric)
            group_by_col: Optional column to group by before normalization
            normalization_method: Method for normalization ('zscore', 'minmax', 'robust')

        Returns:
            DataFrame with normalized estimates
        """
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        df_work = df[numeric_cols + ([group_by_col] if group_by_col else [])].copy()

        # Apply normalization
        if group_by_col:
            # Normalize within groups
            normalized_data = []
            for group_val in df_work[group_by_col].unique():
                group_data = df_work[df_work[group_by_col] == group_val][numeric_cols]

                if normalization_method == 'zscore':
                    normalized_group = (group_data - group_data.mean()) / group_data.std()
                elif normalization_method == 'minmax':
                    normalized_group = (group_data - group_data.min()) / (group_data.max() - group_data.min())
                elif normalization_method == 'robust':
                    median_val = group_data.median()
                    mad = (group_data - median_val).abs().median()
                    normalized_group = (group_data - median_val) / mad

                normalized_group[group_by_col] = group_val
                normalized_data.append(normalized_group)

            normalized_df = pd.concat(normalized_data, ignore_index=True)
        else:
            # Normalize entire dataset
            if normalization_method == 'zscore':
                normalized_df = (df_work - df_work.mean()) / df_work.std()
            elif normalization_method == 'minmax':
                normalized_df = (df_work - df_work.min()) / (df_work.max() - df_work.min())
            elif normalization_method == 'robust':
                median_val = df_work.median()
                mad = (df_work - median_val).abs().median()
                normalized_df = (df_work - median_val) / mad

        return normalized_df


class SpatialAnalyzer:
    """
    Class for spatial analysis of geocoded data
    """

    def __init__(self):
        self.analysis_results = {}

    def create_spatial_scatter_plot(self, df: pd.DataFrame,
                                  lat_col: str, lon_col: str,
                                  value_col: Optional[str] = None,
                                  color_col: Optional[str] = None,
                                  size_col: Optional[str] = None,
                                  title: str = "Spatial Scatter Plot") -> Union[plt.Figure, Any]:
        """
        Create 2D spatial scatter plot of geocoded variables

        Args:
            df: DataFrame containing the data
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            value_col: Column to use for color mapping
            color_col: Column to use for color categories
            size_col: Column to use for point sizes
            title: Plot title

        Returns:
            Matplotlib figure or Plotly figure object
        """
        if not all(col in df.columns for col in [lat_col, lon_col]):
            raise ValueError("Latitude and longitude columns must be in DataFrame")

        df_clean = df.dropna(subset=[lat_col, lon_col])

        if PLOTLY_AVAILABLE:
            # Create interactive Plotly map
            fig = px.scatter_mapbox(
                df_clean,
                lat=lat_col,
                lon=lon_col,
                color=color_col or value_col,
                size=size_col,
                hover_data=df_clean.columns.tolist(),
                title=title,
                mapbox_style="open-street-map",
                zoom=3
            )
            fig.update_layout(height=600)
            return fig
        else:
            # Create static matplotlib plot
            fig, ax = plt.subplots(figsize=(12, 8))

            scatter = ax.scatter(
                df_clean[lon_col],
                df_clean[lat_col],
                c=df_clean[value_col] if value_col else 'blue',
                s=df_clean[size_col] if size_col else 50,
                alpha=0.6,
                cmap='viridis'
            )

            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(title)

            if value_col:
                plt.colorbar(scatter, ax=ax, label=value_col)

            return fig

    def create_3d_spatial_plot(self, df: pd.DataFrame,
                             lat_col: str, lon_col: str, value_col: str,
                             title: str = "3D Spatial Plot") -> Union[plt.Figure, Any]:
        """
        Create 3D spatial scatter plot with measures across geographic extents

        Args:
            df: DataFrame containing the data
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            value_col: Column to use for Z-axis values
            title: Plot title

        Returns:
            Matplotlib 3D figure or Plotly figure object
        """
        if not all(col in df.columns for col in [lat_col, lon_col, value_col]):
            raise ValueError("Latitude, longitude, and value columns must be in DataFrame")

        df_clean = df.dropna(subset=[lat_col, lon_col, value_col])

        if PLOTLY_AVAILABLE:
            # Create interactive 3D Plotly plot
            fig = px.scatter_3d(
                df_clean,
                x=lon_col,
                y=lat_col,
                z=value_col,
                color=value_col,
                title=title,
                labels={
                    lon_col: 'Longitude',
                    lat_col: 'Latitude',
                    value_col: value_col
                }
            )
            fig.update_layout(height=700)
            return fig
        else:
            # Create static matplotlib 3D plot
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(
                df_clean[lon_col],
                df_clean[lat_col],
                df_clean[value_col],
                c=df_clean[value_col],
                cmap='viridis',
                alpha=0.7
            )

            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_zlabel(value_col)
            ax.set_title(title)

            plt.colorbar(scatter, ax=ax, shrink=0.5)

            return fig

    def geographic_aggregation(self, df: pd.DataFrame,
                             lat_col: str, lon_col: str, value_col: str,
                             grid_size: float = 0.1,
                             agg_func: str = 'mean') -> pd.DataFrame:
        """
        Aggregate data points by geographic grid

        Args:
            df: DataFrame containing the data
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            value_col: Column to aggregate
            grid_size: Size of grid cells in degrees
            agg_func: Aggregation function ('mean', 'sum', 'count', 'median')

        Returns:
            DataFrame with aggregated geographic data
        """
        df_clean = df.dropna(subset=[lat_col, lon_col, value_col])

        # Create grid coordinates
        df_clean['lat_grid'] = np.round(df_clean[lat_col] / grid_size) * grid_size
        df_clean['lon_grid'] = np.round(df_clean[lon_col] / grid_size) * grid_size

        # Aggregate by grid
        if agg_func == 'mean':
            agg_df = df_clean.groupby(['lat_grid', 'lon_grid'])[value_col].mean().reset_index()
        elif agg_func == 'sum':
            agg_df = df_clean.groupby(['lat_grid', 'lon_grid'])[value_col].sum().reset_index()
        elif agg_func == 'count':
            agg_df = df_clean.groupby(['lat_grid', 'lon_grid'])[value_col].count().reset_index()
        elif agg_func == 'median':
            agg_df = df_clean.groupby(['lat_grid', 'lon_grid'])[value_col].median().reset_index()

        agg_df.columns = ['latitude', 'longitude', f'{value_col}_{agg_func}']

        return agg_df


class ComprehensiveEDA:
    """
    Main class that combines all EDA functionality
    """

    def __init__(self):
        self.categorical_analyzer = CategoricalAnalyzer()
        self.quantitative_analyzer = QuantitativeAnalyzer()
        self.spatial_analyzer = SpatialAnalyzer()
        self.pivot_analyzer = PivotAnalyzer()
        self.report_results = {}

    def generate_comprehensive_report(self, df: pd.DataFrame,
                                    target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive EDA report for entire DataFrame

        Args:
            df: DataFrame to analyze
            target_column: Optional target variable for supervised learning context

        Returns:
            Dictionary with comprehensive analysis results
        """
        report = {
            'dataset_overview': {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'missing_values': df.isnull().sum().to_dict()
            }
        }

        # Categorize columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Analyze categorical variables
        if categorical_cols:
            report['categorical_analysis'] = {}
            for col in categorical_cols[:5]:  # Limit to first 5 for efficiency
                try:
                    freq_counts = self.categorical_analyzer.get_frequency_counts(df, col)
                    report['categorical_analysis'][col] = {
                        'frequency_counts': freq_counts.head(10).to_dict(),
                        'unique_values': df[col].nunique(),
                        'most_common': freq_counts.iloc[0]['Category'] if len(freq_counts) > 0 else None
                    }
                except Exception as e:
                    report['categorical_analysis'][col] = {'error': str(e)}

        # Analyze quantitative variables
        if numeric_cols:
            report['quantitative_analysis'] = {}
            for col in numeric_cols:
                try:
                    summary = self.quantitative_analyzer.get_distribution_summary(df, col)
                    report['quantitative_analysis'][col] = summary
                except Exception as e:
                    report['quantitative_analysis'][col] = {'error': str(e)}

        # Correlation analysis
        if len(numeric_cols) > 1:
            try:
                corr_analysis = self.quantitative_analyzer.correlation_analysis(df, numeric_cols)
                report['correlation_analysis'] = {
                    'strongest_correlations': corr_analysis['correlation_pairs'].head(10).to_dict(),
                    'correlation_matrix': corr_analysis['correlation_matrix'].to_dict()
                }
            except Exception as e:
                report['correlation_analysis'] = {'error': str(e)}

        # Store comprehensive report
        self.report_results = report

        return report

    def create_eda_dashboard(self, df: pd.DataFrame,
                           output_file: Optional[str] = None) -> str:
        """
        Create comprehensive EDA dashboard

        Args:
            df: DataFrame to analyze
            output_file: Optional file path to save HTML dashboard

        Returns:
            HTML string of dashboard
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required for interactive dashboard")

        # Generate comprehensive report
        report = self.generate_comprehensive_report(df)

        # Create dashboard HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Exploratory Data Analysis Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin: 20px 0; padding: 10px; border: 1px solid #ccc; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f0f0f0; }}
            </style>
        </head>
        <body>
            <h1>Exploratory Data Analysis Dashboard</h1>
            <div class="section">
                <h2>Dataset Overview</h2>
                <div class="metric">Shape: {report['dataset_overview']['shape']}</div>
                <div class="metric">Columns: {len(report['dataset_overview']['columns'])}</div>
                <div class="metric">Memory Usage: {report['dataset_overview']['memory_usage'] / (1024**2):.2f} MB</div>
            </div>
        </body>
        </html>
        """

        if output_file:
            with open(output_file, 'w') as f:
                f.write(html_content)

        return html_content


# Utility Functions
def detect_geographic_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detect potential geographic columns in DataFrame

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with potential geographic column names
    """
    geo_indicators = {
        'latitude': ['lat', 'latitude', 'y', 'lat_deg'],
        'longitude': ['lon', 'lng', 'longitude', 'x', 'lon_deg', 'long'],
        'address': ['address', 'location', 'addr', 'place'],
        'postal': ['zip', 'postal', 'zipcode', 'postcode'],
        'country': ['country', 'nation', 'ctry'],
        'state': ['state', 'province', 'region'],
        'city': ['city', 'town', 'municipality']
    }

    detected = {key: [] for key in geo_indicators.keys()}

    for col in df.columns:
        col_lower = col.lower()
        for geo_type, indicators in geo_indicators.items():
            if any(indicator in col_lower for indicator in indicators):
                detected[geo_type].append(col)

    return {k: v for k, v in detected.items() if v}


def suggest_eda_workflow(df: pd.DataFrame) -> List[str]:
    """
    Suggest EDA workflow based on DataFrame characteristics

    Args:
        df: DataFrame to analyze

    Returns:
        List of suggested analysis steps
    """
    suggestions = []

    # Basic data exploration
    suggestions.append("1. Examine dataset overview (shape, dtypes, missing values)")

    # Categorical analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        suggestions.append(f"2. Analyze categorical variables: {list(categorical_cols[:3])}")
        if len(categorical_cols) >= 2:
            suggestions.append("3. Create cross-tabulation tables for categorical pairs")

    # Quantitative analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        suggestions.append(f"4. Analyze distribution of numeric variables: {list(numeric_cols[:3])}")
        if len(numeric_cols) >= 2:
            suggestions.append("5. Create correlation analysis and scatter plots")

    # Spatial analysis
    geo_cols = detect_geographic_columns(df)
    if geo_cols.get('latitude') and geo_cols.get('longitude'):
        suggestions.append("6. Perform spatial analysis with geographic scatter plots")

    # Advanced analysis
    if len(df) > 10000:
        suggestions.append("7. Consider sampling for large dataset visualization")

    return suggestions


if __name__ == "__main__":
    print("Exploratory Data Analysis Module")
    print("=================================")

    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000

    sample_data = {
        'category_A': np.random.choice(['Type1', 'Type2', 'Type3'], n_samples),
        'category_B': np.random.choice(['Red', 'Blue', 'Green'], n_samples),
        'numeric_1': np.random.normal(100, 15, n_samples),
        'numeric_2': np.random.normal(50, 10, n_samples),
        'latitude': np.random.uniform(30, 50, n_samples),
        'longitude': np.random.uniform(-120, -70, n_samples)
    }

    df_sample = pd.DataFrame(sample_data)
    df_sample['numeric_3'] = df_sample['numeric_1'] * 0.5 + np.random.normal(0, 5, n_samples)

    print(f"Sample dataset created with shape: {df_sample.shape}")

    # Initialize EDA
    eda = ComprehensiveEDA()

    # Get workflow suggestions
    suggestions = suggest_eda_workflow(df_sample)
    print("\nSuggested EDA Workflow:")
    for suggestion in suggestions:
        print(f"  {suggestion}")

    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    report = eda.generate_comprehensive_report(df_sample)
    print(f"Report generated with {len(report)} main sections")

    print("\nEDA module ready for use!")