"""
Data Transformations Module for Time Series Analysis

This module provides comprehensive data transformation capabilities for preparing
vectors within dataframes for time series analysis. It includes various mathematical
transformations that help stabilize variance, normalize distributions, and improve
the statistical properties of time series data.

Key Features:
- Log transformation for exponential growth patterns
- Square root transformation for count data
- Arcsine transformation for proportion data
- Inverse transformation for right-skewed data
- Cube root transformation for preserving sign
- Fourth root transformation for extreme variance
- Box-Cox transformation with optimal lambda estimation

Classes:
    DataTransformer: Main class for applying various transformations
    TransformationAnalyzer: Class for analyzing transformation effects

Dependencies:
    - pandas
    - numpy
    - scipy
    - matplotlib
    - seaborn (optional)
    - warnings

Author: Financial Data Analysis Toolkit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy import stats
from typing import Union, Tuple, Optional, List, Dict, Any

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    warnings.warn("Seaborn not available. Some visualization features will be limited.")

class DataTransformer:
    """
    A comprehensive class for applying various mathematical transformations
    to vectors within dataframes, specifically designed for time series analysis.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataTransformer with a dataframe.

        Parameters:
        -----------
        df : pd.DataFrame
            The input dataframe containing the data to be transformed
        """
        self.df = df.copy()
        self.original_df = df.copy()
        self.transformation_history = {}

    def log_transformation(self, column: str, base: str = 'natural',
                          constant: float = 0) -> pd.Series:
        """
        Apply logarithmic transformation to a column.
        Useful for data with exponential growth patterns or right-skewed distributions.

        Parameters:
        -----------
        column : str
            Name of the column to transform
        base : str, default 'natural'
            Base of logarithm ('natural', '10', '2', or any numeric value)
        constant : float, default 0
            Constant to add before taking log (useful for zero/negative values)

        Returns:
        --------
        pd.Series
            Transformed data series
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")

        data = self.df[column] + constant

        # Check for non-positive values
        if (data <= 0).any():
            min_val = data.min()
            suggested_constant = abs(min_val) + 1
            raise ValueError(f"Non-positive values found. Consider adding constant >= {suggested_constant}")

        if base == 'natural':
            transformed = np.log(data)
        elif base == '10':
            transformed = np.log10(data)
        elif base == '2':
            transformed = np.log2(data)
        else:
            try:
                base_num = float(base)
                transformed = np.log(data) / np.log(base_num)
            except:
                raise ValueError("Base must be 'natural', '10', '2', or a numeric value")

        self.transformation_history[column] = {
            'type': 'log',
            'base': base,
            'constant': constant,
            'original_stats': self._get_stats(self.df[column]),
            'transformed_stats': self._get_stats(transformed)
        }

        return transformed

    def sqrt_transformation(self, column: str, constant: float = 0) -> pd.Series:
        """
        Apply square root transformation to a column.
        Useful for count data or data with moderate right skewness.

        Parameters:
        -----------
        column : str
            Name of the column to transform
        constant : float, default 0
            Constant to add before taking square root

        Returns:
        --------
        pd.Series
            Transformed data series
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")

        data = self.df[column] + constant

        if (data < 0).any():
            min_val = data.min()
            suggested_constant = abs(min_val)
            raise ValueError(f"Negative values found. Consider adding constant >= {suggested_constant}")

        transformed = np.sqrt(data)

        self.transformation_history[column] = {
            'type': 'sqrt',
            'constant': constant,
            'original_stats': self._get_stats(self.df[column]),
            'transformed_stats': self._get_stats(transformed)
        }

        return transformed

    def arcsine_transformation(self, column: str, scale_first: bool = False) -> pd.Series:
        """
        Apply arcsine (inverse sine) transformation to a column.
        Useful for proportion data or data bounded between 0 and 1.

        Parameters:
        -----------
        column : str
            Name of the column to transform
        scale_first : bool, default False
            Whether to scale data to [0,1] range before transformation

        Returns:
        --------
        pd.Series
            Transformed data series
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")

        data = self.df[column].copy()

        if scale_first:
            data_min, data_max = data.min(), data.max()
            if data_max == data_min:
                raise ValueError("Cannot scale constant data")
            data = (data - data_min) / (data_max - data_min)

        if (data < 0).any() or (data > 1).any():
            raise ValueError("Data must be in [0,1] range. Consider setting scale_first=True")

        # Handle edge cases (0 and 1)
        data = np.clip(data, 1e-10, 1 - 1e-10)
        transformed = np.arcsin(np.sqrt(data))

        self.transformation_history[column] = {
            'type': 'arcsine',
            'scale_first': scale_first,
            'original_stats': self._get_stats(self.df[column]),
            'transformed_stats': self._get_stats(transformed)
        }

        return transformed

    def inverse_transformation(self, column: str, constant: float = 0) -> pd.Series:
        """
        Apply inverse (reciprocal) transformation to a column.
        Useful for right-skewed data or when modeling rates/ratios.

        Parameters:
        -----------
        column : str
            Name of the column to transform
        constant : float, default 0
            Constant to add before taking inverse (1/(x+constant))

        Returns:
        --------
        pd.Series
            Transformed data series
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")

        data = self.df[column] + constant

        if (data == 0).any():
            raise ValueError("Zero values found. Consider adding a small constant")

        transformed = 1 / data

        self.transformation_history[column] = {
            'type': 'inverse',
            'constant': constant,
            'original_stats': self._get_stats(self.df[column]),
            'transformed_stats': self._get_stats(transformed)
        }

        return transformed

    def cube_root_transformation(self, column: str) -> pd.Series:
        """
        Apply cube root transformation to a column.
        Useful for preserving the sign of data while reducing skewness.

        Parameters:
        -----------
        column : str
            Name of the column to transform

        Returns:
        --------
        pd.Series
            Transformed data series
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")

        data = self.df[column]

        # Preserve sign for negative values
        transformed = np.sign(data) * np.power(np.abs(data), 1/3)

        self.transformation_history[column] = {
            'type': 'cube_root',
            'original_stats': self._get_stats(self.df[column]),
            'transformed_stats': self._get_stats(transformed)
        }

        return transformed

    def fourth_root_transformation(self, column: str, constant: float = 0) -> pd.Series:
        """
        Apply fourth root transformation to a column.
        Useful for data with extreme variance or very high skewness.

        Parameters:
        -----------
        column : str
            Name of the column to transform
        constant : float, default 0
            Constant to add before taking fourth root

        Returns:
        --------
        pd.Series
            Transformed data series
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")

        data = self.df[column] + constant

        if (data < 0).any():
            min_val = data.min()
            suggested_constant = abs(min_val)
            raise ValueError(f"Negative values found. Consider adding constant >= {suggested_constant}")

        transformed = np.power(data, 1/4)

        self.transformation_history[column] = {
            'type': 'fourth_root',
            'constant': constant,
            'original_stats': self._get_stats(self.df[column]),
            'transformed_stats': self._get_stats(transformed)
        }

        return transformed

    def boxcox_transformation(self, column: str, lambda_val: Optional[float] = None,
                            optimize: bool = True) -> Tuple[pd.Series, float]:
        """
        Apply Box-Cox transformation to a column.
        Automatically finds optimal lambda or uses provided value.

        Parameters:
        -----------
        column : str
            Name of the column to transform
        lambda_val : float, optional
            Lambda parameter for Box-Cox transformation. If None, optimal value is found.
        optimize : bool, default True
            Whether to optimize lambda for normality

        Returns:
        --------
        tuple
            (transformed_series, lambda_used)
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")

        data = self.df[column]

        if (data <= 0).any():
            min_val = data.min()
            suggested_constant = abs(min_val) + 1
            raise ValueError(f"Non-positive values found. Add constant >= {suggested_constant} first")

        if lambda_val is None and optimize:
            # Find optimal lambda
            transformed_data, fitted_lambda = stats.boxcox(data)
        elif lambda_val is not None:
            fitted_lambda = lambda_val
            if lambda_val == 0:
                transformed_data = np.log(data)
            else:
                transformed_data = (np.power(data, lambda_val) - 1) / lambda_val
        else:
            # Use lambda = 1 (no transformation)
            fitted_lambda = 1.0
            transformed_data = data - 1

        transformed_series = pd.Series(transformed_data, index=data.index)

        self.transformation_history[column] = {
            'type': 'boxcox',
            'lambda': fitted_lambda,
            'optimize': optimize,
            'original_stats': self._get_stats(self.df[column]),
            'transformed_stats': self._get_stats(transformed_series)
        }

        return transformed_series, fitted_lambda

    def apply_transformation(self, column: str, transformation_type: str,
                           **kwargs) -> pd.Series:
        """
        Apply a specified transformation to a column.

        Parameters:
        -----------
        column : str
            Name of the column to transform
        transformation_type : str
            Type of transformation ('log', 'sqrt', 'arcsine', 'inverse',
            'cube_root', 'fourth_root', 'boxcox')
        **kwargs
            Additional arguments for the transformation method

        Returns:
        --------
        pd.Series
            Transformed data series
        """
        transformation_map = {
            'log': self.log_transformation,
            'sqrt': self.sqrt_transformation,
            'arcsine': self.arcsine_transformation,
            'inverse': self.inverse_transformation,
            'cube_root': self.cube_root_transformation,
            'fourth_root': self.fourth_root_transformation,
            'boxcox': self.boxcox_transformation
        }

        if transformation_type not in transformation_map:
            raise ValueError(f"Unknown transformation type: {transformation_type}")

        method = transformation_map[transformation_type]

        if transformation_type == 'boxcox':
            result, _ = method(column, **kwargs)
            return result
        else:
            return method(column, **kwargs)

    def batch_transform(self, columns: List[str], transformation_type: str,
                       **kwargs) -> pd.DataFrame:
        """
        Apply the same transformation to multiple columns.

        Parameters:
        -----------
        columns : list of str
            List of column names to transform
        transformation_type : str
            Type of transformation to apply
        **kwargs
            Additional arguments for the transformation

        Returns:
        --------
        pd.DataFrame
            DataFrame with transformed columns
        """
        result_df = self.df.copy()

        for col in columns:
            try:
                transformed = self.apply_transformation(col, transformation_type, **kwargs)
                result_df[f"{col}_{transformation_type}"] = transformed
                print(f"✓ Successfully transformed column '{col}' using {transformation_type}")
            except Exception as e:
                print(f"✗ Failed to transform column '{col}': {str(e)}")

        return result_df

    def get_transformation_summary(self) -> pd.DataFrame:
        """
        Get a summary of all applied transformations.

        Returns:
        --------
        pd.DataFrame
            Summary of transformations with before/after statistics
        """
        if not self.transformation_history:
            print("No transformations have been applied yet.")
            return pd.DataFrame()

        summary_data = []
        for column, info in self.transformation_history.items():
            summary_data.append({
                'Column': column,
                'Transformation': info['type'],
                'Original_Mean': info['original_stats']['mean'],
                'Original_Std': info['original_stats']['std'],
                'Original_Skew': info['original_stats']['skewness'],
                'Transformed_Mean': info['transformed_stats']['mean'],
                'Transformed_Std': info['transformed_stats']['std'],
                'Transformed_Skew': info['transformed_stats']['skewness'],
                'Skew_Improvement': abs(info['original_stats']['skewness']) - abs(info['transformed_stats']['skewness'])
            })

        return pd.DataFrame(summary_data)

    def _get_stats(self, series: pd.Series) -> Dict[str, float]:
        """Calculate basic statistics for a series."""
        return {
            'mean': series.mean(),
            'std': series.std(),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis(),
            'min': series.min(),
            'max': series.max()
        }

class TransformationAnalyzer:
    """
    A class for analyzing the effects of transformations on data distributions
    and time series properties.
    """

    def __init__(self):
        self.comparison_results = {}

    def compare_distributions(self, original: pd.Series, transformed: pd.Series,
                            transformation_name: str = "Transformed") -> Dict[str, Any]:
        """
        Compare the distributions of original and transformed data.

        Parameters:
        -----------
        original : pd.Series
            Original data series
        transformed : pd.Series
            Transformed data series
        transformation_name : str
            Name of the transformation for labeling

        Returns:
        --------
        dict
            Dictionary containing comparison metrics
        """
        comparison = {
            'transformation': transformation_name,
            'original_stats': {
                'mean': original.mean(),
                'std': original.std(),
                'skewness': original.skew(),
                'kurtosis': original.kurtosis(),
                'jarque_bera_p': stats.jarque_bera(original.dropna())[1]
            },
            'transformed_stats': {
                'mean': transformed.mean(),
                'std': transformed.std(),
                'skewness': transformed.skew(),
                'kurtosis': transformed.kurtosis(),
                'jarque_bera_p': stats.jarque_bera(transformed.dropna())[1]
            }
        }

        # Calculate improvements
        comparison['skew_improvement'] = abs(comparison['original_stats']['skewness']) - abs(comparison['transformed_stats']['skewness'])
        comparison['normality_improvement'] = comparison['transformed_stats']['jarque_bera_p'] - comparison['original_stats']['jarque_bera_p']

        return comparison

    def plot_transformation_comparison(self, original: pd.Series, transformed: pd.Series,
                                     transformation_name: str = "Transformed",
                                     figsize: tuple = (15, 10)) -> None:
        """
        Create comprehensive plots comparing original and transformed data.

        Parameters:
        -----------
        original : pd.Series
            Original data series
        transformed : pd.Series
            Transformed data series
        transformation_name : str
            Name of the transformation for labeling
        figsize : tuple
            Figure size for the plots
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Transformation Comparison: {transformation_name}', fontsize=16, fontweight='bold')

        # Time series plots
        axes[0, 0].plot(original.index, original.values, label='Original', alpha=0.7)
        axes[0, 0].set_title('Original Time Series')
        axes[0, 0].set_xlabel('Index')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(transformed.index, transformed.values, label='Transformed',
                       color='orange', alpha=0.7)
        axes[0, 1].set_title(f'{transformation_name} Time Series')
        axes[0, 1].set_xlabel('Index')
        axes[0, 1].set_ylabel('Transformed Value')
        axes[0, 1].grid(True, alpha=0.3)

        # Histograms
        axes[0, 2].hist(original.dropna(), bins=30, alpha=0.7, label='Original', density=True)
        axes[0, 2].hist(transformed.dropna(), bins=30, alpha=0.7, label='Transformed', density=True)
        axes[0, 2].set_title('Distribution Comparison')
        axes[0, 2].set_xlabel('Value')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Q-Q plots
        stats.probplot(original.dropna(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Original Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)

        stats.probplot(transformed.dropna(), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f'{transformation_name} Q-Q Plot')
        axes[1, 1].grid(True, alpha=0.3)

        # Box plots
        box_data = [original.dropna(), transformed.dropna()]
        box_labels = ['Original', transformation_name]
        axes[1, 2].boxplot(box_data, labels=box_labels)
        axes[1, 2].set_title('Box Plot Comparison')
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def recommend_transformation(self, data: pd.Series, column_name: str = "data") -> str:
        """
        Recommend the best transformation based on data characteristics.

        Parameters:
        -----------
        data : pd.Series
            Data series to analyze
        column_name : str
            Name of the column for reporting

        Returns:
        --------
        str
            Recommended transformation with explanation
        """
        # Calculate key statistics
        skewness = data.skew()
        has_zeros = (data == 0).any()
        has_negatives = (data < 0).any()
        is_proportion = (data >= 0).all() and (data <= 1).all()
        cv = data.std() / data.mean() if data.mean() != 0 else float('inf')

        recommendations = []

        # Decision logic
        if is_proportion:
            recommendations.append("Arcsine transformation - Data appears to be proportions")
        elif has_negatives:
            if abs(skewness) > 1:
                recommendations.append("Cube root transformation - Handles negative values and high skewness")
            else:
                recommendations.append("No transformation needed - Data has negative values with acceptable skewness")
        elif has_zeros:
            if skewness > 1:
                recommendations.append("Square root transformation with small constant - Handles zeros and right skew")
            else:
                recommendations.append("Log transformation with small constant - Handles zeros")
        elif skewness > 2:
            recommendations.append("Box-Cox transformation - Very high skewness, needs optimal transformation")
        elif skewness > 1:
            recommendations.append("Log transformation - Moderate to high right skewness")
        elif skewness > 0.5:
            recommendations.append("Square root transformation - Mild right skewness")
        elif cv > 2:
            recommendations.append("Fourth root transformation - Very high variance")
        else:
            recommendations.append("No transformation needed - Data appears well-behaved")

        # Additional considerations
        if data.min() > 0 and cv > 1:
            recommendations.append("Alternative: Log transformation for variance stabilization")

        result = f"\n=== Transformation Recommendation for '{column_name}' ===\n"
        result += f"Data characteristics:\n"
        result += f"  - Skewness: {skewness:.3f}\n"
        result += f"  - Coefficient of Variation: {cv:.3f}\n"
        result += f"  - Has zeros: {has_zeros}\n"
        result += f"  - Has negatives: {has_negatives}\n"
        result += f"  - Is proportion data: {is_proportion}\n\n"
        result += f"Primary recommendation: {recommendations[0]}\n"

        if len(recommendations) > 1:
            result += f"Alternative options:\n"
            for rec in recommendations[1:]:
                result += f"  - {rec}\n"

        return result

    def test_multiple_transformations(self, data: pd.Series, transformations: List[str] = None) -> pd.DataFrame:
        """
        Test multiple transformations and compare their effectiveness.

        Parameters:
        -----------
        data : pd.Series
            Data series to transform
        transformations : list of str, optional
            List of transformations to test. If None, tests all applicable ones.

        Returns:
        --------
        pd.DataFrame
            Comparison results for all transformations
        """
        if transformations is None:
            transformations = ['log', 'sqrt', 'cube_root', 'fourth_root', 'boxcox']

        # Create temporary dataframe for transformer
        temp_df = pd.DataFrame({'data': data})
        transformer = DataTransformer(temp_df)

        results = []

        for trans_type in transformations:
            try:
                if trans_type == 'log' and (data <= 0).any():
                    # Try with small constant
                    transformed = transformer.log_transformation('data', constant=abs(data.min()) + 1)
                elif trans_type == 'arcsine' and not ((data >= 0).all() and (data <= 1).all()):
                    # Scale first
                    transformed = transformer.arcsine_transformation('data', scale_first=True)
                elif trans_type == 'inverse' and (data == 0).any():
                    # Skip if zeros present
                    continue
                else:
                    transformed = transformer.apply_transformation('data', trans_type)

                # Calculate metrics
                original_jb_p = stats.jarque_bera(data.dropna())[1]
                transformed_jb_p = stats.jarque_bera(transformed.dropna())[1]

                results.append({
                    'Transformation': trans_type,
                    'Original_Skew': data.skew(),
                    'Transformed_Skew': transformed.skew(),
                    'Skew_Reduction': abs(data.skew()) - abs(transformed.skew()),
                    'Original_JB_p': original_jb_p,
                    'Transformed_JB_p': transformed_jb_p,
                    'Normality_Improvement': transformed_jb_p - original_jb_p,
                    'Variance_Ratio': transformed.var() / data.var()
                })

            except Exception as e:
                print(f"Could not apply {trans_type} transformation: {str(e)}")

        results_df = pd.DataFrame(results)

        if not results_df.empty:
            # Sort by skewness reduction (best first)
            results_df = results_df.sort_values('Skew_Reduction', ascending=False)
            print(f"\n=== Transformation Comparison Results ===")
            print(f"Best transformation for skewness reduction: {results_df.iloc[0]['Transformation']}")
            print(f"Best transformation for normality: {results_df.loc[results_df['Transformed_JB_p'].idxmax(), 'Transformation']}")

        return results_df

def demonstrate_transformations():
    """
    Demonstrate the usage of the DataTransformer class with sample data.
    """
    print("=== Data Transformations Demo ===\n")

    # Create sample data with different characteristics
    np.random.seed(42)

    # Exponential data (right-skewed)
    exp_data = np.random.exponential(2, 100)

    # Count data
    count_data = np.random.poisson(5, 100)

    # Proportion data
    prop_data = np.random.beta(2, 5, 100)

    # Create DataFrame
    df = pd.DataFrame({
        'exponential_data': exp_data,
        'count_data': count_data,
        'proportion_data': prop_data,
        'time_index': pd.date_range('2020-01-01', periods=100, freq='D')
    })

    # Initialize transformer
    transformer = DataTransformer(df)
    analyzer = TransformationAnalyzer()

    print("1. Testing Log Transformation on Exponential Data:")
    try:
        log_transformed = transformer.log_transformation('exponential_data')
        print(f"   Original skewness: {df['exponential_data'].skew():.3f}")
        print(f"   Transformed skewness: {log_transformed.skew():.3f}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n2. Testing Square Root Transformation on Count Data:")
    sqrt_transformed = transformer.sqrt_transformation('count_data')
    print(f"   Original skewness: {df['count_data'].skew():.3f}")
    print(f"   Transformed skewness: {sqrt_transformed.skew():.3f}")

    print("\n3. Testing Arcsine Transformation on Proportion Data:")
    arcsine_transformed = transformer.arcsine_transformation('proportion_data')
    print(f"   Original skewness: {df['proportion_data'].skew():.3f}")
    print(f"   Transformed skewness: {arcsine_transformed.skew():.3f}")

    print("\n4. Box-Cox Transformation with Optimization:")
    boxcox_transformed, lambda_val = transformer.boxcox_transformation('exponential_data')
    print(f"   Optimal lambda: {lambda_val:.3f}")
    print(f"   Original skewness: {df['exponential_data'].skew():.3f}")
    print(f"   Transformed skewness: {boxcox_transformed.skew():.3f}")

    print("\n5. Transformation Summary:")
    summary = transformer.get_transformation_summary()
    if not summary.empty:
        print(summary.to_string(index=False))

    print("\n6. Transformation Recommendation:")
    recommendation = analyzer.recommend_transformation(df['exponential_data'], 'exponential_data')
    print(recommendation)

    print("\nDemo completed successfully!")

if __name__ == "__main__":
    demonstrate_transformations()