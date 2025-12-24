"""
Data Cleaning Module for Financial and Economic Data

This module provides comprehensive data cleaning functions for:
- Qualitative data cleaning using regular expressions (regex)
- Quantitative data imputation using various interpolation methods
- Data validation and quality assessment
- Text processing and standardization

Dependencies: pandas, numpy, re
"""

import pandas as pd
import numpy as np
import re
import warnings
from typing import Optional, List, Dict, Any, Union, Callable
from datetime import datetime


class RegexGuide:
    """
    Comprehensive guide for regular expressions used in data cleaning
    """

    # Common patterns
    PATTERNS = {
        # Financial patterns
        'currency_symbols': r'[$£€¥₹₽¢₩₪₦₡₨₫₴₸₵₲₱₭₦₹₩₡]',
        'currency_codes': r'\b[A-Z]{3}\b',  # USD, EUR, GBP, etc.
        'percentage': r'\d+\.?\d*%',
        'decimal_numbers': r'-?\d+\.?\d*',
        'integers': r'-?\d+',
        'scientific_notation': r'-?\d+\.?\d*[eE][-+]?\d+',

        # Text cleaning patterns
        'whitespace': r'\s+',
        'leading_trailing_spaces': r'^\s+|\s+$',
        'multiple_spaces': r'\s{2,}',
        'special_chars': r'[^a-zA-Z0-9\s]',
        'punctuation': r'[^\w\s]',
        'digits_only': r'\d+',
        'letters_only': r'[a-zA-Z]+',
        'alphanumeric': r'[a-zA-Z0-9]+',

        # Date patterns
        'date_mmddyyyy': r'\d{1,2}[/-]\d{1,2}[/-]\d{4}',
        'date_ddmmyyyy': r'\d{1,2}[/-]\d{1,2}[/-]\d{4}',
        'date_yyyymmdd': r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
        'date_iso': r'\d{4}-\d{2}-\d{2}',

        # Business/Financial specific
        'stock_symbols': r'\b[A-Z]{1,5}\b',
        'phone_numbers': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'email_addresses': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'urls': r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',

        # Data quality patterns
        'missing_indicators': r'\b(N/A|NA|NULL|null|missing|Missing|MISSING|-|--)\b',
        'placeholder_text': r'\b(TBD|TBC|TODO|PLACEHOLDER)\b',
        'error_indicators': r'\b(ERROR|Error|error|#ERROR|#DIV/0!|#N/A|#NULL!)\b',
    }

    @classmethod
    def get_pattern(cls, pattern_name: str) -> str:
        """Get a regex pattern by name"""
        return cls.PATTERNS.get(pattern_name, '')

    @classmethod
    def list_patterns(cls) -> List[str]:
        """List all available pattern names"""
        return list(cls.PATTERNS.keys())

    @classmethod
    def pattern_examples(cls) -> Dict[str, str]:
        """Get examples of what each pattern matches"""
        examples = {
            'currency_symbols': 'Matches: $, £, €, ¥, etc.',
            'currency_codes': 'Matches: USD, EUR, GBP (3-letter codes)',
            'percentage': 'Matches: 5%, 10.5%, 0.25%',
            'decimal_numbers': 'Matches: 123.45, -67.89, 0.123',
            'integers': 'Matches: 123, -456, 0',
            'scientific_notation': 'Matches: 1.23e10, -4.56E-7',
            'date_mmddyyyy': 'Matches: 12/31/2023, 1-15-2023',
            'date_yyyymmdd': 'Matches: 2023/12/31, 2023-01-15',
            'stock_symbols': 'Matches: AAPL, MSFT, GOOGL',
            'phone_numbers': 'Matches: 123-456-7890, 123.456.7890',
            'email_addresses': 'Matches: user@example.com',
            'missing_indicators': 'Matches: N/A, NULL, missing, -',
            'error_indicators': 'Matches: ERROR, #DIV/0!, #N/A'
        }
        return examples


class QualitativeDataCleaner:
    """
    Class for cleaning qualitative (text/categorical) data using regular expressions
    """

    def __init__(self):
        self.regex_guide = RegexGuide()
        self.cleaning_log = []

    def clean_text_column(self, df: pd.DataFrame, column: str,
                         operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Clean a text column using a series of regex operations

        Args:
            df: DataFrame containing the column to clean
            column: Name of column to clean
            operations: List of cleaning operations, each dict should contain:
                       - 'operation': 'remove', 'replace', 'extract', or 'standardize'
                       - 'pattern': regex pattern or pattern name from RegexGuide
                       - 'replacement': replacement text (for 'replace' operation)
                       - 'flags': regex flags (optional)

        Returns:
            DataFrame with cleaned column
        """
        df_clean = df.copy()

        for op in operations:
            operation = op.get('operation')
            pattern = op.get('pattern')
            replacement = op.get('replacement', '')
            flags = op.get('flags', 0)

            # Get pattern from guide if it's a pattern name
            if pattern in self.regex_guide.PATTERNS:
                pattern = self.regex_guide.get_pattern(pattern)

            if operation == 'remove':
                df_clean[column] = df_clean[column].str.replace(pattern, '', flags=flags, regex=True)
                self.cleaning_log.append(f"Removed pattern '{pattern}' from column '{column}'")

            elif operation == 'replace':
                df_clean[column] = df_clean[column].str.replace(pattern, replacement, flags=flags, regex=True)
                self.cleaning_log.append(f"Replaced pattern '{pattern}' with '{replacement}' in column '{column}'")

            elif operation == 'extract':
                df_clean[f"{column}_extracted"] = df_clean[column].str.extract(f'({pattern})', flags=flags)
                self.cleaning_log.append(f"Extracted pattern '{pattern}' from column '{column}' to new column")

            elif operation == 'standardize':
                df_clean[column] = self._standardize_text(df_clean[column], pattern, replacement)
                self.cleaning_log.append(f"Standardized column '{column}' using pattern '{pattern}'")

        return df_clean

    def remove_currency_symbols(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Remove currency symbols from specified columns"""
        df_clean = df.copy()
        pattern = self.regex_guide.get_pattern('currency_symbols')

        for col in columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.replace(pattern, '', regex=True)
                self.cleaning_log.append(f"Removed currency symbols from column '{col}'")

        return df_clean

    def extract_numbers_from_text(self, df: pd.DataFrame, column: str,
                                 new_column: Optional[str] = None) -> pd.DataFrame:
        """Extract numeric values from text columns"""
        df_clean = df.copy()
        new_col = new_column or f"{column}_numeric"

        # Extract decimal numbers (including negative)
        pattern = self.regex_guide.get_pattern('decimal_numbers')
        df_clean[new_col] = df_clean[column].astype(str).str.extract(f'({pattern})')
        df_clean[new_col] = pd.to_numeric(df_clean[new_col], errors='coerce')

        self.cleaning_log.append(f"Extracted numbers from column '{column}' to '{new_col}'")
        return df_clean

    def standardize_missing_values(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Standardize various missing value indicators to NaN"""
        df_clean = df.copy()
        columns = columns or df_clean.columns.tolist()

        pattern = self.regex_guide.get_pattern('missing_indicators')

        for col in columns:
            if col in df_clean.columns:
                # Replace missing indicators with NaN
                df_clean[col] = df_clean[col].astype(str).str.replace(pattern, '', regex=True, flags=re.IGNORECASE)
                df_clean[col] = df_clean[col].replace('', np.nan)
                self.cleaning_log.append(f"Standardized missing values in column '{col}'")

        return df_clean

    def clean_percentage_values(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Clean percentage values and convert to decimal format"""
        df_clean = df.copy()

        for col in columns:
            if col in df_clean.columns:
                # Extract percentage values
                df_clean[f"{col}_cleaned"] = df_clean[col].astype(str).str.extract(r'(\d+\.?\d*)%')
                df_clean[f"{col}_cleaned"] = pd.to_numeric(df_clean[f"{col}_cleaned"], errors='coerce') / 100
                self.cleaning_log.append(f"Cleaned percentage values in column '{col}'")

        return df_clean

    def standardize_date_formats(self, df: pd.DataFrame, column: str,
                                output_format: str = '%Y-%m-%d') -> pd.DataFrame:
        """Standardize various date formats to a consistent format"""
        df_clean = df.copy()

        # Try to parse various date formats
        date_patterns = [
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',  # MM/DD/YYYY or DD/MM/YYYY
            r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY/MM/DD
        ]

        # Convert to datetime
        df_clean[f"{column}_standardized"] = pd.to_datetime(df_clean[column],
                                                           errors='coerce',
                                                           infer_datetime_format=True)

        # Format as string
        df_clean[f"{column}_standardized"] = df_clean[f"{column}_standardized"].dt.strftime(output_format)

        self.cleaning_log.append(f"Standardized date format in column '{column}'")
        return df_clean

    def clean_categorical_values(self, df: pd.DataFrame, column: str,
                               mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Clean and standardize categorical values"""
        df_clean = df.copy()

        # Basic cleaning: strip whitespace, standardize case
        df_clean[column] = df_clean[column].astype(str).str.strip().str.title()

        # Apply custom mapping if provided
        if mapping:
            df_clean[column] = df_clean[column].replace(mapping)

        self.cleaning_log.append(f"Cleaned categorical values in column '{column}'")
        return df_clean

    def remove_outlier_text_lengths(self, df: pd.DataFrame, column: str,
                                   method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
        """Remove rows with text lengths that are outliers"""
        df_clean = df.copy()

        # Calculate text lengths
        text_lengths = df_clean[column].astype(str).str.len()

        if method == 'iqr':
            Q1 = text_lengths.quantile(0.25)
            Q3 = text_lengths.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR

            mask = (text_lengths >= lower_bound) & (text_lengths <= upper_bound)
            df_clean = df_clean[mask]

            self.cleaning_log.append(f"Removed text length outliers from column '{column}' using IQR method")

        return df_clean

    def validate_data_patterns(self, df: pd.DataFrame, column: str,
                             expected_pattern: str) -> Dict[str, Any]:
        """Validate that data in a column matches expected pattern"""
        pattern = expected_pattern
        if expected_pattern in self.regex_guide.PATTERNS:
            pattern = self.regex_guide.get_pattern(expected_pattern)

        matches = df[column].astype(str).str.match(pattern, na=False)

        validation_results = {
            'total_rows': len(df),
            'matching_rows': matches.sum(),
            'non_matching_rows': (~matches).sum(),
            'match_percentage': (matches.sum() / len(df)) * 100,
            'non_matching_values': df.loc[~matches, column].unique().tolist()[:10]  # Show first 10
        }

        return validation_results

    def _standardize_text(self, series: pd.Series, pattern: str, replacement: str) -> pd.Series:
        """Helper method to standardize text based on pattern"""
        return series.astype(str).str.replace(pattern, replacement, regex=True).str.strip()

    def get_cleaning_log(self) -> List[str]:
        """Get log of all cleaning operations performed"""
        return self.cleaning_log

    def clear_log(self):
        """Clear the cleaning log"""
        self.cleaning_log = []


class QuantitativeDataImputer:
    """
    Class for imputing missing values in quantitative data using various interpolation methods
    """

    def __init__(self):
        self.imputation_log = []
        self.available_methods = [
            'backfill', 'bfill', 'pad', 'ffill', 'nearest', 'zero',
            'slinear', 'quadratic', 'cubic', 'spline', 'barycentric',
            'polynomial', 'krogh', 'piecewise_polynomial', 'pchip', 'akima'
        ]

    def impute_missing_values(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                            method: str = 'linear', **kwargs) -> pd.DataFrame:
        """
        Impute missing values using specified interpolation method

        Args:
            df: DataFrame with missing values
            columns: List of columns to impute (if None, imputes all numeric columns)
            method: Interpolation method
            **kwargs: Additional parameters for pandas interpolate method

        Returns:
            DataFrame with imputed values
        """
        df_imputed = df.copy()

        if columns is None:
            columns = df_imputed.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col in df_imputed.columns:
                missing_count = df_imputed[col].isna().sum()

                if missing_count > 0:
                    if method in ['backfill', 'bfill', 'pad', 'ffill']:
                        df_imputed[col] = df_imputed[col].fillna(method=method)
                    else:
                        df_imputed[col] = df_imputed[col].interpolate(method=method, **kwargs)

                    self.imputation_log.append(
                        f"Imputed {missing_count} missing values in column '{col}' using method '{method}'"
                    )

        return df_imputed

    def forward_fill(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                    limit: Optional[int] = None) -> pd.DataFrame:
        """Forward fill missing values"""
        return self.impute_missing_values(df, columns, method='ffill', limit=limit)

    def backward_fill(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                     limit: Optional[int] = None) -> pd.DataFrame:
        """Backward fill missing values"""
        return self.impute_missing_values(df, columns, method='bfill', limit=limit)

    def linear_interpolation(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                           limit: Optional[int] = None, limit_direction: str = 'forward') -> pd.DataFrame:
        """Linear interpolation of missing values"""
        return self.impute_missing_values(df, columns, method='linear',
                                        limit=limit, limit_direction=limit_direction)

    def polynomial_interpolation(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                               order: int = 2, **kwargs) -> pd.DataFrame:
        """Polynomial interpolation of missing values"""
        return self.impute_missing_values(df, columns, method='polynomial',
                                        order=order, **kwargs)

    def spline_interpolation(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                           order: int = 3, **kwargs) -> pd.DataFrame:
        """Spline interpolation of missing values"""
        return self.impute_missing_values(df, columns, method='spline',
                                        order=order, **kwargs)

    def time_series_interpolation(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                                method: str = 'time', **kwargs) -> pd.DataFrame:
        """
        Interpolation for time series data

        Args:
            df: DataFrame with DatetimeIndex
            columns: Columns to interpolate
            method: 'time' for time-based interpolation
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            warnings.warn("DataFrame index should be DatetimeIndex for time series interpolation")

        return self.impute_missing_values(df, columns, method=method, **kwargs)

    def nearest_interpolation(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Nearest neighbor interpolation"""
        return self.impute_missing_values(df, columns, method='nearest')

    def zero_fill(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Fill missing values with zero"""
        df_filled = df.copy()
        columns = columns or df_filled.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col in df_filled.columns:
                missing_count = df_filled[col].isna().sum()
                if missing_count > 0:
                    df_filled[col] = df_filled[col].fillna(0)
                    self.imputation_log.append(f"Filled {missing_count} missing values with 0 in column '{col}'")

        return df_filled

    def mean_fill(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Fill missing values with column mean"""
        df_filled = df.copy()
        columns = columns or df_filled.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col in df_filled.columns:
                missing_count = df_filled[col].isna().sum()
                if missing_count > 0:
                    mean_value = df_filled[col].mean()
                    df_filled[col] = df_filled[col].fillna(mean_value)
                    self.imputation_log.append(f"Filled {missing_count} missing values with mean ({mean_value:.4f}) in column '{col}'")

        return df_filled

    def median_fill(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Fill missing values with column median"""
        df_filled = df.copy()
        columns = columns or df_filled.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col in df_filled.columns:
                missing_count = df_filled[col].isna().sum()
                if missing_count > 0:
                    median_value = df_filled[col].median()
                    df_filled[col] = df_filled[col].fillna(median_value)
                    self.imputation_log.append(f"Filled {missing_count} missing values with median ({median_value:.4f}) in column '{col}'")

        return df_filled

    def mode_fill(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Fill missing values with column mode (most frequent value)"""
        df_filled = df.copy()
        columns = columns or df_filled.columns.tolist()

        for col in columns:
            if col in df_filled.columns:
                missing_count = df_filled[col].isna().sum()
                if missing_count > 0:
                    mode_value = df_filled[col].mode().iloc[0] if not df_filled[col].mode().empty else np.nan
                    df_filled[col] = df_filled[col].fillna(mode_value)
                    self.imputation_log.append(f"Filled {missing_count} missing values with mode ({mode_value}) in column '{col}'")

        return df_filled

    def rolling_mean_fill(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                         window: int = 3, min_periods: int = 1) -> pd.DataFrame:
        """Fill missing values using rolling mean"""
        df_filled = df.copy()
        columns = columns or df_filled.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col in df_filled.columns:
                missing_count = df_filled[col].isna().sum()
                if missing_count > 0:
                    rolling_mean = df_filled[col].rolling(window=window, min_periods=min_periods, center=True).mean()
                    df_filled[col] = df_filled[col].fillna(rolling_mean)
                    self.imputation_log.append(f"Filled {missing_count} missing values with rolling mean (window={window}) in column '{col}'")

        return df_filled

    def get_missing_value_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get summary of missing values in DataFrame"""
        missing_summary = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum(),
            'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
            'Data_Type': df.dtypes
        })

        missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]
        missing_summary = missing_summary.sort_values('Missing_Count', ascending=False)

        return missing_summary

    def compare_imputation_methods(self, df: pd.DataFrame, column: str,
                                 methods: List[str], test_size: float = 0.2) -> pd.DataFrame:
        """
        Compare different imputation methods by introducing artificial missing values

        Args:
            df: DataFrame with complete data
            column: Column to test imputation methods on
            methods: List of imputation methods to compare
            test_size: Proportion of data to artificially make missing

        Returns:
            DataFrame with comparison results
        """
        results = []

        # Create test data with artificial missing values
        df_test = df.copy()
        n_missing = int(len(df_test) * test_size)
        missing_indices = np.random.choice(df_test.index, n_missing, replace=False)

        # Store original values for comparison
        original_values = df_test.loc[missing_indices, column].copy()
        df_test.loc[missing_indices, column] = np.nan

        for method in methods:
            try:
                df_imputed = self.impute_missing_values(df_test.copy(), [column], method=method)
                imputed_values = df_imputed.loc[missing_indices, column]

                # Calculate error metrics
                mae = np.mean(np.abs(original_values - imputed_values))
                rmse = np.sqrt(np.mean((original_values - imputed_values) ** 2))

                results.append({
                    'Method': method,
                    'MAE': mae,
                    'RMSE': rmse
                })
            except Exception as e:
                results.append({
                    'Method': method,
                    'MAE': np.nan,
                    'RMSE': np.nan,
                    'Error': str(e)
                })

        return pd.DataFrame(results)

    def list_available_methods(self) -> List[str]:
        """List all available interpolation methods"""
        return self.available_methods

    def get_imputation_log(self) -> List[str]:
        """Get log of all imputation operations performed"""
        return self.imputation_log

    def clear_log(self):
        """Clear the imputation log"""
        self.imputation_log = []


# Utility Functions
def detect_data_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detect and categorize columns by data type

    Returns:
        Dictionary with column categorization
    """
    categorization = {
        'numeric': df.select_dtypes(include=[np.number]).columns.tolist(),
        'datetime': df.select_dtypes(include=['datetime64']).columns.tolist(),
        'categorical': df.select_dtypes(include=['category']).columns.tolist(),
        'text': df.select_dtypes(include=['object']).columns.tolist(),
        'boolean': df.select_dtypes(include=['bool']).columns.tolist()
    }

    return categorization


def generate_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive data quality report

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with data quality metrics
    """
    report = {
        'basic_info': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum()
        },
        'missing_data': {
            'total_missing': df.isnull().sum().sum(),
            'missing_by_column': df.isnull().sum().to_dict(),
            'rows_with_missing': (df.isnull().any(axis=1)).sum(),
            'complete_rows': (~df.isnull().any(axis=1)).sum()
        },
        'data_types': detect_data_types(df),
        'duplicates': {
            'duplicate_rows': df.duplicated().sum(),
            'unique_rows': len(df) - df.duplicated().sum()
        }
    }

    # Add numeric column statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        report['numeric_summary'] = df[numeric_cols].describe().to_dict()

    return report


def suggest_cleaning_operations(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Suggest cleaning operations based on data analysis

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with suggested operations
    """
    suggestions = {
        'qualitative_cleaning': [],
        'quantitative_imputation': [],
        'general_cleaning': []
    }

    # Analyze each column
    for col in df.columns:
        col_data = df[col]

        # Check for missing values
        if col_data.isnull().sum() > 0:
            if col_data.dtype in ['object', 'category']:
                suggestions['qualitative_cleaning'].append(f"Handle missing values in '{col}' (categorical)")
            elif np.issubdtype(col_data.dtype, np.number):
                suggestions['quantitative_imputation'].append(f"Impute missing values in '{col}' using appropriate interpolation")

        # Check for potential currency symbols or percentages
        if col_data.dtype == 'object':
            sample_values = col_data.dropna().astype(str).head(100)
            if any('$' in str(val) or '€' in str(val) or '£' in str(val) for val in sample_values):
                suggestions['qualitative_cleaning'].append(f"Remove currency symbols from '{col}'")
            if any('%' in str(val) for val in sample_values):
                suggestions['qualitative_cleaning'].append(f"Clean percentage values in '{col}'")

        # Check for duplicates
        if df.duplicated().sum() > 0:
            suggestions['general_cleaning'].append("Remove duplicate rows")

    return suggestions


# Example usage and demonstration functions
def demonstrate_regex_cleaning():
    """Demonstrate regex-based text cleaning"""
    print("=== Regular Expression Cleaning Demo ===")

    # Create sample data
    sample_data = {
        'price': ['$100.50', '€250.75', '£89.99', '$1,234.56'],
        'percentage': ['10.5%', '25%', '3.14%', '0.5%'],
        'phone': ['123-456-7890', '(555) 123-4567', '555.987.6543', 'invalid'],
        'status': ['Active', 'inactive', 'PENDING', 'N/A']
    }

    df = pd.DataFrame(sample_data)
    print("Original Data:")
    print(df)

    cleaner = QualitativeDataCleaner()

    # Clean price column
    df_clean = cleaner.remove_currency_symbols(df, ['price'])
    df_clean = cleaner.extract_numbers_from_text(df_clean, 'price')

    # Clean percentage column
    df_clean = cleaner.clean_percentage_values(df_clean, ['percentage'])

    # Standardize missing values
    df_clean = cleaner.standardize_missing_values(df_clean, ['status'])

    print("\nCleaned Data:")
    print(df_clean)

    print("\nCleaning Log:")
    for log_entry in cleaner.get_cleaning_log():
        print(f"- {log_entry}")


def demonstrate_quantitative_imputation():
    """Demonstrate quantitative data imputation"""
    print("\n=== Quantitative Data Imputation Demo ===")

    # Create sample data with missing values
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=20, freq='D')
    data = {
        'date': dates,
        'price': [100 + i + np.random.normal(0, 5) for i in range(20)],
        'volume': [1000 + i*50 + np.random.normal(0, 100) for i in range(20)]
    }

    df = pd.DataFrame(data)

    # Introduce missing values
    df.loc[3:5, 'price'] = np.nan
    df.loc[10:12, 'volume'] = np.nan
    df.loc[15, 'price'] = np.nan

    print("Data with Missing Values:")
    print(df[df.isnull().any(axis=1)])

    imputer = QuantitativeDataImputer()

    # Get missing value summary
    print("\nMissing Value Summary:")
    print(imputer.get_missing_value_summary(df))

    # Try different imputation methods
    df_linear = imputer.linear_interpolation(df.copy(), ['price', 'volume'])
    df_spline = imputer.spline_interpolation(df.copy(), ['price', 'volume'])
    df_forward = imputer.forward_fill(df.copy(), ['price', 'volume'])

    print("\nImputation Log:")
    for log_entry in imputer.get_imputation_log():
        print(f"- {log_entry}")


if __name__ == "__main__":
    print("Data Cleaning Module")
    print("====================")

    # Show available regex patterns
    print("\nAvailable Regex Patterns:")
    guide = RegexGuide()
    for pattern_name in guide.list_patterns():
        print(f"  {pattern_name}: {guide.pattern_examples().get(pattern_name, 'No example available')}")

    # Show available imputation methods
    print(f"\nAvailable Imputation Methods:")
    imputer = QuantitativeDataImputer()
    for method in imputer.list_available_methods():
        print(f"  - {method}")

    # Run demonstrations
    demonstrate_regex_cleaning()
    demonstrate_quantitative_imputation()