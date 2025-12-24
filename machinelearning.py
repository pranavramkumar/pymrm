"""
Machine Learning Feature Engineering Module
==========================================

This module provides comprehensive feature engineering tools for machine learning including:
- Feature matrix and target vector construction
- Outlier detection and removal
- Feature dropping based on nulls, cardinality, and data leakage
- Multicollinearity detection and handling
- Class imbalance correction with sampling techniques
- Train-validation-test splitting

Author: Claude Code Assistant
Date: 2025-09-29
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, FactorAnalysis as SklearnFactorAnalysis
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer

# Linear models
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge, ElasticNet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Tree-based models
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                             ExtraTreesClassifier, ExtraTreesRegressor,
                             GradientBoostingClassifier, GradientBoostingRegressor,
                             VotingClassifier, VotingRegressor,
                             BaggingClassifier, BaggingRegressor,
                             AdaBoostClassifier, AdaBoostRegressor)

# Other classifiers
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR

# Metrics
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                            accuracy_score, precision_score, recall_score, f1_score,
                            mean_squared_error, mean_absolute_error, r2_score,
                            roc_curve, precision_recall_curve)
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
from collections import Counter, defaultdict
import itertools
import pickle
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize


@dataclass
class FeatureEngineeringConfig:
    """Configuration class for feature engineering parameters."""

    # Outlier detection
    outlier_method: str = 'iqr'  # 'iqr', 'zscore', 'isolation_forest', 'local_outlier'
    outlier_threshold: float = 3.0
    outlier_contamination: float = 0.1

    # Missing value handling
    max_null_ratio: float = 0.5  # Drop features with > 50% nulls
    null_imputation: str = 'median'  # 'mean', 'median', 'mode', 'constant', 'forward_fill'

    # Cardinality handling
    max_cardinality_ratio: float = 0.95  # Drop if unique/total > 95%
    min_cardinality_threshold: int = 2  # Drop if unique values < 2
    rare_category_threshold: float = 0.01  # Group categories with < 1% frequency

    # Multicollinearity
    correlation_threshold: float = 0.95
    vif_threshold: float = 10.0

    # Leakage detection
    target_correlation_threshold: float = 0.99
    future_data_check: bool = True

    # Sampling
    sampling_strategy: str = 'auto'  # 'auto', 'minority', 'majority', 'all'
    sampling_method: str = 'smote'  # 'random', 'smote', 'adasyn', 'borderline'

    # Splitting
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    time_series_split: bool = False

    # Dimensionality reduction
    apply_dimensionality_reduction: bool = False  # Whether to apply dim reduction
    dimensionality_reduction_method: str = 'pca'  # 'pca', 'fa', 'varclus'
    pca_variance_threshold: float = 0.95  # Cumulative variance to retain
    pca_n_components: Optional[int] = None  # Specific number of components
    fa_n_factors: Optional[int] = None  # Number of factors for FA
    fa_rotation: str = 'varimax'  # 'varimax', 'promax', 'none'
    fa_method: str = 'principal'  # 'principal', 'ml', 'minres'
    varclus_max_clusters: int = 10  # Maximum clusters for variable clustering
    varclus_min_correlation: float = 0.7  # Minimum within-cluster correlation

    # Model pipeline configuration
    numerical_imputer: str = 'simple'  # 'simple', 'knn'
    categorical_imputer: str = 'most_frequent'  # 'most_frequent', 'constant'
    numerical_scaler: str = 'standard'  # 'standard', 'minmax', 'robust'
    categorical_encoder: str = 'onehot'  # 'onehot', 'ordinal', 'label'
    handle_unknown_categories: str = 'ignore'  # 'ignore', 'error'

    # Model selection
    enable_cross_validation: bool = True
    cv_folds: int = 5
    scoring_metric: str = 'auto'  # 'auto', 'accuracy', 'roc_auc', 'f1', 'r2', 'mse'
    enable_hyperparameter_tuning: bool = False
    max_iter: int = 1000  # Maximum iterations for iterative algorithms


class OutlierDetector:
    """
    Comprehensive outlier detection and removal.
    """

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.outlier_indices = {}
        self.outlier_stats = {}

    def detect_outliers_iqr(self, data: pd.Series, multiplier: float = 1.5) -> np.ndarray:
        """Detect outliers using Interquartile Range method."""
        try:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            outliers = (data < lower_bound) | (data > upper_bound)
            return outliers.values

        except Exception as e:
            warnings.warn(f"IQR outlier detection failed: {str(e)}")
            return np.zeros(len(data), dtype=bool)

    def detect_outliers_zscore(self, data: pd.Series, threshold: float = 3.0) -> np.ndarray:
        """Detect outliers using Z-score method."""
        try:
            z_scores = np.abs(stats.zscore(data.dropna()))
            outliers = np.zeros(len(data), dtype=bool)

            # Map z-scores back to original data positions
            non_null_mask = data.notna()
            outliers[non_null_mask] = z_scores > threshold

            return outliers

        except Exception as e:
            warnings.warn(f"Z-score outlier detection failed: {str(e)}")
            return np.zeros(len(data), dtype=bool)

    def detect_outliers_isolation_forest(self, data: pd.DataFrame,
                                       contamination: float = 0.1) -> np.ndarray:
        """Detect outliers using Isolation Forest."""
        try:
            from sklearn.ensemble import IsolationForest

            # Select only numeric columns
            numeric_data = data.select_dtypes(include=[np.number])

            if numeric_data.empty:
                return np.zeros(len(data), dtype=bool)

            # Handle missing values
            numeric_data_filled = numeric_data.fillna(numeric_data.median())

            iso_forest = IsolationForest(contamination=contamination,
                                       random_state=self.config.random_state)
            outlier_labels = iso_forest.fit_predict(numeric_data_filled)

            # -1 indicates outlier, 1 indicates inlier
            outliers = outlier_labels == -1

            return outliers

        except ImportError:
            warnings.warn("scikit-learn not available for Isolation Forest")
            return np.zeros(len(data), dtype=bool)
        except Exception as e:
            warnings.warn(f"Isolation Forest outlier detection failed: {str(e)}")
            return np.zeros(len(data), dtype=bool)

    def detect_outliers_local_outlier_factor(self, data: pd.DataFrame,
                                            contamination: float = 0.1) -> np.ndarray:
        """Detect outliers using Local Outlier Factor."""
        try:
            from sklearn.neighbors import LocalOutlierFactor

            # Select only numeric columns
            numeric_data = data.select_dtypes(include=[np.number])

            if numeric_data.empty:
                return np.zeros(len(data), dtype=bool)

            # Handle missing values
            numeric_data_filled = numeric_data.fillna(numeric_data.median())

            lof = LocalOutlierFactor(contamination=contamination)
            outlier_labels = lof.fit_predict(numeric_data_filled)

            # -1 indicates outlier, 1 indicates inlier
            outliers = outlier_labels == -1

            return outliers

        except ImportError:
            warnings.warn("scikit-learn not available for Local Outlier Factor")
            return np.zeros(len(data), dtype=bool)
        except Exception as e:
            warnings.warn(f"Local Outlier Factor detection failed: {str(e)}")
            return np.zeros(len(data), dtype=bool)

    def detect_outliers(self, data: pd.DataFrame,
                       columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive outlier detection across multiple methods.

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        columns : List[str], optional
            Specific columns to analyze for outliers

        Returns:
        --------
        Dict with outlier detection results
        """
        try:
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns.tolist()

            results = {
                'method': self.config.outlier_method,
                'outlier_indices': {},
                'outlier_counts': {},
                'outlier_percentages': {},
                'total_outliers': 0,
                'columns_analyzed': columns,
                'detection_date': datetime.now()
            }

            if self.config.outlier_method == 'iqr':
                # Apply IQR method to each numeric column
                combined_outliers = np.zeros(len(data), dtype=bool)

                for col in columns:
                    if col in data.columns and data[col].dtype in [np.number]:
                        col_outliers = self.detect_outliers_iqr(
                            data[col],
                            multiplier=self.config.outlier_threshold
                        )
                        results['outlier_indices'][col] = np.where(col_outliers)[0].tolist()
                        results['outlier_counts'][col] = int(np.sum(col_outliers))
                        results['outlier_percentages'][col] = float(np.mean(col_outliers) * 100)

                        combined_outliers |= col_outliers

            elif self.config.outlier_method == 'zscore':
                # Apply Z-score method to each numeric column
                combined_outliers = np.zeros(len(data), dtype=bool)

                for col in columns:
                    if col in data.columns and data[col].dtype in [np.number]:
                        col_outliers = self.detect_outliers_zscore(
                            data[col],
                            threshold=self.config.outlier_threshold
                        )
                        results['outlier_indices'][col] = np.where(col_outliers)[0].tolist()
                        results['outlier_counts'][col] = int(np.sum(col_outliers))
                        results['outlier_percentages'][col] = float(np.mean(col_outliers) * 100)

                        combined_outliers |= col_outliers

            elif self.config.outlier_method == 'isolation_forest':
                # Apply Isolation Forest to entire dataset
                combined_outliers = self.detect_outliers_isolation_forest(
                    data[columns],
                    contamination=self.config.outlier_contamination
                )
                results['outlier_indices']['multivariate'] = np.where(combined_outliers)[0].tolist()
                results['outlier_counts']['multivariate'] = int(np.sum(combined_outliers))
                results['outlier_percentages']['multivariate'] = float(np.mean(combined_outliers) * 100)

            elif self.config.outlier_method == 'local_outlier':
                # Apply Local Outlier Factor to entire dataset
                combined_outliers = self.detect_outliers_local_outlier_factor(
                    data[columns],
                    contamination=self.config.outlier_contamination
                )
                results['outlier_indices']['multivariate'] = np.where(combined_outliers)[0].tolist()
                results['outlier_counts']['multivariate'] = int(np.sum(combined_outliers))
                results['outlier_percentages']['multivariate'] = float(np.mean(combined_outliers) * 100)

            else:
                raise ValueError(f"Unknown outlier method: {self.config.outlier_method}")

            results['total_outliers'] = int(np.sum(combined_outliers))
            results['total_percentage'] = float(np.mean(combined_outliers) * 100)
            results['outlier_mask'] = combined_outliers

            # Store for later use
            self.outlier_indices = results['outlier_indices']
            self.outlier_stats = {
                'total_outliers': results['total_outliers'],
                'total_percentage': results['total_percentage'],
                'method': self.config.outlier_method
            }

            return results

        except Exception as e:
            return {'error': str(e), 'method': self.config.outlier_method}

    def remove_outliers(self, data: pd.DataFrame,
                       outlier_results: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Remove detected outliers from dataset.

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        outlier_results : Dict, optional
            Previous outlier detection results

        Returns:
        --------
        Tuple of (cleaned_data, removal_summary)
        """
        try:
            if outlier_results is None:
                outlier_results = self.detect_outliers(data)

            if 'error' in outlier_results:
                return data, {'error': 'Outlier detection failed'}

            outlier_mask = outlier_results.get('outlier_mask', np.zeros(len(data), dtype=bool))

            # Remove outliers
            cleaned_data = data[~outlier_mask].copy()

            removal_summary = {
                'original_rows': len(data),
                'outliers_removed': int(np.sum(outlier_mask)),
                'remaining_rows': len(cleaned_data),
                'removal_percentage': float(np.mean(outlier_mask) * 100),
                'method_used': outlier_results.get('method'),
                'removal_date': datetime.now()
            }

            return cleaned_data, removal_summary

        except Exception as e:
            return data, {'error': str(e)}

    def visualize_outliers(self, data: pd.DataFrame, columns: List[str] = None,
                          save_path: str = None) -> Dict[str, Any]:
        """Create visualizations for outlier analysis."""
        try:
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns.tolist()[:6]  # Limit to 6 columns

            n_cols = min(3, len(columns))
            n_rows = (len(columns) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)

            for idx, col in enumerate(columns):
                row = idx // n_cols
                col_idx = idx % n_cols

                if col in data.columns:
                    # Box plot to show outliers
                    data[col].plot.box(ax=axes[row, col_idx])
                    axes[row, col_idx].set_title(f'Outliers in {col}')
                    axes[row, col_idx].set_ylabel('Values')

            # Hide empty subplots
            for idx in range(len(columns), n_rows * n_cols):
                row = idx // n_cols
                col_idx = idx % n_cols
                axes[row, col_idx].set_visible(False)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            result = {
                'visualization_type': 'outlier_analysis',
                'columns_plotted': columns,
                'save_path': save_path,
                'visualization_date': datetime.now()
            }

            return result

        except Exception as e:
            return {'error': str(e)}


class FeatureSelector:
    """
    Advanced feature selection and cleaning.
    """

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.dropped_features = {}
        self.feature_stats = {}

    def analyze_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing value patterns in the dataset."""
        try:
            missing_stats = {}

            for col in data.columns:
                null_count = data[col].isnull().sum()
                null_ratio = null_count / len(data)

                missing_stats[col] = {
                    'null_count': int(null_count),
                    'null_ratio': float(null_ratio),
                    'dtype': str(data[col].dtype),
                    'total_count': len(data)
                }

            # Sort by null ratio descending
            sorted_stats = dict(sorted(missing_stats.items(),
                                     key=lambda x: x[1]['null_ratio'],
                                     reverse=True))

            result = {
                'missing_value_stats': sorted_stats,
                'columns_with_nulls': [col for col, stats in missing_stats.items()
                                     if stats['null_ratio'] > 0],
                'high_null_columns': [col for col, stats in missing_stats.items()
                                    if stats['null_ratio'] > self.config.max_null_ratio],
                'analysis_date': datetime.now()
            }

            return result

        except Exception as e:
            return {'error': str(e)}

    def analyze_cardinality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cardinality of categorical and numeric features."""
        try:
            cardinality_stats = {}

            for col in data.columns:
                unique_count = data[col].nunique()
                total_count = len(data)
                cardinality_ratio = unique_count / total_count if total_count > 0 else 0

                # Get value counts for categorical analysis
                value_counts = data[col].value_counts()
                most_frequent_ratio = value_counts.iloc[0] / total_count if len(value_counts) > 0 else 0

                cardinality_stats[col] = {
                    'unique_count': int(unique_count),
                    'total_count': int(total_count),
                    'cardinality_ratio': float(cardinality_ratio),
                    'most_frequent_value': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'most_frequent_ratio': float(most_frequent_ratio),
                    'dtype': str(data[col].dtype),
                    'is_categorical': data[col].dtype == 'object' or data[col].dtype.name == 'category'
                }

            # Identify problematic columns
            high_cardinality_cols = [col for col, stats in cardinality_stats.items()
                                   if stats['cardinality_ratio'] > self.config.max_cardinality_ratio]

            low_cardinality_cols = [col for col, stats in cardinality_stats.items()
                                  if stats['unique_count'] < self.config.min_cardinality_threshold]

            result = {
                'cardinality_stats': cardinality_stats,
                'high_cardinality_columns': high_cardinality_cols,
                'low_cardinality_columns': low_cardinality_cols,
                'analysis_date': datetime.now()
            }

            return result

        except Exception as e:
            return {'error': str(e)}

    def detect_leaky_features(self, data: pd.DataFrame,
                            target_column: str) -> Dict[str, Any]:
        """Detect potential data leakage in features."""
        try:
            if target_column not in data.columns:
                return {'error': f'Target column {target_column} not found'}

            leaky_features = {}
            suspicious_features = {}

            target_data = data[target_column]

            for col in data.columns:
                if col == target_column:
                    continue

                # Skip non-numeric columns for correlation analysis
                if data[col].dtype not in [np.number] and target_data.dtype not in [np.number]:
                    continue

                try:
                    # Calculate correlation with target
                    if data[col].dtype in [np.number] and target_data.dtype in [np.number]:
                        correlation, p_value = pearsonr(data[col].dropna(),
                                                      target_data[data[col].notna()])
                        abs_correlation = abs(correlation)

                        feature_info = {
                            'correlation': float(correlation),
                            'abs_correlation': float(abs_correlation),
                            'p_value': float(p_value),
                            'suspicious_reason': []
                        }

                        # Check for perfect or near-perfect correlation
                        if abs_correlation > self.config.target_correlation_threshold:
                            feature_info['suspicious_reason'].append('high_correlation')
                            leaky_features[col] = feature_info
                        elif abs_correlation > 0.9:
                            feature_info['suspicious_reason'].append('very_high_correlation')
                            suspicious_features[col] = feature_info

                        # Check for potential future information leakage
                        if self.config.future_data_check:
                            # Look for features that might contain future information
                            future_indicators = ['future', 'next', 'after', 'post', 'outcome', 'result']
                            col_lower = col.lower()

                            if any(indicator in col_lower for indicator in future_indicators):
                                feature_info['suspicious_reason'].append('potential_future_data')
                                if col not in leaky_features:
                                    suspicious_features[col] = feature_info

                        # Check for duplicate or derived features
                        if col.lower().replace('_', '').replace(' ', '') == target_column.lower().replace('_', '').replace(' ', ''):
                            feature_info['suspicious_reason'].append('duplicate_target')
                            leaky_features[col] = feature_info

                except Exception:
                    continue  # Skip problematic columns

            result = {
                'leaky_features': leaky_features,
                'suspicious_features': suspicious_features,
                'target_column': target_column,
                'correlation_threshold': self.config.target_correlation_threshold,
                'analysis_date': datetime.now()
            }

            return result

        except Exception as e:
            return {'error': str(e)}

    def detect_multicollinearity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect multicollinearity between features."""
        try:
            # Select only numeric columns
            numeric_data = data.select_dtypes(include=[np.number])

            if numeric_data.empty:
                return {'error': 'No numeric columns found for multicollinearity analysis'}

            # Calculate correlation matrix
            correlation_matrix = numeric_data.corr()

            # Find highly correlated pairs
            highly_correlated_pairs = []
            correlation_threshold = self.config.correlation_threshold

            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    col1 = correlation_matrix.columns[i]
                    col2 = correlation_matrix.columns[j]
                    corr_value = correlation_matrix.iloc[i, j]

                    if abs(corr_value) > correlation_threshold:
                        highly_correlated_pairs.append({
                            'feature1': col1,
                            'feature2': col2,
                            'correlation': float(corr_value),
                            'abs_correlation': float(abs(corr_value))
                        })

            # Calculate VIF (Variance Inflation Factor) if possible
            vif_scores = {}
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor

                # Handle missing values
                numeric_data_clean = numeric_data.fillna(numeric_data.median())

                # Calculate VIF for each feature
                for i, col in enumerate(numeric_data_clean.columns):
                    try:
                        vif = variance_inflation_factor(numeric_data_clean.values, i)
                        vif_scores[col] = float(vif) if not np.isinf(vif) and not np.isnan(vif) else float('inf')
                    except Exception:
                        vif_scores[col] = float('inf')

            except ImportError:
                warnings.warn("statsmodels not available for VIF calculation")

            # Identify features to drop based on correlation and VIF
            features_to_drop = set()

            # From correlation pairs, drop the feature with higher mean correlation
            for pair in highly_correlated_pairs:
                col1, col2 = pair['feature1'], pair['feature2']

                # Calculate mean correlation with other features
                mean_corr1 = correlation_matrix[col1].abs().mean()
                mean_corr2 = correlation_matrix[col2].abs().mean()

                # Drop feature with higher mean correlation
                if mean_corr1 > mean_corr2:
                    features_to_drop.add(col1)
                else:
                    features_to_drop.add(col2)

            # Add features with high VIF
            high_vif_features = [col for col, vif in vif_scores.items()
                               if vif > self.config.vif_threshold]
            features_to_drop.update(high_vif_features)

            result = {
                'correlation_matrix': correlation_matrix.to_dict(),
                'highly_correlated_pairs': highly_correlated_pairs,
                'vif_scores': vif_scores,
                'high_vif_features': high_vif_features,
                'recommended_drops': list(features_to_drop),
                'correlation_threshold': correlation_threshold,
                'vif_threshold': self.config.vif_threshold,
                'analysis_date': datetime.now()
            }

            return result

        except Exception as e:
            return {'error': str(e)}

    def drop_problematic_features(self, data: pd.DataFrame,
                                target_column: str = None,
                                custom_drops: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Comprehensive feature dropping based on various criteria.

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        target_column : str, optional
            Target column name for leakage detection
        custom_drops : List[str], optional
            Additional columns to drop

        Returns:
        --------
        Tuple of (cleaned_data, drop_summary)
        """
        try:
            cleaned_data = data.copy()
            drop_summary = {
                'dropped_features': {},
                'original_features': list(data.columns),
                'original_feature_count': len(data.columns),
                'drop_reasons': defaultdict(list)
            }

            # 1. Drop features with high null ratios
            missing_analysis = self.analyze_missing_values(cleaned_data)
            if 'high_null_columns' in missing_analysis:
                high_null_cols = missing_analysis['high_null_columns']
                for col in high_null_cols:
                    if col in cleaned_data.columns:
                        null_ratio = missing_analysis['missing_value_stats'][col]['null_ratio']
                        drop_summary['dropped_features'][col] = f'High null ratio: {null_ratio:.3f}'
                        drop_summary['drop_reasons']['high_nulls'].append(col)

                cleaned_data = cleaned_data.drop(columns=high_null_cols, errors='ignore')

            # 2. Drop features with problematic cardinality
            cardinality_analysis = self.analyze_cardinality(cleaned_data)
            if 'high_cardinality_columns' in cardinality_analysis:
                high_card_cols = cardinality_analysis['high_cardinality_columns']
                for col in high_card_cols:
                    if col in cleaned_data.columns:
                        card_ratio = cardinality_analysis['cardinality_stats'][col]['cardinality_ratio']
                        drop_summary['dropped_features'][col] = f'High cardinality: {card_ratio:.3f}'
                        drop_summary['drop_reasons']['high_cardinality'].append(col)

                cleaned_data = cleaned_data.drop(columns=high_card_cols, errors='ignore')

            if 'low_cardinality_columns' in cardinality_analysis:
                low_card_cols = cardinality_analysis['low_cardinality_columns']
                for col in low_card_cols:
                    if col in cleaned_data.columns and col != target_column:
                        unique_count = cardinality_analysis['cardinality_stats'][col]['unique_count']
                        drop_summary['dropped_features'][col] = f'Low cardinality: {unique_count} unique values'
                        drop_summary['drop_reasons']['low_cardinality'].append(col)

                cleaned_data = cleaned_data.drop(columns=low_card_cols, errors='ignore')

            # 3. Drop leaky features if target is specified
            if target_column and target_column in cleaned_data.columns:
                leakage_analysis = self.detect_leaky_features(cleaned_data, target_column)
                if 'leaky_features' in leakage_analysis:
                    leaky_cols = list(leakage_analysis['leaky_features'].keys())
                    for col in leaky_cols:
                        if col in cleaned_data.columns:
                            correlation = leakage_analysis['leaky_features'][col]['correlation']
                            drop_summary['dropped_features'][col] = f'Data leakage: correlation {correlation:.3f}'
                            drop_summary['drop_reasons']['data_leakage'].append(col)

                    cleaned_data = cleaned_data.drop(columns=leaky_cols, errors='ignore')

            # 4. Drop features with high multicollinearity
            multicollinearity_analysis = self.detect_multicollinearity(cleaned_data)
            if 'recommended_drops' in multicollinearity_analysis:
                multicoll_cols = multicollinearity_analysis['recommended_drops']
                for col in multicoll_cols:
                    if col in cleaned_data.columns and col != target_column:
                        if col in multicollinearity_analysis.get('vif_scores', {}):
                            vif = multicollinearity_analysis['vif_scores'][col]
                            drop_summary['dropped_features'][col] = f'High multicollinearity: VIF {vif:.2f}'
                        else:
                            drop_summary['dropped_features'][col] = 'High multicollinearity'
                        drop_summary['drop_reasons']['multicollinearity'].append(col)

                cleaned_data = cleaned_data.drop(columns=multicoll_cols, errors='ignore')

            # 5. Drop custom specified features
            if custom_drops:
                for col in custom_drops:
                    if col in cleaned_data.columns:
                        drop_summary['dropped_features'][col] = 'Custom drop request'
                        drop_summary['drop_reasons']['custom'].append(col)

                cleaned_data = cleaned_data.drop(columns=custom_drops, errors='ignore')

            # Finalize summary
            drop_summary.update({
                'final_features': list(cleaned_data.columns),
                'final_feature_count': len(cleaned_data.columns),
                'total_dropped': len(drop_summary['dropped_features']),
                'drop_percentage': len(drop_summary['dropped_features']) / len(data.columns) * 100,
                'cleaning_date': datetime.now()
            })

            # Store for later reference
            self.dropped_features = drop_summary['dropped_features']

            return cleaned_data, drop_summary

        except Exception as e:
            return data, {'error': str(e)}


class ClassImbalanceHandler:
    """
    Handle class imbalance through various sampling techniques.
    """

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.sampling_stats = {}

    def analyze_class_distribution(self, y: pd.Series) -> Dict[str, Any]:
        """Analyze target variable class distribution."""
        try:
            value_counts = y.value_counts()
            value_proportions = y.value_counts(normalize=True)

            # Calculate imbalance metrics
            majority_class = value_counts.index[0]
            minority_class = value_counts.index[-1]
            imbalance_ratio = value_counts.iloc[-1] / value_counts.iloc[0]

            result = {
                'class_counts': value_counts.to_dict(),
                'class_proportions': value_proportions.to_dict(),
                'total_samples': len(y),
                'n_classes': len(value_counts),
                'majority_class': majority_class,
                'minority_class': minority_class,
                'majority_count': int(value_counts.iloc[0]),
                'minority_count': int(value_counts.iloc[-1]),
                'imbalance_ratio': float(imbalance_ratio),
                'is_imbalanced': imbalance_ratio < 0.5,  # Consider imbalanced if minority < 50% of majority
                'analysis_date': datetime.now()
            }

            return result

        except Exception as e:
            return {'error': str(e)}

    def random_undersample(self, X: pd.DataFrame, y: pd.Series,
                          strategy: str = 'auto') -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Apply random undersampling to balance classes."""
        try:
            from sklearn.utils import resample

            # Combine X and y for resampling
            data_combined = pd.concat([X, y], axis=1)
            target_col = y.name if y.name else 'target'

            class_counts = y.value_counts()
            min_class_size = class_counts.min()

            balanced_data = []

            for class_label in class_counts.index:
                class_data = data_combined[data_combined[target_col] == class_label]

                if strategy == 'auto' or strategy == 'majority':
                    n_samples = min_class_size
                else:
                    n_samples = min(len(class_data), min_class_size)

                resampled_class = resample(class_data,
                                         n_samples=n_samples,
                                         random_state=self.config.random_state,
                                         replace=False)
                balanced_data.append(resampled_class)

            balanced_df = pd.concat(balanced_data, ignore_index=True)

            X_resampled = balanced_df.drop(columns=[target_col])
            y_resampled = balanced_df[target_col]

            sampling_info = {
                'method': 'random_undersample',
                'original_distribution': class_counts.to_dict(),
                'new_distribution': y_resampled.value_counts().to_dict(),
                'original_size': len(y),
                'new_size': len(y_resampled),
                'sampling_date': datetime.now()
            }

            return X_resampled, y_resampled, sampling_info

        except Exception as e:
            return X, y, {'error': str(e)}

    def random_oversample(self, X: pd.DataFrame, y: pd.Series,
                         strategy: str = 'auto') -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Apply random oversampling to balance classes."""
        try:
            from sklearn.utils import resample

            # Combine X and y for resampling
            data_combined = pd.concat([X, y], axis=1)
            target_col = y.name if y.name else 'target'

            class_counts = y.value_counts()
            max_class_size = class_counts.max()

            balanced_data = []

            for class_label in class_counts.index:
                class_data = data_combined[data_combined[target_col] == class_label]

                if strategy == 'auto' or strategy == 'minority':
                    n_samples = max_class_size
                else:
                    n_samples = max(len(class_data), max_class_size)

                resampled_class = resample(class_data,
                                         n_samples=n_samples,
                                         random_state=self.config.random_state,
                                         replace=True)
                balanced_data.append(resampled_class)

            balanced_df = pd.concat(balanced_data, ignore_index=True)

            X_resampled = balanced_df.drop(columns=[target_col])
            y_resampled = balanced_df[target_col]

            sampling_info = {
                'method': 'random_oversample',
                'original_distribution': class_counts.to_dict(),
                'new_distribution': y_resampled.value_counts().to_dict(),
                'original_size': len(y),
                'new_size': len(y_resampled),
                'sampling_date': datetime.now()
            }

            return X_resampled, y_resampled, sampling_info

        except Exception as e:
            return X, y, {'error': str(e)}

    def smote_oversample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Apply SMOTE (Synthetic Minority Oversampling Technique)."""
        try:
            from imblearn.over_sampling import SMOTE

            # SMOTE requires numeric features only
            numeric_features = X.select_dtypes(include=[np.number]).columns
            X_numeric = X[numeric_features]

            if X_numeric.empty:
                return self.random_oversample(X, y)

            # Handle missing values
            X_numeric_filled = X_numeric.fillna(X_numeric.median())

            smote = SMOTE(random_state=self.config.random_state,
                         sampling_strategy=self.config.sampling_strategy)

            X_resampled, y_resampled = smote.fit_resample(X_numeric_filled, y)

            # Convert back to DataFrame
            X_resampled = pd.DataFrame(X_resampled, columns=X_numeric.columns)
            y_resampled = pd.Series(y_resampled, name=y.name)

            # Add back non-numeric features if they exist
            non_numeric_features = X.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_features) > 0:
                # For non-numeric features, we'll use the original data pattern
                # This is a simplified approach - more sophisticated methods exist
                warnings.warn("Non-numeric features excluded from SMOTE. Consider encoding them first.")

            sampling_info = {
                'method': 'smote',
                'original_distribution': y.value_counts().to_dict(),
                'new_distribution': y_resampled.value_counts().to_dict(),
                'original_size': len(y),
                'new_size': len(y_resampled),
                'features_used': list(X_numeric.columns),
                'sampling_date': datetime.now()
            }

            return X_resampled, y_resampled, sampling_info

        except ImportError:
            warnings.warn("imbalanced-learn not available. Using random oversampling.")
            return self.random_oversample(X, y)
        except Exception as e:
            return X, y, {'error': str(e)}

    def adasyn_oversample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Apply ADASYN (Adaptive Synthetic Sampling)."""
        try:
            from imblearn.over_sampling import ADASYN

            # ADASYN requires numeric features only
            numeric_features = X.select_dtypes(include=[np.number]).columns
            X_numeric = X[numeric_features]

            if X_numeric.empty:
                return self.random_oversample(X, y)

            # Handle missing values
            X_numeric_filled = X_numeric.fillna(X_numeric.median())

            adasyn = ADASYN(random_state=self.config.random_state,
                           sampling_strategy=self.config.sampling_strategy)

            X_resampled, y_resampled = adasyn.fit_resample(X_numeric_filled, y)

            # Convert back to DataFrame
            X_resampled = pd.DataFrame(X_resampled, columns=X_numeric.columns)
            y_resampled = pd.Series(y_resampled, name=y.name)

            sampling_info = {
                'method': 'adasyn',
                'original_distribution': y.value_counts().to_dict(),
                'new_distribution': y_resampled.value_counts().to_dict(),
                'original_size': len(y),
                'new_size': len(y_resampled),
                'features_used': list(X_numeric.columns),
                'sampling_date': datetime.now()
            }

            return X_resampled, y_resampled, sampling_info

        except ImportError:
            warnings.warn("imbalanced-learn not available. Using random oversampling.")
            return self.random_oversample(X, y)
        except Exception as e:
            return X, y, {'error': str(e)}

    def borderline_smote(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Apply Borderline-SMOTE."""
        try:
            from imblearn.over_sampling import BorderlineSMOTE

            # BorderlineSMOTE requires numeric features only
            numeric_features = X.select_dtypes(include=[np.number]).columns
            X_numeric = X[numeric_features]

            if X_numeric.empty:
                return self.random_oversample(X, y)

            # Handle missing values
            X_numeric_filled = X_numeric.fillna(X_numeric.median())

            borderline_smote = BorderlineSMOTE(random_state=self.config.random_state,
                                             sampling_strategy=self.config.sampling_strategy)

            X_resampled, y_resampled = borderline_smote.fit_resample(X_numeric_filled, y)

            # Convert back to DataFrame
            X_resampled = pd.DataFrame(X_resampled, columns=X_numeric.columns)
            y_resampled = pd.Series(y_resampled, name=y.name)

            sampling_info = {
                'method': 'borderline_smote',
                'original_distribution': y.value_counts().to_dict(),
                'new_distribution': y_resampled.value_counts().to_dict(),
                'original_size': len(y),
                'new_size': len(y_resampled),
                'features_used': list(X_numeric.columns),
                'sampling_date': datetime.now()
            }

            return X_resampled, y_resampled, sampling_info

        except ImportError:
            warnings.warn("imbalanced-learn not available. Using random oversampling.")
            return self.random_oversample(X, y)
        except Exception as e:
            return X, y, {'error': str(e)}

    def balance_classes(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Apply the configured sampling method to balance classes.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector

        Returns:
        --------
        Tuple of (X_balanced, y_balanced, sampling_info)
        """
        try:
            # First analyze the class distribution
            class_analysis = self.analyze_class_distribution(y)

            if 'error' in class_analysis:
                return X, y, class_analysis

            # Check if balancing is needed
            if not class_analysis.get('is_imbalanced', False):
                return X, y, {
                    'method': 'none',
                    'reason': 'Classes already balanced',
                    'distribution': class_analysis['class_counts']
                }

            # Apply the specified sampling method
            if self.config.sampling_method == 'random':
                # Decide between over and under sampling based on data size
                if len(y) > 10000:  # Use undersampling for large datasets
                    return self.random_undersample(X, y, self.config.sampling_strategy)
                else:
                    return self.random_oversample(X, y, self.config.sampling_strategy)

            elif self.config.sampling_method == 'smote':
                return self.smote_oversample(X, y)

            elif self.config.sampling_method == 'adasyn':
                return self.adasyn_oversample(X, y)

            elif self.config.sampling_method == 'borderline':
                return self.borderline_smote(X, y)

            else:
                raise ValueError(f"Unknown sampling method: {self.config.sampling_method}")

        except Exception as e:
            return X, y, {'error': str(e)}


class DataSplitter:
    """
    Advanced train-validation-test splitting with various strategies.
    """

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.split_info = {}

    def basic_split(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform basic train-validation-test split."""
        try:
            # First split: train+val vs test
            test_size = self.config.test_size
            stratify_y = y if self.config.stratify and len(y.unique()) > 1 else None

            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=self.config.random_state,
                stratify=stratify_y
            )

            # Second split: train vs validation
            val_size_adjusted = self.config.validation_size / (1 - test_size)
            stratify_y_temp = y_temp if self.config.stratify and len(y_temp.unique()) > 1 else None

            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=self.config.random_state,
                stratify=stratify_y_temp
            )

            split_result = {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'split_info': {
                    'method': 'basic_split',
                    'train_size': len(X_train),
                    'val_size': len(X_val),
                    'test_size': len(X_test),
                    'train_ratio': len(X_train) / len(X),
                    'val_ratio': len(X_val) / len(X),
                    'test_ratio': len(X_test) / len(X),
                    'stratified': self.config.stratify,
                    'random_state': self.config.random_state,
                    'split_date': datetime.now()
                }
            }

            # Add class distribution info if classification
            if self.config.stratify:
                split_result['split_info']['train_distribution'] = y_train.value_counts().to_dict()
                split_result['split_info']['val_distribution'] = y_val.value_counts().to_dict()
                split_result['split_info']['test_distribution'] = y_test.value_counts().to_dict()

            return split_result

        except Exception as e:
            return {'error': str(e)}

    def time_series_split(self, X: pd.DataFrame, y: pd.Series,
                         date_column: str = None) -> Dict[str, Any]:
        """Perform time-series aware splitting."""
        try:
            if date_column and date_column in X.columns:
                # Sort by date column
                X_sorted = X.copy().sort_values(by=date_column)
                y_sorted = y.loc[X_sorted.index]
            else:
                # Assume data is already sorted by time
                X_sorted = X.copy()
                y_sorted = y.copy()

            total_size = len(X_sorted)
            test_size = int(total_size * self.config.test_size)
            val_size = int(total_size * self.config.validation_size)
            train_size = total_size - test_size - val_size

            # Time series split: train (earliest) -> val -> test (latest)
            X_train = X_sorted.iloc[:train_size]
            X_val = X_sorted.iloc[train_size:train_size + val_size]
            X_test = X_sorted.iloc[train_size + val_size:]

            y_train = y_sorted.iloc[:train_size]
            y_val = y_sorted.iloc[train_size:train_size + val_size]
            y_test = y_sorted.iloc[train_size + val_size:]

            split_result = {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'split_info': {
                    'method': 'time_series_split',
                    'date_column': date_column,
                    'train_size': len(X_train),
                    'val_size': len(X_val),
                    'test_size': len(X_test),
                    'train_ratio': len(X_train) / len(X),
                    'val_ratio': len(X_val) / len(X),
                    'test_ratio': len(X_test) / len(X),
                    'split_date': datetime.now()
                }
            }

            # Add date range info
            if date_column and date_column in X.columns:
                split_result['split_info']['train_date_range'] = [
                    str(X_train[date_column].min()),
                    str(X_train[date_column].max())
                ]
                split_result['split_info']['val_date_range'] = [
                    str(X_val[date_column].min()),
                    str(X_val[date_column].max())
                ]
                split_result['split_info']['test_date_range'] = [
                    str(X_test[date_column].min()),
                    str(X_test[date_column].max())
                ]

            return split_result

        except Exception as e:
            return {'error': str(e)}

    def stratified_group_split(self, X: pd.DataFrame, y: pd.Series,
                              group_column: str) -> Dict[str, Any]:
        """Perform stratified split respecting group boundaries."""
        try:
            if group_column not in X.columns:
                return {'error': f'Group column {group_column} not found'}

            # Get unique groups
            unique_groups = X[group_column].unique()
            group_labels = {}

            # For each group, determine its majority class
            for group in unique_groups:
                group_mask = X[group_column] == group
                group_y = y[group_mask]
                majority_class = group_y.value_counts().index[0]
                group_labels[group] = majority_class

            # Split groups stratified by their majority class
            groups_df = pd.DataFrame({
                'group': list(group_labels.keys()),
                'label': list(group_labels.values())
            })

            test_size = self.config.test_size
            val_size_adjusted = self.config.validation_size / (1 - test_size)

            # Split groups
            groups_temp, groups_test = train_test_split(
                groups_df,
                test_size=test_size,
                random_state=self.config.random_state,
                stratify=groups_df['label'] if len(groups_df['label'].unique()) > 1 else None
            )

            groups_train, groups_val = train_test_split(
                groups_temp,
                test_size=val_size_adjusted,
                random_state=self.config.random_state,
                stratify=groups_temp['label'] if len(groups_temp['label'].unique()) > 1 else None
            )

            # Map back to original data
            train_groups = set(groups_train['group'])
            val_groups = set(groups_val['group'])
            test_groups = set(groups_test['group'])

            train_mask = X[group_column].isin(train_groups)
            val_mask = X[group_column].isin(val_groups)
            test_mask = X[group_column].isin(test_groups)

            X_train = X[train_mask]
            X_val = X[val_mask]
            X_test = X[test_mask]

            y_train = y[train_mask]
            y_val = y[val_mask]
            y_test = y[test_mask]

            split_result = {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'split_info': {
                    'method': 'stratified_group_split',
                    'group_column': group_column,
                    'train_groups': len(train_groups),
                    'val_groups': len(val_groups),
                    'test_groups': len(test_groups),
                    'train_size': len(X_train),
                    'val_size': len(X_val),
                    'test_size': len(X_test),
                    'train_ratio': len(X_train) / len(X),
                    'val_ratio': len(X_val) / len(X),
                    'test_ratio': len(X_test) / len(X),
                    'split_date': datetime.now()
                }
            }

            return split_result

        except Exception as e:
            return {'error': str(e)}

    def split_data(self, X: pd.DataFrame, y: pd.Series,
                  split_method: str = 'basic',
                  **kwargs) -> Dict[str, Any]:
        """
        Main method to split data using specified strategy.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector
        split_method : str
            Splitting method ('basic', 'time_series', 'stratified_group')
        **kwargs : additional arguments for specific split methods

        Returns:
        --------
        Dict containing split datasets and metadata
        """
        try:
            if split_method == 'basic':
                return self.basic_split(X, y)

            elif split_method == 'time_series':
                date_column = kwargs.get('date_column')
                return self.time_series_split(X, y, date_column)

            elif split_method == 'stratified_group':
                group_column = kwargs.get('group_column')
                if not group_column:
                    return {'error': 'group_column required for stratified_group split'}
                return self.stratified_group_split(X, y, group_column)

            else:
                return {'error': f'Unknown split method: {split_method}'}

        except Exception as e:
            return {'error': str(e)}


class EigenAnalysis:
    """
    Eigenvalue and eigenvector computation with comprehensive analysis.
    """

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.eigenvalues = None
        self.eigenvectors = None
        self.analysis_results = {}

    def compute_eigendecomposition(self, correlation_matrix: np.ndarray,
                                 method: str = 'numpy') -> Dict[str, Any]:
        """
        Compute eigenvalues and eigenvectors of correlation matrix.

        Parameters:
        -----------
        correlation_matrix : np.ndarray
            Correlation or covariance matrix
        method : str
            Method for computation ('numpy', 'scipy')

        Returns:
        --------
        Dict with eigenanalysis results
        """
        try:
            if method == 'numpy':
                eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
            elif method == 'scipy':
                from scipy.linalg import eigh
                eigenvalues, eigenvectors = eigh(correlation_matrix)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Sort by eigenvalues in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Calculate explained variance ratios
            total_variance = np.sum(eigenvalues)
            explained_variance_ratio = eigenvalues / total_variance
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

            # Kaiser criterion (eigenvalues > 1 for correlation matrix)
            kaiser_components = np.sum(eigenvalues > 1.0)

            # Scree test - find elbow point
            scree_components = self._find_scree_elbow(eigenvalues)

            result = {
                'eigenvalues': eigenvalues.tolist(),
                'eigenvectors': eigenvectors.tolist(),
                'explained_variance_ratio': explained_variance_ratio.tolist(),
                'cumulative_variance_ratio': cumulative_variance_ratio.tolist(),
                'total_variance': float(total_variance),
                'kaiser_components': int(kaiser_components),
                'scree_components': int(scree_components),
                'method': method,
                'matrix_shape': correlation_matrix.shape,
                'computation_date': datetime.now()
            }

            # Store results
            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors
            self.analysis_results = result

            return result

        except Exception as e:
            return {'error': str(e)}

    def _find_scree_elbow(self, eigenvalues: np.ndarray) -> int:
        """Find elbow point in scree plot using rate of change."""
        try:
            if len(eigenvalues) < 3:
                return len(eigenvalues)

            # Calculate second derivatives
            differences = np.diff(eigenvalues)
            second_diff = np.diff(differences)

            # Find the point where second derivative is maximum
            elbow_idx = np.argmax(second_diff) + 2  # +2 due to double differencing

            return min(elbow_idx, len(eigenvalues))

        except Exception:
            return len(eigenvalues)

    def parallel_analysis(self, data: pd.DataFrame, n_iterations: int = 100) -> Dict[str, Any]:
        """
        Perform parallel analysis to determine number of components.

        Parameters:
        -----------
        data : pd.DataFrame
            Original data for comparison
        n_iterations : int
            Number of random datasets to generate

        Returns:
        --------
        Dict with parallel analysis results
        """
        try:
            n_vars, n_obs = data.shape

            # Generate random eigenvalues
            random_eigenvalues = []

            for _ in range(n_iterations):
                # Generate random data with same dimensions
                random_data = np.random.normal(0, 1, (n_obs, n_vars))
                random_corr = np.corrcoef(random_data.T)

                # Compute eigenvalues
                random_eigs = np.linalg.eigvals(random_corr)
                random_eigs = np.sort(random_eigs)[::-1]
                random_eigenvalues.append(random_eigs)

            # Calculate percentiles
            random_eigenvalues = np.array(random_eigenvalues)
            percentile_95 = np.percentile(random_eigenvalues, 95, axis=0)
            percentile_99 = np.percentile(random_eigenvalues, 99, axis=0)

            # Compare with actual eigenvalues
            if self.eigenvalues is not None:
                actual_eigenvalues = self.eigenvalues
            else:
                # Compute if not already done
                corr_matrix = data.corr().values
                actual_eigenvalues = np.linalg.eigvals(corr_matrix)
                actual_eigenvalues = np.sort(actual_eigenvalues)[::-1]

            # Determine number of components
            n_components_95 = np.sum(actual_eigenvalues > percentile_95)
            n_components_99 = np.sum(actual_eigenvalues > percentile_99)

            result = {
                'actual_eigenvalues': actual_eigenvalues.tolist(),
                'random_eigenvalues_95th': percentile_95.tolist(),
                'random_eigenvalues_99th': percentile_99.tolist(),
                'n_components_95th': int(n_components_95),
                'n_components_99th': int(n_components_99),
                'n_iterations': n_iterations,
                'parallel_analysis_date': datetime.now()
            }

            return result

        except Exception as e:
            return {'error': str(e)}

    def visualize_eigenanalysis(self, save_path: str = None) -> Dict[str, Any]:
        """Create visualizations for eigenanalysis."""
        try:
            if self.eigenvalues is None:
                return {'error': 'No eigenvalues computed. Run compute_eigendecomposition first.'}

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

            # Scree plot
            ax1.plot(range(1, len(self.eigenvalues) + 1), self.eigenvalues, 'bo-')
            ax1.axhline(y=1, color='r', linestyle='--', label='Kaiser criterion')
            ax1.set_xlabel('Component Number')
            ax1.set_ylabel('Eigenvalue')
            ax1.set_title('Scree Plot')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Cumulative variance explained
            if 'cumulative_variance_ratio' in self.analysis_results:
                cumvar = self.analysis_results['cumulative_variance_ratio']
                ax2.plot(range(1, len(cumvar) + 1), cumvar, 'go-')
                ax2.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
                ax2.set_xlabel('Component Number')
                ax2.set_ylabel('Cumulative Variance Explained')
                ax2.set_title('Cumulative Variance Explained')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            # Individual variance explained
            if 'explained_variance_ratio' in self.analysis_results:
                var_ratio = self.analysis_results['explained_variance_ratio']
                ax3.bar(range(1, len(var_ratio) + 1), var_ratio)
                ax3.set_xlabel('Component Number')
                ax3.set_ylabel('Variance Explained')
                ax3.set_title('Individual Component Variance')
                ax3.grid(True, alpha=0.3)

            # Eigenvalue distribution
            ax4.hist(self.eigenvalues, bins=min(20, len(self.eigenvalues)//2), alpha=0.7)
            ax4.axvline(x=1, color='r', linestyle='--', label='Kaiser criterion')
            ax4.set_xlabel('Eigenvalue')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Eigenvalue Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            result = {
                'visualization_type': 'eigenanalysis',
                'components_plotted': len(self.eigenvalues),
                'save_path': save_path,
                'visualization_date': datetime.now()
            }

            return result

        except Exception as e:
            return {'error': str(e)}


class PrincipalComponentAnalysis:
    """
    Comprehensive Principal Component Analysis implementation.
    """

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.pca_model = None
        self.loadings = None
        self.scores = None
        self.eigen_analysis = EigenAnalysis(config)

    def fit_pca(self, data: pd.DataFrame,
                n_components: Optional[int] = None,
                standardize: bool = True) -> Dict[str, Any]:
        """
        Fit PCA model to data.

        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        n_components : int, optional
            Number of components to retain
        standardize : bool
            Whether to standardize data before PCA

        Returns:
        --------
        Dict with PCA results
        """
        try:
            # Select only numeric columns
            numeric_data = data.select_dtypes(include=[np.number])

            if numeric_data.empty:
                return {'error': 'No numeric columns found for PCA'}

            # Handle missing values
            numeric_data_clean = numeric_data.fillna(numeric_data.mean())

            # Standardize if requested
            if standardize:
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(numeric_data_clean)
                data_scaled = pd.DataFrame(data_scaled, columns=numeric_data_clean.columns,
                                         index=numeric_data_clean.index)
            else:
                data_scaled = numeric_data_clean

            # Determine number of components
            if n_components is None:
                if self.config.pca_n_components is not None:
                    n_components = self.config.pca_n_components
                else:
                    # Use variance threshold
                    pca_full = PCA()
                    pca_full.fit(data_scaled)
                    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
                    n_components = np.argmax(cumvar >= self.config.pca_variance_threshold) + 1

            # Fit PCA
            pca = PCA(n_components=n_components, random_state=self.config.random_state)
            scores = pca.fit_transform(data_scaled)

            # Calculate loadings (correlations between original variables and components)
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

            # Component scores as DataFrame
            component_names = [f'PC{i+1}' for i in range(n_components)]
            scores_df = pd.DataFrame(scores, columns=component_names, index=data_scaled.index)

            # Loadings as DataFrame
            loadings_df = pd.DataFrame(loadings, columns=component_names, index=data_scaled.columns)

            # Eigenanalysis
            correlation_matrix = data_scaled.corr().values
            eigen_results = self.eigen_analysis.compute_eigendecomposition(correlation_matrix)

            # Calculate communalities (proportion of variance explained for each variable)
            communalities = np.sum(loadings**2, axis=1)

            # Calculate component correlations
            component_correlations = np.corrcoef(scores.T)

            result = {
                'pca_model': pca,
                'scores': scores_df,
                'loadings': loadings_df,
                'explained_variance': pca.explained_variance_.tolist(),
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'communalities': dict(zip(data_scaled.columns, communalities)),
                'component_correlations': component_correlations.tolist(),
                'n_components': n_components,
                'n_features': len(data_scaled.columns),
                'n_observations': len(data_scaled),
                'standardized': standardize,
                'eigenanalysis': eigen_results,
                'feature_names': list(data_scaled.columns),
                'component_names': component_names,
                'pca_date': datetime.now()
            }

            # Store results
            self.pca_model = pca
            self.loadings = loadings_df
            self.scores = scores_df

            return result

        except Exception as e:
            return {'error': str(e)}

    def transform_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Transform new data using fitted PCA model."""
        try:
            if self.pca_model is None:
                return None, {'error': 'PCA model not fitted. Call fit_pca first.'}

            # Select numeric columns and handle missing values
            numeric_data = data.select_dtypes(include=[np.number])
            numeric_data_clean = numeric_data.fillna(numeric_data.mean())

            # Transform data
            scores = self.pca_model.transform(numeric_data_clean)

            # Convert to DataFrame
            component_names = [f'PC{i+1}' for i in range(scores.shape[1])]
            scores_df = pd.DataFrame(scores, columns=component_names, index=data.index)

            transform_info = {
                'transformed_shape': scores_df.shape,
                'original_shape': numeric_data_clean.shape,
                'n_components': scores.shape[1],
                'transform_date': datetime.now()
            }

            return scores_df, transform_info

        except Exception as e:
            return None, {'error': str(e)}

    def interpret_components(self, loadings_threshold: float = 0.3) -> Dict[str, Any]:
        """Interpret PCA components based on loadings."""
        try:
            if self.loadings is None:
                return {'error': 'No loadings available. Run fit_pca first.'}

            interpretation = {}

            for component in self.loadings.columns:
                loadings_col = self.loadings[component]

                # Find significant loadings
                significant_vars = loadings_col[abs(loadings_col) >= loadings_threshold]
                significant_vars = significant_vars.sort_values(key=abs, ascending=False)

                # Categorize as positive and negative contributors
                positive_contributors = significant_vars[significant_vars > 0]
                negative_contributors = significant_vars[significant_vars < 0]

                interpretation[component] = {
                    'explained_variance': float(self.pca_model.explained_variance_ratio_[
                        int(component.replace('PC', '')) - 1
                    ]),
                    'significant_variables': len(significant_vars),
                    'positive_contributors': positive_contributors.to_dict(),
                    'negative_contributors': negative_contributors.to_dict(),
                    'top_contributor': significant_vars.index[0] if len(significant_vars) > 0 else None,
                    'top_loading': float(significant_vars.iloc[0]) if len(significant_vars) > 0 else 0
                }

            result = {
                'component_interpretation': interpretation,
                'loadings_threshold': loadings_threshold,
                'interpretation_date': datetime.now()
            }

            return result

        except Exception as e:
            return {'error': str(e)}

    def biplot(self, component_x: int = 1, component_y: int = 2,
               save_path: str = None) -> Dict[str, Any]:
        """Create PCA biplot."""
        try:
            if self.scores is None or self.loadings is None:
                return {'error': 'PCA not fitted. Run fit_pca first.'}

            pc_x = f'PC{component_x}'
            pc_y = f'PC{component_y}'

            if pc_x not in self.scores.columns or pc_y not in self.scores.columns:
                return {'error': f'Components {pc_x} or {pc_y} not available'}

            fig, ax = plt.subplots(figsize=(12, 10))

            # Plot scores
            ax.scatter(self.scores[pc_x], self.scores[pc_y], alpha=0.6, s=50)

            # Plot loadings as arrows
            for i, var in enumerate(self.loadings.index):
                ax.arrow(0, 0,
                        self.loadings.loc[var, pc_x] * 3,
                        self.loadings.loc[var, pc_y] * 3,
                        head_width=0.1, head_length=0.1,
                        fc='red', ec='red', alpha=0.7)

                # Add variable labels
                ax.text(self.loadings.loc[var, pc_x] * 3.2,
                       self.loadings.loc[var, pc_y] * 3.2,
                       var, fontsize=9, ha='center', va='center')

            # Get explained variance for axes labels
            var_x = self.pca_model.explained_variance_ratio_[component_x - 1] * 100
            var_y = self.pca_model.explained_variance_ratio_[component_y - 1] * 100

            ax.set_xlabel(f'{pc_x} ({var_x:.1f}% variance)')
            ax.set_ylabel(f'{pc_y} ({var_y:.1f}% variance)')
            ax.set_title('PCA Biplot')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            result = {
                'biplot_components': [component_x, component_y],
                'explained_variance': [var_x, var_y],
                'save_path': save_path,
                'biplot_date': datetime.now()
            }

            return result

        except Exception as e:
            return {'error': str(e)}


class FactorRotation:
    """
    Factor rotation methods for Factor Analysis.
    """

    @staticmethod
    def varimax_rotation(loadings: np.ndarray, max_iter: int = 1000,
                        tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Varimax rotation on factor loadings.

        Parameters:
        -----------
        loadings : np.ndarray
            Original factor loadings matrix
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance

        Returns:
        --------
        Tuple of (rotated_loadings, rotation_matrix)
        """
        try:
            n_vars, n_factors = loadings.shape
            rotation_matrix = np.eye(n_factors)

            for iteration in range(max_iter):
                # Current loadings
                current_loadings = loadings @ rotation_matrix

                # Compute gradient for Varimax criterion
                # Varimax maximizes the variance of squared loadings
                u, s, vh = np.linalg.svd(
                    loadings.T @ (current_loadings**3 - current_loadings @ np.diag(np.sum(current_loadings**2, axis=0)) / n_vars)
                )

                rotation_update = u @ vh
                rotation_matrix = rotation_matrix @ rotation_update

                # Check convergence
                if np.abs(1 - np.abs(np.sum(np.diag(rotation_update)))) < tol:
                    break

            rotated_loadings = loadings @ rotation_matrix

            return rotated_loadings, rotation_matrix

        except Exception as e:
            warnings.warn(f"Varimax rotation failed: {str(e)}")
            return loadings, np.eye(loadings.shape[1])

    @staticmethod
    def promax_rotation(loadings: np.ndarray, power: int = 4,
                       max_iter: int = 1000, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Promax rotation on factor loadings.

        Parameters:
        -----------
        loadings : np.ndarray
            Original factor loadings matrix
        power : int
            Power for Promax rotation (typically 2-4)
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance

        Returns:
        --------
        Tuple of (rotated_loadings, rotation_matrix)
        """
        try:
            # First perform Varimax rotation
            varimax_loadings, varimax_rotation = FactorRotation.varimax_rotation(
                loadings, max_iter, tol
            )

            # Create target matrix for Promax
            # Raise absolute values to power and preserve signs
            target_matrix = np.sign(varimax_loadings) * (np.abs(varimax_loadings) ** power)

            # Normalize target matrix
            target_matrix = target_matrix / np.sqrt(np.sum(target_matrix**2, axis=0))

            # Procrustes rotation to match target
            u, s, vh = np.linalg.svd(varimax_loadings.T @ target_matrix)
            promax_rotation = u @ vh

            # Combined rotation matrix
            total_rotation = varimax_rotation @ promax_rotation
            rotated_loadings = loadings @ total_rotation

            return rotated_loadings, total_rotation

        except Exception as e:
            warnings.warn(f"Promax rotation failed: {str(e)}")
            return loadings, np.eye(loadings.shape[1])


class FactorAnalysis:
    """
    Comprehensive Factor Analysis implementation.
    """

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.factor_model = None
        self.loadings = None
        self.scores = None
        self.rotation = FactorRotation()

    def fit_factor_analysis(self, data: pd.DataFrame,
                           n_factors: Optional[int] = None,
                           method: str = 'principal',
                           rotation: str = 'varimax',
                           standardize: bool = True) -> Dict[str, Any]:
        """
        Fit Factor Analysis model.

        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        n_factors : int, optional
            Number of factors
        method : str
            Extraction method ('principal', 'ml', 'minres')
        rotation : str
            Rotation method ('varimax', 'promax', 'none')
        standardize : bool
            Whether to standardize data

        Returns:
        --------
        Dict with Factor Analysis results
        """
        try:
            # Select numeric columns
            numeric_data = data.select_dtypes(include=[np.number])

            if numeric_data.empty:
                return {'error': 'No numeric columns found for Factor Analysis'}

            # Handle missing values
            numeric_data_clean = numeric_data.fillna(numeric_data.mean())

            # Standardize if requested
            if standardize:
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(numeric_data_clean)
                data_scaled = pd.DataFrame(data_scaled, columns=numeric_data_clean.columns,
                                         index=numeric_data_clean.index)
            else:
                data_scaled = numeric_data_clean

            # Determine number of factors
            if n_factors is None:
                n_factors = self.config.fa_n_factors or self._determine_n_factors(data_scaled)

            # Perform factor extraction based on method
            if method == 'principal':
                loadings, communalities, eigenvalues = self._principal_factor_method(
                    data_scaled, n_factors
                )
            elif method == 'ml':
                loadings, communalities, eigenvalues = self._maximum_likelihood_method(
                    data_scaled, n_factors
                )
            elif method == 'minres':
                loadings, communalities, eigenvalues = self._minimum_residual_method(
                    data_scaled, n_factors
                )
            else:
                raise ValueError(f"Unknown extraction method: {method}")

            # Apply rotation
            if rotation == 'varimax':
                rotated_loadings, rotation_matrix = self.rotation.varimax_rotation(loadings)
            elif rotation == 'promax':
                rotated_loadings, rotation_matrix = self.rotation.promax_rotation(loadings)
            elif rotation == 'none':
                rotated_loadings = loadings
                rotation_matrix = np.eye(loadings.shape[1])
            else:
                raise ValueError(f"Unknown rotation method: {rotation}")

            # Calculate factor scores
            factor_scores = self._calculate_factor_scores(data_scaled, rotated_loadings)

            # Create DataFrames
            factor_names = [f'Factor{i+1}' for i in range(n_factors)]
            loadings_df = pd.DataFrame(rotated_loadings, columns=factor_names,
                                     index=data_scaled.columns)
            scores_df = pd.DataFrame(factor_scores, columns=factor_names,
                                   index=data_scaled.index)

            # Calculate additional statistics
            variance_explained = np.sum(rotated_loadings**2, axis=0)
            proportion_variance = variance_explained / len(data_scaled.columns)
            cumulative_variance = np.cumsum(proportion_variance)

            # Factor correlations (if oblique rotation)
            factor_correlations = np.corrcoef(factor_scores.T)

            result = {
                'loadings': loadings_df,
                'scores': scores_df,
                'communalities': dict(zip(data_scaled.columns, communalities)),
                'eigenvalues': eigenvalues.tolist() if eigenvalues is not None else None,
                'variance_explained': variance_explained.tolist(),
                'proportion_variance': proportion_variance.tolist(),
                'cumulative_variance': cumulative_variance.tolist(),
                'factor_correlations': factor_correlations.tolist(),
                'rotation_matrix': rotation_matrix.tolist(),
                'n_factors': n_factors,
                'n_variables': len(data_scaled.columns),
                'n_observations': len(data_scaled),
                'extraction_method': method,
                'rotation_method': rotation,
                'standardized': standardize,
                'factor_names': factor_names,
                'variable_names': list(data_scaled.columns),
                'fa_date': datetime.now()
            }

            # Store results
            self.loadings = loadings_df
            self.scores = scores_df

            return result

        except Exception as e:
            return {'error': str(e)}

    def _determine_n_factors(self, data: pd.DataFrame) -> int:
        """Determine optimal number of factors using various criteria."""
        try:
            # Use Kaiser criterion and parallel analysis
            corr_matrix = data.corr().values
            eigenvalues = np.linalg.eigvals(corr_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]

            # Kaiser criterion
            kaiser_factors = np.sum(eigenvalues > 1.0)

            # Use Kaiser criterion but limit to reasonable number
            return min(kaiser_factors, len(data.columns) // 2)

        except Exception:
            return min(3, len(data.columns) // 2)

    def _principal_factor_method(self, data: pd.DataFrame,
                                n_factors: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Principal Factor Analysis method."""
        try:
            # Start with correlation matrix
            R = data.corr().values

            # Initial communality estimates (squared multiple correlations)
            communalities = np.zeros(len(data.columns))
            for i in range(len(data.columns)):
                others = [j for j in range(len(data.columns)) if j != i]
                if len(others) > 0:
                    reg_matrix = R[np.ix_(others, others)]
                    target_corr = R[i, others]
                    try:
                        coeffs = np.linalg.solve(reg_matrix, target_corr)
                        communalities[i] = np.sum(coeffs * target_corr)
                    except np.linalg.LinAlgError:
                        communalities[i] = np.max(np.abs(R[i, others])) ** 2

            # Iterate to convergence
            max_iter = 100
            tol = 1e-6

            for iteration in range(max_iter):
                # Reduced correlation matrix
                R_reduced = R.copy()
                np.fill_diagonal(R_reduced, communalities)

                # Eigendecomposition
                eigenvalues, eigenvectors = np.linalg.eigh(R_reduced)
                idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]

                # Extract factors
                loadings = eigenvectors[:, :n_factors] @ np.diag(np.sqrt(np.maximum(eigenvalues[:n_factors], 0)))

                # Update communalities
                new_communalities = np.sum(loadings**2, axis=1)

                # Check convergence
                if np.max(np.abs(new_communalities - communalities)) < tol:
                    break

                communalities = new_communalities

            return loadings, communalities, eigenvalues

        except Exception as e:
            warnings.warn(f"Principal factor method failed: {str(e)}")
            # Fallback to simple PCA
            pca = PCA(n_components=n_factors)
            pca.fit(data)
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            communalities = np.sum(loadings**2, axis=1)
            return loadings, communalities, pca.explained_variance_

    def _maximum_likelihood_method(self, data: pd.DataFrame,
                                  n_factors: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Maximum Likelihood Factor Analysis."""
        try:
            # Use sklearn's FactorAnalysis
            fa = FactorAnalysis(n_components=n_factors, random_state=self.config.random_state)
            fa.fit(data)

            loadings = fa.components_.T
            communalities = np.sum(loadings**2, axis=1)
            eigenvalues = None  # Not directly available from sklearn

            return loadings, communalities, eigenvalues

        except Exception as e:
            warnings.warn(f"ML factor method failed: {str(e)}")
            return self._principal_factor_method(data, n_factors)

    def _minimum_residual_method(self, data: pd.DataFrame,
                                n_factors: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Minimum Residual (MinRes) Factor Analysis."""
        try:
            # This is a simplified implementation
            # Full MinRes would require optimization of residual matrix
            R = data.corr().values

            # Use PCA as starting point
            pca = PCA(n_components=n_factors)
            pca.fit(data)
            initial_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

            # Minimize residuals (simplified)
            communalities = np.sum(initial_loadings**2, axis=1)

            return initial_loadings, communalities, pca.explained_variance_

        except Exception as e:
            warnings.warn(f"MinRes factor method failed: {str(e)}")
            return self._principal_factor_method(data, n_factors)

    def _calculate_factor_scores(self, data: pd.DataFrame,
                                loadings: np.ndarray) -> np.ndarray:
        """Calculate factor scores using regression method."""
        try:
            R = data.corr().values
            R_inv = np.linalg.pinv(R)  # Use pseudo-inverse for numerical stability

            # Factor score coefficients
            score_coeffs = R_inv @ loadings

            # Calculate scores
            data_standardized = (data - data.mean()) / data.std()
            factor_scores = data_standardized.values @ score_coeffs

            return factor_scores

        except Exception as e:
            warnings.warn(f"Factor score calculation failed: {str(e)}")
            return np.zeros((len(data), loadings.shape[1]))

    def interpret_factors(self, loadings_threshold: float = 0.3) -> Dict[str, Any]:
        """Interpret factors based on loadings."""
        try:
            if self.loadings is None:
                return {'error': 'No loadings available. Run fit_factor_analysis first.'}

            interpretation = {}

            for factor in self.loadings.columns:
                loadings_col = self.loadings[factor]

                # Find significant loadings
                significant_vars = loadings_col[abs(loadings_col) >= loadings_threshold]
                significant_vars = significant_vars.sort_values(key=abs, ascending=False)

                # Categorize variables
                high_positive = significant_vars[significant_vars >= loadings_threshold]
                high_negative = significant_vars[significant_vars <= -loadings_threshold]

                interpretation[factor] = {
                    'significant_variables': len(significant_vars),
                    'high_positive_loadings': high_positive.to_dict(),
                    'high_negative_loadings': high_negative.to_dict(),
                    'primary_variables': list(significant_vars.head(3).index),
                    'interpretation_strength': float(np.mean(np.abs(significant_vars))) if len(significant_vars) > 0 else 0
                }

            result = {
                'factor_interpretation': interpretation,
                'loadings_threshold': loadings_threshold,
                'interpretation_date': datetime.now()
            }

            return result

        except Exception as e:
            return {'error': str(e)}


class VariableClustering:
    """
    Variable Clustering (VARCLUS) implementation.
    """

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.clusters = {}
        self.cluster_stats = {}

    def varclus_analysis(self, data: pd.DataFrame,
                        max_clusters: Optional[int] = None,
                        min_correlation: Optional[float] = None) -> Dict[str, Any]:
        """
        Perform Variable Clustering analysis.

        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        max_clusters : int, optional
            Maximum number of clusters
        min_correlation : float, optional
            Minimum within-cluster correlation

        Returns:
        --------
        Dict with clustering results
        """
        try:
            # Select numeric columns
            numeric_data = data.select_dtypes(include=[np.number])

            if numeric_data.empty:
                return {'error': 'No numeric columns found for Variable Clustering'}

            # Handle missing values
            numeric_data_clean = numeric_data.fillna(numeric_data.mean())

            # Set parameters
            if max_clusters is None:
                max_clusters = self.config.varclus_max_clusters
            if min_correlation is None:
                min_correlation = self.config.varclus_min_correlation

            # Calculate correlation matrix
            corr_matrix = numeric_data_clean.corr()

            # Hierarchical clustering of variables
            # Use 1-correlation as distance
            distance_matrix = 1 - np.abs(corr_matrix)
            np.fill_diagonal(distance_matrix, 0)

            # Perform hierarchical clustering
            condensed_dist = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(condensed_dist, method='ward')

            # Determine optimal number of clusters
            optimal_clusters = self._determine_optimal_clusters(
                corr_matrix, linkage_matrix, max_clusters, min_correlation
            )

            # Get cluster assignments
            cluster_labels = fcluster(linkage_matrix, optimal_clusters, criterion='maxclust')

            # Organize variables by cluster
            clusters = {}
            for var, cluster_id in zip(corr_matrix.columns, cluster_labels):
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(var)

            # Calculate cluster statistics
            cluster_stats = {}
            for cluster_id, variables in clusters.items():
                if len(variables) > 1:
                    cluster_corr = corr_matrix.loc[variables, variables]

                    # Within-cluster correlations
                    upper_triangle = cluster_corr.where(
                        np.triu(np.ones(cluster_corr.shape), k=1).astype(bool)
                    )
                    within_corr = upper_triangle.stack().values

                    cluster_stats[cluster_id] = {
                        'n_variables': len(variables),
                        'variables': variables,
                        'mean_within_correlation': float(np.mean(within_corr)),
                        'min_within_correlation': float(np.min(within_corr)),
                        'max_within_correlation': float(np.max(within_corr)),
                        'std_within_correlation': float(np.std(within_corr))
                    }
                else:
                    cluster_stats[cluster_id] = {
                        'n_variables': 1,
                        'variables': variables,
                        'mean_within_correlation': 1.0,
                        'min_within_correlation': 1.0,
                        'max_within_correlation': 1.0,
                        'std_within_correlation': 0.0
                    }

            # Calculate between-cluster correlations
            between_cluster_corr = self._calculate_between_cluster_correlations(
                corr_matrix, clusters
            )

            # Principal component analysis for each cluster
            cluster_pca = {}
            for cluster_id, variables in clusters.items():
                if len(variables) > 1:
                    cluster_data = numeric_data_clean[variables]
                    pca = PCA(n_components=1)
                    pca.fit(cluster_data)

                    cluster_pca[cluster_id] = {
                        'explained_variance_ratio': float(pca.explained_variance_ratio_[0]),
                        'component_loadings': dict(zip(variables, pca.components_[0]))
                    }

            result = {
                'optimal_clusters': optimal_clusters,
                'cluster_assignments': dict(zip(corr_matrix.columns, cluster_labels)),
                'clusters': clusters,
                'cluster_statistics': cluster_stats,
                'between_cluster_correlations': between_cluster_corr,
                'cluster_pca': cluster_pca,
                'linkage_matrix': linkage_matrix.tolist(),
                'n_variables': len(corr_matrix.columns),
                'parameters': {
                    'max_clusters': max_clusters,
                    'min_correlation': min_correlation
                },
                'varclus_date': datetime.now()
            }

            # Store results
            self.clusters = clusters
            self.cluster_stats = cluster_stats

            return result

        except Exception as e:
            return {'error': str(e)}

    def _determine_optimal_clusters(self, corr_matrix: pd.DataFrame,
                                   linkage_matrix: np.ndarray,
                                   max_clusters: int,
                                   min_correlation: float) -> int:
        """Determine optimal number of clusters."""
        try:
            best_score = -1
            optimal_clusters = 2

            for n_clusters in range(2, min(max_clusters + 1, len(corr_matrix.columns))):
                cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

                # Calculate average within-cluster correlation
                total_within_corr = 0
                total_pairs = 0

                for cluster_id in range(1, n_clusters + 1):
                    cluster_vars = [var for var, label in zip(corr_matrix.columns, cluster_labels)
                                  if label == cluster_id]

                    if len(cluster_vars) > 1:
                        cluster_corr = corr_matrix.loc[cluster_vars, cluster_vars]
                        upper_triangle = cluster_corr.where(
                            np.triu(np.ones(cluster_corr.shape), k=1).astype(bool)
                        )
                        within_corr = upper_triangle.stack().values
                        total_within_corr += np.sum(within_corr)
                        total_pairs += len(within_corr)

                if total_pairs > 0:
                    avg_within_corr = total_within_corr / total_pairs

                    # Score based on within-cluster correlation and number of clusters
                    # Prefer higher correlation but penalize too many clusters
                    score = avg_within_corr - 0.1 * (n_clusters / len(corr_matrix.columns))

                    if avg_within_corr >= min_correlation and score > best_score:
                        best_score = score
                        optimal_clusters = n_clusters

            return optimal_clusters

        except Exception:
            return min(3, len(corr_matrix.columns) // 2)

    def _calculate_between_cluster_correlations(self, corr_matrix: pd.DataFrame,
                                              clusters: Dict[int, List[str]]) -> Dict[str, float]:
        """Calculate correlations between clusters."""
        try:
            between_corr = {}

            cluster_ids = list(clusters.keys())
            for i, cluster1 in enumerate(cluster_ids):
                for cluster2 in cluster_ids[i+1:]:
                    vars1 = clusters[cluster1]
                    vars2 = clusters[cluster2]

                    # Calculate all pairwise correlations between clusters
                    cross_corr = []
                    for var1 in vars1:
                        for var2 in vars2:
                            cross_corr.append(abs(corr_matrix.loc[var1, var2]))

                    mean_between_corr = np.mean(cross_corr) if cross_corr else 0

                    between_corr[f'cluster_{cluster1}_vs_{cluster2}'] = float(mean_between_corr)

            return between_corr

        except Exception:
            return {}

    def select_cluster_representatives(self, data: pd.DataFrame,
                                     method: str = 'pca') -> Dict[str, Any]:
        """Select representative variables from each cluster."""
        try:
            if not self.clusters:
                return {'error': 'No clusters available. Run varclus_analysis first.'}

            representatives = {}
            selection_info = {}

            numeric_data = data.select_dtypes(include=[np.number])
            numeric_data_clean = numeric_data.fillna(numeric_data.mean())

            for cluster_id, variables in self.clusters.items():
                if len(variables) == 1:
                    representatives[cluster_id] = variables[0]
                    selection_info[cluster_id] = {
                        'method': 'single_variable',
                        'reason': 'Only one variable in cluster'
                    }
                else:
                    cluster_data = numeric_data_clean[variables]

                    if method == 'pca':
                        # Select variable with highest loading on first PC
                        pca = PCA(n_components=1)
                        pca.fit(cluster_data)
                        loadings = abs(pca.components_[0])
                        best_var_idx = np.argmax(loadings)
                        representatives[cluster_id] = variables[best_var_idx]

                        selection_info[cluster_id] = {
                            'method': 'pca',
                            'pca_loading': float(loadings[best_var_idx]),
                            'explained_variance': float(pca.explained_variance_ratio_[0])
                        }

                    elif method == 'highest_correlation':
                        # Select variable with highest average correlation with others
                        corr_matrix = cluster_data.corr()
                        avg_corr = corr_matrix.mean()
                        best_var = avg_corr.idxmax()
                        representatives[cluster_id] = best_var

                        selection_info[cluster_id] = {
                            'method': 'highest_correlation',
                            'average_correlation': float(avg_corr[best_var])
                        }

                    elif method == 'centroid':
                        # Select variable closest to cluster centroid
                        cluster_standardized = (cluster_data - cluster_data.mean()) / cluster_data.std()
                        centroid = cluster_standardized.mean(axis=1)

                        distances = []
                        for var in variables:
                            distance = np.sqrt(np.sum((cluster_standardized[var] - centroid)**2))
                            distances.append(distance)

                        best_var_idx = np.argmin(distances)
                        representatives[cluster_id] = variables[best_var_idx]

                        selection_info[cluster_id] = {
                            'method': 'centroid',
                            'distance_to_centroid': float(distances[best_var_idx])
                        }

            result = {
                'cluster_representatives': representatives,
                'selection_method': method,
                'selection_info': selection_info,
                'representative_variables': list(representatives.values()),
                'n_original_variables': len(numeric_data.columns),
                'n_representatives': len(representatives),
                'reduction_ratio': len(representatives) / len(numeric_data.columns),
                'selection_date': datetime.now()
            }

            return result

        except Exception as e:
            return {'error': str(e)}

    def visualize_clusters(self, save_path: str = None) -> Dict[str, Any]:
        """Visualize variable clustering results."""
        try:
            if not self.clusters:
                return {'error': 'No clusters available. Run varclus_analysis first.'}

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Cluster size distribution
            cluster_sizes = [len(variables) for variables in self.clusters.values()]
            ax1.bar(range(1, len(cluster_sizes) + 1), cluster_sizes)
            ax1.set_xlabel('Cluster ID')
            ax1.set_ylabel('Number of Variables')
            ax1.set_title('Variables per Cluster')
            ax1.grid(True, alpha=0.3)

            # Within-cluster correlation distribution
            within_correlations = []
            for cluster_id, stats in self.cluster_stats.items():
                if stats['n_variables'] > 1:
                    within_correlations.append(stats['mean_within_correlation'])

            if within_correlations:
                ax2.hist(within_correlations, bins=min(10, len(within_correlations)), alpha=0.7)
                ax2.axvline(x=self.config.varclus_min_correlation, color='r', linestyle='--',
                           label=f'Min threshold ({self.config.varclus_min_correlation})')
                ax2.set_xlabel('Mean Within-Cluster Correlation')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Distribution of Within-Cluster Correlations')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            result = {
                'visualization_type': 'variable_clustering',
                'n_clusters': len(self.clusters),
                'save_path': save_path,
                'visualization_date': datetime.now()
            }

            return result

        except Exception as e:
            return {'error': str(e)}


class DimensionalityReduction:
    """
    Comprehensive dimensionality reduction pipeline.
    """

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.pca = PrincipalComponentAnalysis(config)
        self.fa = FactorAnalysis(config)
        self.varclus = VariableClustering(config)
        self.reduction_results = {}

    def compare_methods(self, data: pd.DataFrame,
                       methods: List[str] = ['pca', 'fa', 'varclus']) -> Dict[str, Any]:
        """
        Compare different dimensionality reduction methods.

        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        methods : List[str]
            Methods to compare

        Returns:
        --------
        Dict with comparison results
        """
        try:
            comparison_results = {}
            numeric_data = data.select_dtypes(include=[np.number])

            if numeric_data.empty:
                return {'error': 'No numeric columns found'}

            original_dimensions = len(numeric_data.columns)

            # PCA
            if 'pca' in methods:
                pca_result = self.pca.fit_pca(data)
                if 'error' not in pca_result:
                    comparison_results['pca'] = {
                        'method': 'Principal Component Analysis',
                        'components_retained': pca_result['n_components'],
                        'variance_explained': sum(pca_result['explained_variance_ratio']),
                        'dimension_reduction': (original_dimensions - pca_result['n_components']) / original_dimensions,
                        'interpretability': 'Low',  # Components are linear combinations
                        'results': pca_result
                    }

            # Factor Analysis
            if 'fa' in methods:
                fa_result = self.fa.fit_factor_analysis(data)
                if 'error' not in fa_result:
                    comparison_results['fa'] = {
                        'method': 'Factor Analysis',
                        'factors_retained': fa_result['n_factors'],
                        'variance_explained': sum(fa_result['proportion_variance']),
                        'dimension_reduction': (original_dimensions - fa_result['n_factors']) / original_dimensions,
                        'interpretability': 'Medium',  # Factors represent latent constructs
                        'results': fa_result
                    }

            # Variable Clustering
            if 'varclus' in methods:
                varclus_result = self.varclus.varclus_analysis(data)
                if 'error' not in varclus_result:
                    representatives = self.varclus.select_cluster_representatives(data)
                    if 'error' not in representatives:
                        comparison_results['varclus'] = {
                            'method': 'Variable Clustering',
                            'variables_retained': representatives['n_representatives'],
                            'clusters_formed': varclus_result['optimal_clusters'],
                            'dimension_reduction': representatives['reduction_ratio'],
                            'interpretability': 'High',  # Original variables retained
                            'results': {
                                'clustering': varclus_result,
                                'representatives': representatives
                            }
                        }

            # Overall comparison
            if comparison_results:
                summary = {
                    'original_dimensions': original_dimensions,
                    'methods_compared': list(comparison_results.keys()),
                    'best_variance_explained': None,
                    'best_interpretability': None,
                    'best_dimension_reduction': None
                }

                # Find best methods by different criteria
                variance_scores = {method: results.get('variance_explained', 0)
                                 for method, results in comparison_results.items()}
                if variance_scores:
                    summary['best_variance_explained'] = max(variance_scores, key=variance_scores.get)

                reduction_scores = {method: results.get('dimension_reduction', 0)
                                  for method, results in comparison_results.items()}
                if reduction_scores:
                    summary['best_dimension_reduction'] = max(reduction_scores, key=reduction_scores.get)

                # Interpretability ranking
                interp_ranking = {'High': 3, 'Medium': 2, 'Low': 1}
                interp_scores = {method: interp_ranking.get(results.get('interpretability', 'Low'), 1)
                               for method, results in comparison_results.items()}
                if interp_scores:
                    summary['best_interpretability'] = max(interp_scores, key=interp_scores.get)

                comparison_results['summary'] = summary

            comparison_results['comparison_date'] = datetime.now()
            self.reduction_results = comparison_results

            return comparison_results

        except Exception as e:
            return {'error': str(e)}

    def recommend_method(self, data: pd.DataFrame,
                        goal: str = 'variance') -> Dict[str, Any]:
        """
        Recommend dimensionality reduction method based on goals.

        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        goal : str
            Primary goal ('variance', 'interpretability', 'reduction')

        Returns:
        --------
        Dict with recommendation
        """
        try:
            # Run comparison if not already done
            if not self.reduction_results:
                comparison = self.compare_methods(data)
                if 'error' in comparison:
                    return comparison
            else:
                comparison = self.reduction_results

            if 'summary' not in comparison:
                return {'error': 'No valid comparison results available'}

            summary = comparison['summary']
            recommendation = {}

            if goal == 'variance':
                best_method = summary.get('best_variance_explained')
                recommendation = {
                    'recommended_method': best_method,
                    'reason': 'Maximizes explained variance',
                    'goal': goal
                }

            elif goal == 'interpretability':
                best_method = summary.get('best_interpretability')
                recommendation = {
                    'recommended_method': best_method,
                    'reason': 'Maintains interpretability of original variables',
                    'goal': goal
                }

            elif goal == 'reduction':
                best_method = summary.get('best_dimension_reduction')
                recommendation = {
                    'recommended_method': best_method,
                    'reason': 'Achieves maximum dimension reduction',
                    'goal': goal
                }

            else:
                # Balanced recommendation
                methods = list(comparison.keys())
                if 'summary' in methods:
                    methods.remove('summary')

                # Score methods on multiple criteria
                scores = {}
                for method in methods:
                    if method in comparison:
                        variance_score = comparison[method].get('variance_explained', 0)
                        reduction_score = comparison[method].get('dimension_reduction', 0)
                        interp_score = {'High': 1, 'Medium': 0.5, 'Low': 0}.get(
                            comparison[method].get('interpretability', 'Low'), 0
                        )

                        # Weighted score
                        total_score = 0.4 * variance_score + 0.3 * reduction_score + 0.3 * interp_score
                        scores[method] = total_score

                best_method = max(scores, key=scores.get) if scores else None
                recommendation = {
                    'recommended_method': best_method,
                    'reason': 'Best overall balance of variance, reduction, and interpretability',
                    'goal': 'balanced',
                    'method_scores': scores
                }

            # Add method details
            if recommendation['recommended_method'] and recommendation['recommended_method'] in comparison:
                recommendation['method_details'] = comparison[recommendation['recommended_method']]

            recommendation['recommendation_date'] = datetime.now()

            return recommendation

        except Exception as e:
            return {'error': str(e)}


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline with encoders and imputers.
    """

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.preprocessor = None
        self.feature_names = None
        self.preprocessing_steps = {}

    def create_preprocessor(self, X: pd.DataFrame,
                          categorical_features: Optional[List[str]] = None,
                          numerical_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create preprocessing pipeline for features.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        categorical_features : List[str], optional
            List of categorical feature names
        numerical_features : List[str], optional
            List of numerical feature names

        Returns:
        --------
        Dict with preprocessor and metadata
        """
        try:
            # Auto-detect feature types if not provided
            if categorical_features is None:
                categorical_features = list(X.select_dtypes(include=['object', 'category']).columns)

            if numerical_features is None:
                numerical_features = list(X.select_dtypes(include=[np.number]).columns)

            # Remove overlap
            categorical_features = [col for col in categorical_features if col not in numerical_features]

            # Create numerical pipeline
            numerical_steps = []

            # Imputation
            if self.config.numerical_imputer == 'simple':
                numerical_steps.append(('imputer', SimpleImputer(strategy='median')))
            elif self.config.numerical_imputer == 'knn':
                try:
                    numerical_steps.append(('imputer', KNNImputer(n_neighbors=5)))
                except ImportError:
                    warnings.warn("KNNImputer not available, using SimpleImputer")
                    numerical_steps.append(('imputer', SimpleImputer(strategy='median')))

            # Scaling
            if self.config.numerical_scaler == 'standard':
                numerical_steps.append(('scaler', StandardScaler()))
            elif self.config.numerical_scaler == 'minmax':
                numerical_steps.append(('scaler', MinMaxScaler()))
            elif self.config.numerical_scaler == 'robust':
                numerical_steps.append(('scaler', RobustScaler()))

            numerical_pipeline = Pipeline(numerical_steps)

            # Create categorical pipeline
            categorical_steps = []

            # Imputation
            if self.config.categorical_imputer == 'most_frequent':
                categorical_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
            elif self.config.categorical_imputer == 'constant':
                categorical_steps.append(('imputer', SimpleImputer(strategy='constant', fill_value='missing')))

            # Encoding
            if self.config.categorical_encoder == 'onehot':
                categorical_steps.append(('encoder', OneHotEncoder(
                    handle_unknown=self.config.handle_unknown_categories,
                    sparse_output=False
                )))
            elif self.config.categorical_encoder == 'ordinal':
                categorical_steps.append(('encoder', OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                )))
            elif self.config.categorical_encoder == 'label':
                categorical_steps.append(('encoder', LabelEncoder()))

            categorical_pipeline = Pipeline(categorical_steps)

            # Combine pipelines
            transformers = []

            if numerical_features:
                transformers.append(('num', numerical_pipeline, numerical_features))

            if categorical_features:
                transformers.append(('cat', categorical_pipeline, categorical_features))

            if not transformers:
                return {'error': 'No features to process'}

            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough'
            )

            # Store metadata
            self.preprocessing_steps = {
                'numerical_features': numerical_features,
                'categorical_features': categorical_features,
                'numerical_pipeline': numerical_steps,
                'categorical_pipeline': categorical_steps,
                'total_features': len(numerical_features) + len(categorical_features)
            }

            result = {
                'preprocessor': self.preprocessor,
                'numerical_features': numerical_features,
                'categorical_features': categorical_features,
                'preprocessing_steps': self.preprocessing_steps,
                'creation_date': datetime.now()
            }

            return result

        except Exception as e:
            return {'error': str(e)}

    def fit_transform(self, X: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fit preprocessor and transform data."""
        try:
            if self.preprocessor is None:
                prep_result = self.create_preprocessor(X)
                if 'error' in prep_result:
                    return None, prep_result

            # Fit and transform
            X_transformed = self.preprocessor.fit_transform(X)

            # Generate feature names for transformed data
            feature_names = self._get_feature_names_out(X)

            transform_info = {
                'original_shape': X.shape,
                'transformed_shape': X_transformed.shape,
                'feature_names': feature_names,
                'transform_date': datetime.now()
            }

            self.feature_names = feature_names

            return X_transformed, transform_info

        except Exception as e:
            return None, {'error': str(e)}

    def transform(self, X: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Transform new data using fitted preprocessor."""
        try:
            if self.preprocessor is None:
                return None, {'error': 'Preprocessor not fitted. Call fit_transform first.'}

            X_transformed = self.preprocessor.transform(X)

            transform_info = {
                'original_shape': X.shape,
                'transformed_shape': X_transformed.shape,
                'feature_names': self.feature_names,
                'transform_date': datetime.now()
            }

            return X_transformed, transform_info

        except Exception as e:
            return None, {'error': str(e)}

    def _get_feature_names_out(self, X: pd.DataFrame) -> List[str]:
        """Generate feature names for transformed data."""
        try:
            feature_names = []

            for name, transformer, features in self.preprocessor.transformers_:
                if name == 'remainder':
                    continue

                if name == 'num':
                    # Numerical features keep their names
                    feature_names.extend(features)
                elif name == 'cat':
                    # Categorical features depend on encoder type
                    if self.config.categorical_encoder == 'onehot':
                        # Get encoded feature names from OneHotEncoder
                        encoder = transformer.named_steps['encoder']
                        try:
                            if hasattr(encoder, 'get_feature_names_out'):
                                encoded_names = encoder.get_feature_names_out(features)
                            else:
                                # Fallback for older sklearn versions
                                encoded_names = []
                                for i, feature in enumerate(features):
                                    categories = encoder.categories_[i]
                                    for category in categories:
                                        encoded_names.append(f"{feature}_{category}")
                            feature_names.extend(encoded_names)
                        except:
                            # Fallback
                            feature_names.extend([f"{feat}_encoded" for feat in features])
                    else:
                        # For ordinal/label encoding, use original names
                        feature_names.extend(features)

            return feature_names

        except Exception:
            # Fallback to generic names
            n_features = self.preprocessor.transform(X).shape[1]
            return [f'feature_{i}' for i in range(n_features)]


class ModelFactory:
    """
    Factory class for creating various machine learning models.
    """

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    def create_linear_models(self, problem_type: str = 'classification') -> Dict[str, Any]:
        """Create linear models for regression or classification."""
        models = {}

        if problem_type == 'regression':
            models.update({
                'linear_regression': LinearRegression(),
                'lasso': Lasso(alpha=1.0, max_iter=self.config.max_iter),
                'ridge': Ridge(alpha=1.0, max_iter=self.config.max_iter),
                'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=self.config.max_iter)
            })
        else:  # classification
            models.update({
                'logistic_regression': LogisticRegression(
                    max_iter=self.config.max_iter,
                    random_state=self.config.random_state
                ),
                'lda': LinearDiscriminantAnalysis(),
                'qda': QuadraticDiscriminantAnalysis()
            })

        return models

    def create_tree_models(self, problem_type: str = 'classification') -> Dict[str, Any]:
        """Create tree-based models."""
        models = {}

        if problem_type == 'regression':
            models.update({
                'decision_tree': DecisionTreeRegressor(random_state=self.config.random_state),
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.config.random_state
                ),
                'extra_trees': ExtraTreesRegressor(
                    n_estimators=100,
                    random_state=self.config.random_state
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    random_state=self.config.random_state
                ),
                'ada_boost': AdaBoostRegressor(random_state=self.config.random_state)
            })
        else:  # classification
            models.update({
                'decision_tree': DecisionTreeClassifier(random_state=self.config.random_state),
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.config.random_state
                ),
                'extra_trees': ExtraTreesClassifier(
                    n_estimators=100,
                    random_state=self.config.random_state
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=self.config.random_state
                ),
                'ada_boost': AdaBoostClassifier(random_state=self.config.random_state)
            })

        return models

    def create_other_models(self, problem_type: str = 'classification') -> Dict[str, Any]:
        """Create other specialized models."""
        models = {}

        if problem_type == 'classification':
            models.update({
                'gaussian_nb': GaussianNB(),
                'multinomial_nb': MultinomialNB(),
                'svm_linear': LinearSVC(random_state=self.config.random_state, max_iter=self.config.max_iter),
                'svm_rbf': SVC(kernel='rbf', probability=True, random_state=self.config.random_state)
            })
        else:  # regression
            models.update({
                'svr_linear': LinearSVR(random_state=self.config.random_state, max_iter=self.config.max_iter),
                'svr_rbf': SVR(kernel='rbf')
            })

        return models

    def create_ensemble_models(self, base_models: Dict[str, Any],
                             problem_type: str = 'classification') -> Dict[str, Any]:
        """Create ensemble models from base models."""
        ensemble_models = {}

        try:
            if len(base_models) < 2:
                return {}

            # Select a subset of base models for ensemble
            selected_models = list(base_models.items())[:5]  # Limit to 5 models

            if problem_type == 'classification':
                # Voting classifier
                ensemble_models['voting_soft'] = VotingClassifier(
                    estimators=selected_models,
                    voting='soft'
                )
                ensemble_models['voting_hard'] = VotingClassifier(
                    estimators=selected_models,
                    voting='hard'
                )

                # Bagging with best base model
                best_model_name, best_model = selected_models[0]
                ensemble_models['bagging'] = BaggingClassifier(
                    estimator=best_model,
                    n_estimators=10,
                    random_state=self.config.random_state
                )

            else:  # regression
                ensemble_models['voting'] = VotingRegressor(estimators=selected_models)

                # Bagging with best base model
                best_model_name, best_model = selected_models[0]
                ensemble_models['bagging'] = BaggingRegressor(
                    estimator=best_model,
                    n_estimators=10,
                    random_state=self.config.random_state
                )

        except Exception as e:
            warnings.warn(f"Error creating ensemble models: {str(e)}")

        return ensemble_models

    def create_boosting_models(self, problem_type: str = 'classification') -> Dict[str, Any]:
        """Create boosting models including XGBoost and LightGBM if available."""
        models = {}

        # Add XGBoost if available
        try:
            import xgboost as xgb
            if problem_type == 'classification':
                models['xgboost'] = xgb.XGBClassifier(
                    random_state=self.config.random_state,
                    eval_metric='logloss'
                )
            else:
                models['xgboost'] = xgb.XGBRegressor(
                    random_state=self.config.random_state
                )
        except ImportError:
            warnings.warn("XGBoost not available")

        # Add LightGBM if available
        try:
            import lightgbm as lgb
            if problem_type == 'classification':
                models['lightgbm'] = lgb.LGBMClassifier(
                    random_state=self.config.random_state,
                    verbose=-1
                )
            else:
                models['lightgbm'] = lgb.LGBMRegressor(
                    random_state=self.config.random_state,
                    verbose=-1
                )
        except ImportError:
            warnings.warn("LightGBM not available")

        return models

    def get_all_models(self, problem_type: str = 'classification') -> Dict[str, Any]:
        """Get all available models for the specified problem type."""
        all_models = {}

        # Add linear models
        all_models.update(self.create_linear_models(problem_type))

        # Add tree models
        all_models.update(self.create_tree_models(problem_type))

        # Add other models
        all_models.update(self.create_other_models(problem_type))

        # Add boosting models
        all_models.update(self.create_boosting_models(problem_type))

        # Add ensemble models
        ensemble_models = self.create_ensemble_models(all_models, problem_type)
        all_models.update(ensemble_models)

        return all_models


class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison.
    """

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    def determine_problem_type(self, y: pd.Series) -> str:
        """Automatically determine if problem is classification or regression."""
        try:
            # Check if target is continuous or discrete
            unique_values = y.nunique()
            total_values = len(y)

            # Heuristic: if unique values / total values < 0.05 or unique values < 20,
            # it's likely classification
            if unique_values / total_values < 0.05 or unique_values <= 20:
                return 'classification'
            else:
                return 'regression'

        except Exception:
            return 'classification'  # Default to classification

    def get_scoring_metric(self, problem_type: str) -> str:
        """Get appropriate scoring metric for the problem type."""
        if self.config.scoring_metric != 'auto':
            return self.config.scoring_metric

        if problem_type == 'classification':
            return 'accuracy'
        else:
            return 'r2'

    def evaluate_model(self, model, X_train: np.ndarray, y_train: pd.Series,
                      X_test: np.ndarray, y_test: pd.Series,
                      problem_type: str) -> Dict[str, Any]:
        """Evaluate a single model."""
        try:
            # Fit model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            evaluation = {
                'model_name': model.__class__.__name__,
                'problem_type': problem_type,
                'predictions': y_pred.tolist(),
                'evaluation_date': datetime.now()
            }

            if problem_type == 'classification':
                evaluation.update({
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                })

                # Add ROC AUC for binary classification
                if len(np.unique(y_test)) == 2:
                    try:
                        if hasattr(model, 'predict_proba'):
                            y_proba = model.predict_proba(X_test)[:, 1]
                        elif hasattr(model, 'decision_function'):
                            y_proba = model.decision_function(X_test)
                        else:
                            y_proba = y_pred

                        evaluation['roc_auc'] = roc_auc_score(y_test, y_proba)
                    except Exception as e:
                        warnings.warn(f"Could not calculate ROC AUC: {str(e)}")

                # Confusion matrix
                evaluation['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()

            else:  # regression
                evaluation.update({
                    'mse': mean_squared_error(y_test, y_pred),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                })

            # Cross-validation if enabled
            if self.config.enable_cross_validation:
                try:
                    scoring = self.get_scoring_metric(problem_type)
                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=self.config.cv_folds,
                        scoring=scoring
                    )
                    evaluation['cv_scores'] = cv_scores.tolist()
                    evaluation['cv_mean'] = cv_scores.mean()
                    evaluation['cv_std'] = cv_scores.std()
                except Exception as e:
                    warnings.warn(f"Cross-validation failed: {str(e)}")

            return evaluation

        except Exception as e:
            return {
                'model_name': model.__class__.__name__,
                'error': str(e),
                'evaluation_date': datetime.now()
            }

    def compare_models(self, models: Dict[str, Any], X_train: np.ndarray, y_train: pd.Series,
                      X_test: np.ndarray, y_test: pd.Series,
                      problem_type: str) -> Dict[str, Any]:
        """Compare multiple models."""
        try:
            model_results = {}
            scoring_metric = self.get_scoring_metric(problem_type)

            for model_name, model in models.items():
                print(f"Evaluating {model_name}...")
                evaluation = self.evaluate_model(model, X_train, y_train, X_test, y_test, problem_type)
                model_results[model_name] = evaluation

            # Rank models
            rankings = self._rank_models(model_results, problem_type)

            comparison_result = {
                'model_results': model_results,
                'rankings': rankings,
                'problem_type': problem_type,
                'scoring_metric': scoring_metric,
                'total_models': len(models),
                'comparison_date': datetime.now()
            }

            return comparison_result

        except Exception as e:
            return {'error': str(e)}

    def _rank_models(self, model_results: Dict[str, Any], problem_type: str) -> Dict[str, Any]:
        """Rank models based on performance."""
        try:
            valid_results = {name: result for name, result in model_results.items()
                           if 'error' not in result}

            if not valid_results:
                return {'error': 'No valid model results to rank'}

            # Determine ranking metric
            if problem_type == 'classification':
                if self.config.enable_cross_validation:
                    ranking_metric = 'cv_mean'
                else:
                    ranking_metric = 'accuracy'
                ascending = False  # Higher is better
            else:  # regression
                if self.config.enable_cross_validation:
                    ranking_metric = 'cv_mean'
                    ascending = False  # R2 score - higher is better
                else:
                    ranking_metric = 'r2'
                    ascending = False  # R2 score - higher is better

            # Create ranking
            model_scores = []
            for name, result in valid_results.items():
                if ranking_metric in result:
                    model_scores.append((name, result[ranking_metric]))

            # Sort by metric
            model_scores.sort(key=lambda x: x[1], reverse=not ascending)

            rankings = {
                'ranking_metric': ranking_metric,
                'ranked_models': [
                    {
                        'rank': i + 1,
                        'model_name': name,
                        'score': score
                    }
                    for i, (name, score) in enumerate(model_scores)
                ],
                'best_model': model_scores[0][0] if model_scores else None,
                'best_score': model_scores[0][1] if model_scores else None
            }

            return rankings

        except Exception as e:
            return {'error': str(e)}


class ModelPipeline:
    """
    Complete machine learning pipeline with preprocessing and modeling.
    """

    def __init__(self, config: FeatureEngineeringConfig = None):
        self.config = config or FeatureEngineeringConfig()
        self.preprocessor = DataPreprocessor(self.config)
        self.model_factory = ModelFactory(self.config)
        self.evaluator = ModelEvaluator(self.config)

        self.pipeline_results = {}
        self.best_model = None
        self.best_pipeline = None

    def create_complete_pipeline(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series,
                                models_to_try: Optional[List[str]] = None,
                                categorical_features: Optional[List[str]] = None,
                                numerical_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create and evaluate complete ML pipeline.

        Parameters:
        -----------
        X_train, X_test : pd.DataFrame
            Training and test features
        y_train, y_test : pd.Series
            Training and test targets
        models_to_try : List[str], optional
            Specific models to evaluate
        categorical_features, numerical_features : List[str], optional
            Feature type specifications

        Returns:
        --------
        Dict with complete pipeline results
        """
        try:
            pipeline_start = datetime.now()

            # Step 1: Determine problem type
            problem_type = self.evaluator.determine_problem_type(y_train)
            print(f"Detected problem type: {problem_type}")

            # Step 2: Create preprocessor
            print("Creating preprocessing pipeline...")
            prep_result = self.preprocessor.create_preprocessor(
                X_train, categorical_features, numerical_features
            )
            if 'error' in prep_result:
                return prep_result

            # Step 3: Fit and transform training data
            print("Preprocessing training data...")
            X_train_processed, train_transform_info = self.preprocessor.fit_transform(X_train)
            if X_train_processed is None:
                return train_transform_info

            # Step 4: Transform test data
            print("Preprocessing test data...")
            X_test_processed, test_transform_info = self.preprocessor.transform(X_test)
            if X_test_processed is None:
                return test_transform_info

            # Step 5: Get models to evaluate
            print("Preparing models...")
            all_models = self.model_factory.get_all_models(problem_type)

            if models_to_try:
                models = {name: model for name, model in all_models.items()
                         if name in models_to_try}
            else:
                models = all_models

            print(f"Evaluating {len(models)} models...")

            # Step 6: Evaluate models
            comparison_result = self.evaluator.compare_models(
                models, X_train_processed, y_train,
                X_test_processed, y_test, problem_type
            )

            if 'error' in comparison_result:
                return comparison_result

            # Step 7: Create best pipeline
            best_model_name = comparison_result['rankings']['best_model']
            if best_model_name:
                best_model = models[best_model_name]
                best_model.fit(X_train_processed, y_train)

                # Create complete pipeline
                complete_pipeline = Pipeline([
                    ('preprocessor', self.preprocessor.preprocessor),
                    ('model', best_model)
                ])

                self.best_model = best_model
                self.best_pipeline = complete_pipeline

            # Compile results
            pipeline_results = {
                'problem_type': problem_type,
                'preprocessing': {
                    'config': prep_result,
                    'train_transform': train_transform_info,
                    'test_transform': test_transform_info
                },
                'model_comparison': comparison_result,
                'best_model_name': best_model_name,
                'best_pipeline': complete_pipeline if best_model_name else None,
                'pipeline_duration': datetime.now() - pipeline_start,
                'pipeline_date': pipeline_start
            }

            self.pipeline_results = pipeline_results

            print(f"Pipeline completed! Best model: {best_model_name}")
            return pipeline_results

        except Exception as e:
            return {'error': str(e)}

    def predict(self, X_new: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Make predictions on new data using the best pipeline."""
        try:
            if self.best_pipeline is None:
                return None, {'error': 'No pipeline fitted. Run create_complete_pipeline first.'}

            predictions = self.best_pipeline.predict(X_new)

            prediction_info = {
                'n_predictions': len(predictions),
                'model_used': self.best_model.__class__.__name__,
                'prediction_date': datetime.now()
            }

            # Add prediction probabilities for classification
            if hasattr(self.best_model, 'predict_proba'):
                try:
                    probabilities = self.best_pipeline.predict_proba(X_new)
                    prediction_info['probabilities'] = probabilities.tolist()
                except:
                    pass

            return predictions, prediction_info

        except Exception as e:
            return None, {'error': str(e)}

    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance from the best model if available."""
        try:
            if self.best_model is None:
                return {'error': 'No model fitted'}

            importance_info = {
                'model_name': self.best_model.__class__.__name__,
                'feature_names': self.preprocessor.feature_names
            }

            # Try different methods to get feature importance
            if hasattr(self.best_model, 'feature_importances_'):
                importance_info['importances'] = self.best_model.feature_importances_.tolist()
                importance_info['method'] = 'feature_importances_'

            elif hasattr(self.best_model, 'coef_'):
                # For linear models, use absolute coefficients
                if len(self.best_model.coef_.shape) == 1:
                    importance_info['importances'] = np.abs(self.best_model.coef_).tolist()
                else:
                    # For multiclass, use mean of absolute coefficients
                    importance_info['importances'] = np.mean(np.abs(self.best_model.coef_), axis=0).tolist()
                importance_info['method'] = 'coefficients'

            else:
                return {'error': 'Model does not provide feature importance'}

            # Create feature importance ranking
            if 'importances' in importance_info and self.preprocessor.feature_names:
                feature_importance_pairs = list(zip(
                    self.preprocessor.feature_names,
                    importance_info['importances']
                ))
                feature_importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

                importance_info['ranked_features'] = [
                    {'feature': feature, 'importance': importance}
                    for feature, importance in feature_importance_pairs
                ]

            importance_info['analysis_date'] = datetime.now()

            return importance_info

        except Exception as e:
            return {'error': str(e)}


class FeatureEngineering:
    """
    Main feature engineering pipeline combining all components.
    """

    def __init__(self, config: FeatureEngineeringConfig = None):
        self.config = config or FeatureEngineeringConfig()
        self.outlier_detector = OutlierDetector(self.config)
        self.feature_selector = FeatureSelector(self.config)
        self.imbalance_handler = ClassImbalanceHandler(self.config)
        self.data_splitter = DataSplitter(self.config)
        self.dimensionality_reducer = DimensionalityReduction(self.config)

        # New components
        self.clustering = ClusteringAnalysis(self.config)
        self.model_analyzer = ModelAnalyzer(self.config)
        self.visualizer = ModelVisualizer(self.config)
        self.persistence = ModelPersistence(self.config)

        self.pipeline_history = []
        self.feature_matrix = None
        self.target_vector = None
        self.processed_data = {}

    def construct_feature_matrix(self, data: pd.DataFrame,
                                target_column: str,
                                feature_columns: List[str] = None,
                                exclude_columns: List[str] = None) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Construct feature matrix and target vector from raw data.

        Parameters:
        -----------
        data : pd.DataFrame
            Raw input dataset
        target_column : str
            Name of target column
        feature_columns : List[str], optional
            Specific columns to include as features
        exclude_columns : List[str], optional
            Columns to exclude from features

        Returns:
        --------
        Tuple of (X, y, construction_info)
        """
        try:
            if target_column not in data.columns:
                return None, None, {'error': f'Target column {target_column} not found'}

            # Extract target vector
            y = data[target_column].copy()

            # Determine feature columns
            if feature_columns is not None:
                # Use specified feature columns
                missing_cols = [col for col in feature_columns if col not in data.columns]
                if missing_cols:
                    warnings.warn(f"Feature columns not found: {missing_cols}")
                    feature_columns = [col for col in feature_columns if col in data.columns]
                X = data[feature_columns].copy()
            else:
                # Use all columns except target and excluded columns
                exclude_list = [target_column]
                if exclude_columns:
                    exclude_list.extend(exclude_columns)

                feature_columns = [col for col in data.columns if col not in exclude_list]
                X = data[feature_columns].copy()

            construction_info = {
                'target_column': target_column,
                'feature_columns': list(X.columns),
                'excluded_columns': exclude_columns or [],
                'feature_count': len(X.columns),
                'sample_count': len(X),
                'target_type': str(y.dtype),
                'feature_types': X.dtypes.to_dict(),
                'missing_values': X.isnull().sum().to_dict(),
                'construction_date': datetime.now()
            }

            # Store for pipeline
            self.feature_matrix = X
            self.target_vector = y

            return X, y, construction_info

        except Exception as e:
            return None, None, {'error': str(e)}

    def run_full_pipeline(self, data: pd.DataFrame,
                         target_column: str,
                         feature_columns: List[str] = None,
                         exclude_columns: List[str] = None,
                         split_method: str = 'basic',
                         **split_kwargs) -> Dict[str, Any]:
        """
        Run the complete feature engineering pipeline.

        Parameters:
        -----------
        data : pd.DataFrame
            Raw input dataset
        target_column : str
            Target column name
        feature_columns : List[str], optional
            Specific feature columns to use
        exclude_columns : List[str], optional
            Columns to exclude
        split_method : str
            Data splitting method
        **split_kwargs : additional splitting arguments

        Returns:
        --------
        Dict containing all pipeline results
        """
        try:
            pipeline_start = datetime.now()
            pipeline_results = {
                'pipeline_config': self.config.__dict__,
                'pipeline_start': pipeline_start
            }

            # Step 1: Construct feature matrix and target vector
            print("Step 1: Constructing feature matrix and target vector...")
            X, y, construction_info = self.construct_feature_matrix(
                data, target_column, feature_columns, exclude_columns
            )

            if X is None:
                return {'error': 'Feature matrix construction failed', 'details': construction_info}

            pipeline_results['construction'] = construction_info

            # Step 2: Outlier detection and removal
            print("Step 2: Detecting and removing outliers...")
            outlier_results = self.outlier_detector.detect_outliers(X)
            if 'error' not in outlier_results:
                X, outlier_removal_summary = self.outlier_detector.remove_outliers(X, outlier_results)
                y = y.loc[X.index]  # Align target with cleaned features
                pipeline_results['outlier_detection'] = outlier_results
                pipeline_results['outlier_removal'] = outlier_removal_summary
            else:
                pipeline_results['outlier_detection'] = outlier_results

            # Step 3: Feature selection and cleaning
            print("Step 3: Feature selection and cleaning...")
            X_cleaned, feature_drop_summary = self.feature_selector.drop_problematic_features(
                X, target_column
            )
            y_cleaned = y.loc[X_cleaned.index]  # Align target
            pipeline_results['feature_cleaning'] = feature_drop_summary

            # Step 4: Class imbalance handling
            print("Step 4: Handling class imbalance...")
            class_analysis = self.imbalance_handler.analyze_class_distribution(y_cleaned)
            pipeline_results['class_analysis'] = class_analysis

            if class_analysis.get('is_imbalanced', False):
                X_balanced, y_balanced, sampling_info = self.imbalance_handler.balance_classes(
                    X_cleaned, y_cleaned
                )
                pipeline_results['class_balancing'] = sampling_info
            else:
                X_balanced, y_balanced = X_cleaned, y_cleaned
                pipeline_results['class_balancing'] = {'method': 'none', 'reason': 'Already balanced'}

            # Step 5: Dimensionality reduction (optional)
            print("Step 5: Performing dimensionality reduction...")
            if hasattr(self.config, 'apply_dimensionality_reduction') and getattr(self.config, 'apply_dimensionality_reduction', False):
                reduction_method = getattr(self.config, 'dimensionality_reduction_method', 'pca')

                if reduction_method == 'pca':
                    reduction_result = self.dimensionality_reducer.pca.fit_pca(X_balanced)
                    if 'error' not in reduction_result:
                        X_reduced = reduction_result['scores']
                        pipeline_results['dimensionality_reduction'] = {
                            'method': 'pca',
                            'original_dimensions': X_balanced.shape[1],
                            'reduced_dimensions': X_reduced.shape[1],
                            'variance_explained': sum(reduction_result['explained_variance_ratio']),
                            'results': reduction_result
                        }
                        X_final = X_reduced
                    else:
                        X_final = X_balanced
                        pipeline_results['dimensionality_reduction'] = reduction_result

                elif reduction_method == 'fa':
                    reduction_result = self.dimensionality_reducer.fa.fit_factor_analysis(X_balanced)
                    if 'error' not in reduction_result:
                        X_reduced = reduction_result['scores']
                        pipeline_results['dimensionality_reduction'] = {
                            'method': 'fa',
                            'original_dimensions': X_balanced.shape[1],
                            'reduced_dimensions': X_reduced.shape[1],
                            'variance_explained': sum(reduction_result['proportion_variance']),
                            'results': reduction_result
                        }
                        X_final = X_reduced
                    else:
                        X_final = X_balanced
                        pipeline_results['dimensionality_reduction'] = reduction_result

                elif reduction_method == 'varclus':
                    varclus_result = self.dimensionality_reducer.varclus.varclus_analysis(X_balanced)
                    if 'error' not in varclus_result:
                        representatives = self.dimensionality_reducer.varclus.select_cluster_representatives(X_balanced)
                        if 'error' not in representatives:
                            representative_vars = representatives['representative_variables']
                            X_reduced = X_balanced[representative_vars]
                            pipeline_results['dimensionality_reduction'] = {
                                'method': 'varclus',
                                'original_dimensions': X_balanced.shape[1],
                                'reduced_dimensions': X_reduced.shape[1],
                                'reduction_ratio': representatives['reduction_ratio'],
                                'results': {
                                    'clustering': varclus_result,
                                    'representatives': representatives
                                }
                            }
                            X_final = X_reduced
                        else:
                            X_final = X_balanced
                            pipeline_results['dimensionality_reduction'] = representatives
                    else:
                        X_final = X_balanced
                        pipeline_results['dimensionality_reduction'] = varclus_result

                else:
                    X_final = X_balanced
                    pipeline_results['dimensionality_reduction'] = {'error': f'Unknown method: {reduction_method}'}

            else:
                X_final = X_balanced
                pipeline_results['dimensionality_reduction'] = {'method': 'none', 'reason': 'Not requested'}

            # Step 6: Train-validation-test split
            print("Step 6: Splitting data...")
            split_results = self.data_splitter.split_data(
                X_final, y_balanced, split_method, **split_kwargs
            )

            if 'error' not in split_results:
                pipeline_results['data_splits'] = split_results
                pipeline_results['final_datasets'] = {
                    'X_train': split_results['X_train'],
                    'X_val': split_results['X_val'],
                    'X_test': split_results['X_test'],
                    'y_train': split_results['y_train'],
                    'y_val': split_results['y_val'],
                    'y_test': split_results['y_test']
                }
            else:
                pipeline_results['data_splits'] = split_results

            # Pipeline summary
            pipeline_end = datetime.now()
            pipeline_results.update({
                'pipeline_end': pipeline_end,
                'pipeline_duration': pipeline_end - pipeline_start,
                'pipeline_success': 'error' not in split_results,
                'final_feature_count': len(X_balanced.columns) if X_balanced is not None else 0,
                'final_sample_count': len(X_balanced) if X_balanced is not None else 0
            })

            # Store results
            self.pipeline_history.append(pipeline_results)
            self.processed_data = pipeline_results.get('final_datasets', {})

            print(f"Pipeline completed in {pipeline_results['pipeline_duration']}")
            print(f"Final dataset: {pipeline_results['final_feature_count']} features, {pipeline_results['final_sample_count']} samples")

            return pipeline_results

        except Exception as e:
            return {'error': str(e), 'pipeline_step': 'unknown'}

    def generate_feature_report(self, pipeline_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive feature engineering report."""
        try:
            if pipeline_results is None:
                if not self.pipeline_history:
                    return {'error': 'No pipeline results available'}
                pipeline_results = self.pipeline_history[-1]

            report = {
                'report_generation_date': datetime.now(),
                'pipeline_summary': {},
                'data_quality_issues': {},
                'feature_transformations': {},
                'recommendations': []
            }

            # Pipeline summary
            if 'construction' in pipeline_results:
                construction = pipeline_results['construction']
                report['pipeline_summary'] = {
                    'original_features': construction['feature_count'],
                    'original_samples': construction['sample_count'],
                    'target_column': construction['target_column'],
                    'target_type': construction['target_type']
                }

            # Add final counts if available
            if 'final_feature_count' in pipeline_results:
                report['pipeline_summary']['final_features'] = pipeline_results['final_feature_count']
                report['pipeline_summary']['final_samples'] = pipeline_results['final_sample_count']
                report['pipeline_summary']['features_dropped'] = (
                    construction['feature_count'] - pipeline_results['final_feature_count']
                )

            # Data quality issues
            if 'outlier_detection' in pipeline_results:
                outlier_info = pipeline_results['outlier_detection']
                report['data_quality_issues']['outliers'] = {
                    'detection_method': outlier_info.get('method'),
                    'total_outliers': outlier_info.get('total_outliers', 0),
                    'outlier_percentage': outlier_info.get('total_percentage', 0)
                }

            if 'feature_cleaning' in pipeline_results:
                cleaning_info = pipeline_results['feature_cleaning']
                report['data_quality_issues']['feature_issues'] = {
                    'total_dropped': cleaning_info.get('total_dropped', 0),
                    'drop_percentage': cleaning_info.get('drop_percentage', 0),
                    'drop_reasons': cleaning_info.get('drop_reasons', {})
                }

            # Feature transformations
            if 'class_balancing' in pipeline_results:
                balancing_info = pipeline_results['class_balancing']
                report['feature_transformations']['class_balancing'] = {
                    'method': balancing_info.get('method'),
                    'was_needed': balancing_info.get('method') != 'none',
                    'original_distribution': balancing_info.get('original_distribution'),
                    'new_distribution': balancing_info.get('new_distribution')
                }

            # Generate recommendations
            recommendations = []

            # Outlier recommendations
            if report['data_quality_issues'].get('outliers', {}).get('outlier_percentage', 0) > 10:
                recommendations.append("High outlier percentage detected. Consider investigating data collection process.")

            # Feature dropping recommendations
            drop_percentage = report['data_quality_issues'].get('feature_issues', {}).get('drop_percentage', 0)
            if drop_percentage > 50:
                recommendations.append("Over 50% of features were dropped. Consider improving data quality.")

            # Class imbalance recommendations
            balancing = report['feature_transformations'].get('class_balancing', {})
            if balancing.get('was_needed', False):
                recommendations.append(f"Class imbalance was addressed using {balancing['method']}. Monitor model performance on minority classes.")

            report['recommendations'] = recommendations

            return report

        except Exception as e:
            return {'error': str(e)}

    def save_pipeline_results(self, pipeline_results: Dict[str, Any],
                            file_path: str) -> Dict[str, Any]:
        """Save pipeline results to file."""
        try:
            # Prepare data for saving (remove DataFrames, keep metadata)
            save_data = {}

            for key, value in pipeline_results.items():
                if key == 'final_datasets':
                    # Save dataset shapes and column info instead of full data
                    save_data[key] = {
                        'X_train_shape': value['X_train'].shape if 'X_train' in value else None,
                        'X_val_shape': value['X_val'].shape if 'X_val' in value else None,
                        'X_test_shape': value['X_test'].shape if 'X_test' in value else None,
                        'feature_columns': list(value['X_train'].columns) if 'X_train' in value else None
                    }
                elif isinstance(value, (pd.DataFrame, pd.Series)):
                    # Skip DataFrames and Series
                    continue
                elif isinstance(value, dict):
                    # Recursively handle nested dictionaries
                    nested_dict = {}
                    for k, v in value.items():
                        if not isinstance(v, (pd.DataFrame, pd.Series)):
                            nested_dict[k] = v
                    save_data[key] = nested_dict
                else:
                    save_data[key] = value

            # Convert datetime objects to strings
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                else:
                    return obj

            save_data = convert_datetime(save_data)

            # Save to JSON
            with open(file_path, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)

            return {
                'save_path': file_path,
                'save_date': datetime.now(),
                'success': True
            }

        except Exception as e:
            return {'error': str(e), 'save_path': file_path}

    def apply_dimensionality_reduction(self, data: pd.DataFrame,
                                     method: str = 'auto',
                                     n_components: Optional[int] = None) -> Dict[str, Any]:
        """
        Apply dimensionality reduction to data.

        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        method : str
            Method to use ('auto', 'pca', 'fa', 'varclus')
        n_components : int, optional
            Number of components/factors to retain

        Returns:
        --------
        Dict with reduction results
        """
        try:
            if method == 'auto':
                # Automatically recommend best method
                recommendation = self.dimensionality_reducer.recommend_method(data)
                if 'error' in recommendation:
                    return recommendation

                method = recommendation.get('recommended_method', 'pca')

            # Apply the selected method
            if method == 'pca':
                result = self.dimensionality_reducer.pca.fit_pca(
                    data, n_components=n_components
                )
                if 'error' not in result:
                    result['reduced_data'] = result['scores']

            elif method == 'fa':
                result = self.dimensionality_reducer.fa.fit_factor_analysis(
                    data, n_factors=n_components
                )
                if 'error' not in result:
                    result['reduced_data'] = result['scores']

            elif method == 'varclus':
                varclus_result = self.dimensionality_reducer.varclus.varclus_analysis(data)
                if 'error' not in varclus_result:
                    representatives = self.dimensionality_reducer.varclus.select_cluster_representatives(data)
                    if 'error' not in representatives:
                        representative_vars = representatives['representative_variables']
                        reduced_data = data[representative_vars]
                        result = {
                            'method': 'varclus',
                            'reduced_data': reduced_data,
                            'clustering_results': varclus_result,
                            'representative_selection': representatives,
                            'original_dimensions': data.shape[1],
                            'reduced_dimensions': reduced_data.shape[1],
                            'reduction_ratio': representatives['reduction_ratio']
                        }
                    else:
                        result = representatives
                else:
                    result = varclus_result

            else:
                result = {'error': f'Unknown method: {method}'}

            result['method_used'] = method
            result['application_date'] = datetime.now()

            return result

        except Exception as e:
            return {'error': str(e)}

    def compare_dimensionality_methods(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare all available dimensionality reduction methods.

        Parameters:
        -----------
        data : pd.DataFrame
            Input data

        Returns:
        --------
        Dict with comparison results
        """
        try:
            return self.dimensionality_reducer.compare_methods(data)

        except Exception as e:
            return {'error': str(e)}

    def get_eigenanalysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive eigenanalysis on data.

        Parameters:
        -----------
        data : pd.DataFrame
            Input data

        Returns:
        --------
        Dict with eigenanalysis results
        """
        try:
            # Select numeric columns
            numeric_data = data.select_dtypes(include=[np.number])

            if numeric_data.empty:
                return {'error': 'No numeric columns found'}

            # Handle missing values
            numeric_data_clean = numeric_data.fillna(numeric_data.mean())

            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(numeric_data_clean)

            # Compute correlation matrix
            correlation_matrix = np.corrcoef(data_scaled.T)

            # Perform eigenanalysis
            eigen_analyzer = self.dimensionality_reducer.pca.eigen_analysis
            eigen_results = eigen_analyzer.compute_eigendecomposition(correlation_matrix)

            if 'error' not in eigen_results:
                # Add parallel analysis
                parallel_results = eigen_analyzer.parallel_analysis(
                    pd.DataFrame(data_scaled, columns=numeric_data_clean.columns)
                )

                eigen_results['parallel_analysis'] = parallel_results

            return eigen_results

        except Exception as e:
            return {'error': str(e)}


class ClusteringAnalysis:
    """
    Comprehensive clustering analysis including hierarchical and K-means clustering.
    """

    def __init__(self, config: FeatureEngineeringConfig = None):
        self.config = config or FeatureEngineeringConfig()
        self.cluster_results = {}
        self.fitted_models = {}

    def hierarchical_clustering(self, data: pd.DataFrame,
                              method: str = 'ward',
                              metric: str = 'euclidean',
                              n_clusters: Optional[int] = None,
                              plot_dendrogram: bool = True) -> Dict[str, Any]:
        """
        Perform hierarchical clustering with dendrogram visualization.

        Parameters:
        -----------
        data : pd.DataFrame
            Input data for clustering
        method : str
            Linkage method ('ward', 'complete', 'average', 'single')
        metric : str
            Distance metric
        n_clusters : int, optional
            Number of clusters to form
        plot_dendrogram : bool
            Whether to create dendrogram plot

        Returns:
        --------
        Dict with clustering results and dendrogram
        """
        try:
            from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
            from scipy.spatial.distance import pdist
            import matplotlib.pyplot as plt

            # Prepare data
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                return {'error': 'No numeric columns found for clustering'}

            # Handle missing values
            data_clean = numeric_data.fillna(numeric_data.mean())

            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_clean)

            # Compute linkage matrix
            linkage_matrix = linkage(data_scaled, method=method, metric=metric)

            results = {
                'method': 'hierarchical',
                'linkage_method': method,
                'distance_metric': metric,
                'linkage_matrix': linkage_matrix,
                'data_shape': data_scaled.shape,
                'feature_names': list(numeric_data.columns),
                'analysis_date': datetime.now()
            }

            # Create dendrogram if requested
            if plot_dendrogram:
                try:
                    plt.figure(figsize=(12, 8))
                    dendrogram_data = dendrogram(
                        linkage_matrix,
                        leaf_rotation=90,
                        leaf_font_size=8,
                        show_leaf_counts=True
                    )
                    plt.title(f'Hierarchical Clustering Dendrogram ({method} linkage)')
                    plt.xlabel('Sample Index or (Cluster Size)')
                    plt.ylabel('Distance')

                    results['dendrogram'] = {
                        'leaves': dendrogram_data['leaves'],
                        'icoord': dendrogram_data['icoord'],
                        'dcoord': dendrogram_data['dcoord'],
                        'leaf_label_func': dendrogram_data.get('leaf_label_func')
                    }

                    # Save plot if possible
                    try:
                        import tempfile
                        import os
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                        plt.savefig(temp_file.name, dpi=300, bbox_inches='tight')
                        results['dendrogram_plot_path'] = temp_file.name
                        temp_file.close()
                    except:
                        pass

                    plt.close()

                except Exception as plot_error:
                    results['dendrogram_error'] = str(plot_error)

            # Form clusters if n_clusters specified
            if n_clusters is not None:
                cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                results['cluster_labels'] = cluster_labels.tolist()
                results['n_clusters'] = n_clusters

                # Compute cluster statistics
                cluster_stats = self._compute_cluster_statistics(
                    data_scaled, cluster_labels, numeric_data.columns
                )
                results['cluster_statistics'] = cluster_stats

            # Store results
            self.cluster_results['hierarchical'] = results

            return results

        except Exception as e:
            return {'error': str(e)}

    def kmeans_clustering(self, data: pd.DataFrame,
                         n_clusters: int = 3,
                         n_init: int = 10,
                         max_iter: int = 300,
                         random_state: int = 42,
                         find_optimal_k: bool = True) -> Dict[str, Any]:
        """
        Perform K-means clustering with optimal k selection.

        Parameters:
        -----------
        data : pd.DataFrame
            Input data for clustering
        n_clusters : int
            Number of clusters
        n_init : int
            Number of random initializations
        max_iter : int
            Maximum iterations
        random_state : int
            Random seed
        find_optimal_k : bool
            Whether to find optimal number of clusters

        Returns:
        --------
        Dict with clustering results
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score, calinski_harabasz_score

            # Prepare data
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                return {'error': 'No numeric columns found for clustering'}

            # Handle missing values
            data_clean = numeric_data.fillna(numeric_data.mean())

            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_clean)

            results = {
                'method': 'kmeans',
                'data_shape': data_scaled.shape,
                'feature_names': list(numeric_data.columns),
                'analysis_date': datetime.now()
            }

            # Find optimal k using elbow method and silhouette analysis
            if find_optimal_k:
                k_range = range(2, min(11, len(data_scaled) // 2))
                inertias = []
                silhouette_scores = []
                calinski_scores = []

                for k in k_range:
                    kmeans = KMeans(
                        n_clusters=k,
                        n_init=n_init,
                        max_iter=max_iter,
                        random_state=random_state
                    )
                    cluster_labels = kmeans.fit_predict(data_scaled)

                    inertias.append(kmeans.inertia_)
                    silhouette_scores.append(silhouette_score(data_scaled, cluster_labels))
                    calinski_scores.append(calinski_harabasz_score(data_scaled, cluster_labels))

                # Find elbow point
                optimal_k_elbow = self._find_elbow_point(list(k_range), inertias)
                optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]

                results['k_selection'] = {
                    'k_range': list(k_range),
                    'inertias': inertias,
                    'silhouette_scores': silhouette_scores,
                    'calinski_harabasz_scores': calinski_scores,
                    'optimal_k_elbow': optimal_k_elbow,
                    'optimal_k_silhouette': optimal_k_silhouette,
                    'recommended_k': optimal_k_silhouette  # Use silhouette as primary recommendation
                }

                # Use recommended k
                n_clusters = optimal_k_silhouette

            # Fit final model
            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=n_init,
                max_iter=max_iter,
                random_state=random_state
            )
            cluster_labels = kmeans.fit_predict(data_scaled)

            results.update({
                'n_clusters': n_clusters,
                'cluster_labels': cluster_labels.tolist(),
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'inertia': kmeans.inertia_,
                'n_iter': kmeans.n_iter_,
                'converged': kmeans.n_iter_ < max_iter
            })

            # Compute cluster statistics
            cluster_stats = self._compute_cluster_statistics(
                data_scaled, cluster_labels, numeric_data.columns
            )
            results['cluster_statistics'] = cluster_stats

            # Store fitted model
            self.fitted_models['kmeans'] = {
                'model': kmeans,
                'scaler': scaler,
                'feature_names': list(numeric_data.columns)
            }

            # Store results
            self.cluster_results['kmeans'] = results

            return results

        except Exception as e:
            return {'error': str(e)}

    def _compute_cluster_statistics(self, data_scaled: np.ndarray,
                                  cluster_labels: np.ndarray,
                                  feature_names: List[str]) -> Dict[str, Any]:
        """Compute comprehensive statistics for each cluster."""
        try:
            unique_clusters = np.unique(cluster_labels)
            cluster_stats = {}

            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_data = data_scaled[cluster_mask]

                stats = {
                    'size': int(np.sum(cluster_mask)),
                    'percentage': float(np.sum(cluster_mask) / len(cluster_labels) * 100),
                    'centroid': np.mean(cluster_data, axis=0).tolist(),
                    'std': np.std(cluster_data, axis=0).tolist(),
                    'min': np.min(cluster_data, axis=0).tolist(),
                    'max': np.max(cluster_data, axis=0).tolist()
                }

                # Feature importance within cluster (based on variance)
                feature_importance = np.var(cluster_data, axis=0)
                feature_ranking = sorted(
                    zip(feature_names, feature_importance),
                    key=lambda x: x[1], reverse=True
                )

                stats['feature_importance'] = [
                    {'feature': feat, 'variance': float(imp)}
                    for feat, imp in feature_ranking
                ]

                cluster_stats[f'cluster_{cluster_id}'] = stats

            return cluster_stats

        except Exception as e:
            return {'error': str(e)}

    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """Find elbow point in K-means inertia curve."""
        try:
            # Simple elbow detection using second derivative
            if len(inertias) < 3:
                return k_values[0]

            # Compute second differences
            differences = np.diff(inertias, 2)
            elbow_idx = np.argmax(differences) + 1  # +1 because of double diff

            return k_values[min(elbow_idx, len(k_values) - 1)]

        except Exception as e:
            return k_values[0] if k_values else 2

    def predict_clusters(self, new_data: pd.DataFrame,
                        method: str = 'kmeans') -> Dict[str, Any]:
        """Predict cluster membership for new data."""
        try:
            if method not in self.fitted_models:
                return {'error': f'No fitted {method} model found'}

            model_info = self.fitted_models[method]
            model = model_info['model']
            scaler = model_info['scaler']
            feature_names = model_info['feature_names']

            # Prepare new data
            new_data_features = new_data[feature_names]
            new_data_clean = new_data_features.fillna(new_data_features.mean())
            new_data_scaled = scaler.transform(new_data_clean)

            # Make predictions
            predictions = model.predict(new_data_scaled)

            return {
                'predictions': predictions.tolist(),
                'method': method,
                'n_samples': len(predictions),
                'prediction_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def get_cluster_centroids(self, method: str = 'kmeans') -> Dict[str, Any]:
        """Extract cluster centroids."""
        try:
            if method == 'kmeans':
                if 'kmeans' not in self.cluster_results:
                    return {'error': 'No K-means results found'}

                results = self.cluster_results['kmeans']
                feature_names = results['feature_names']

                centroids_info = {
                    'method': 'kmeans',
                    'n_clusters': results['n_clusters'],
                    'feature_names': feature_names,
                    'centroids': results['cluster_centers'],
                    'extraction_date': datetime.now()
                }

                # Create interpretable centroid descriptions
                centroids_df = pd.DataFrame(
                    results['cluster_centers'],
                    columns=feature_names
                )

                centroids_info['centroids_dataframe'] = centroids_df.to_dict('records')

                return centroids_info

            else:
                return {'error': f'Centroid extraction not supported for {method}'}

        except Exception as e:
            return {'error': str(e)}


class ModelAnalyzer:
    """
    Comprehensive model analysis including baseline predictions, error analysis,
    parameter extraction, SHAP values, and hyperparameter tuning.
    """

    def __init__(self, config: FeatureEngineeringConfig = None):
        self.config = config or FeatureEngineeringConfig()
        self.fitted_models = {}
        self.analysis_results = {}

    def compute_baseline_prediction(self, y_train: pd.Series,
                                  problem_type: str = 'auto') -> Dict[str, Any]:
        """
        Compute baseline prediction for comparison.

        Parameters:
        -----------
        y_train : pd.Series
            Training target values
        problem_type : str
            'classification', 'regression', or 'auto'

        Returns:
        --------
        Dict with baseline prediction info
        """
        try:
            if problem_type == 'auto':
                # Auto-detect problem type
                if pd.api.types.is_numeric_dtype(y_train):
                    if len(y_train.unique()) <= 10:
                        problem_type = 'classification'
                    else:
                        problem_type = 'regression'
                else:
                    problem_type = 'classification'

            baseline_info = {
                'problem_type': problem_type,
                'target_distribution': y_train.value_counts().to_dict(),
                'n_samples': len(y_train),
                'baseline_date': datetime.now()
            }

            if problem_type == 'classification':
                # Most frequent class
                baseline_prediction = y_train.mode().iloc[0]
                baseline_accuracy = (y_train == baseline_prediction).mean()

                baseline_info.update({
                    'baseline_prediction': baseline_prediction,
                    'baseline_accuracy': float(baseline_accuracy),
                    'baseline_method': 'most_frequent_class'
                })

            elif problem_type == 'regression':
                # Mean prediction
                baseline_prediction = y_train.mean()
                baseline_mse = np.mean((y_train - baseline_prediction) ** 2)
                baseline_mae = np.mean(np.abs(y_train - baseline_prediction))

                baseline_info.update({
                    'baseline_prediction': float(baseline_prediction),
                    'baseline_mse': float(baseline_mse),
                    'baseline_mae': float(baseline_mae),
                    'baseline_method': 'mean_prediction'
                })

            return baseline_info

        except Exception as e:
            return {'error': str(e)}

    def fit_model_with_analysis(self, model, X_train: pd.DataFrame,
                              y_train: pd.Series, X_test: pd.DataFrame = None,
                              y_test: pd.Series = None,
                              model_name: str = None) -> Dict[str, Any]:
        """
        Fit model and perform comprehensive analysis.

        Parameters:
        -----------
        model : sklearn model
            Model to fit and analyze
        X_train, y_train : training data
        X_test, y_test : test data (optional)
        model_name : str
            Name for the model

        Returns:
        --------
        Dict with comprehensive model analysis
        """
        try:
            model_name = model_name or model.__class__.__name__

            # Fit the model
            fit_start = datetime.now()
            model.fit(X_train, y_train)
            fit_time = datetime.now() - fit_start

            # Initialize results
            results = {
                'model_name': model_name,
                'model_class': model.__class__.__name__,
                'fit_time': fit_time.total_seconds(),
                'fit_date': fit_start,
                'training_samples': len(X_train),
                'features': list(X_train.columns) if hasattr(X_train, 'columns') else X_train.shape[1]
            }

            # Training predictions and errors
            train_predictions = model.predict(X_train)
            train_errors = self._compute_prediction_errors(
                y_train, train_predictions, 'training'
            )
            results['training_errors'] = train_errors

            # Test predictions and errors (if test data provided)
            if X_test is not None and y_test is not None:
                test_predictions = model.predict(X_test)
                test_errors = self._compute_prediction_errors(
                    y_test, test_predictions, 'testing'
                )
                results['test_errors'] = test_errors
                results['test_samples'] = len(X_test)

            # Extract model parameters and coefficients
            model_params = self._extract_model_parameters(model)
            results['model_parameters'] = model_params

            # Feature importance
            feature_importance = self._extract_feature_importance(model, X_train.columns)
            if feature_importance:
                results['feature_importance'] = feature_importance

            # Store fitted model
            self.fitted_models[model_name] = {
                'model': model,
                'training_data_shape': X_train.shape,
                'feature_names': list(X_train.columns) if hasattr(X_train, 'columns') else None,
                'fit_date': fit_start
            }

            # Store analysis results
            self.analysis_results[model_name] = results

            return results

        except Exception as e:
            return {'error': str(e)}

    def _compute_prediction_errors(self, y_true: pd.Series,
                                 y_pred: np.ndarray,
                                 dataset_type: str) -> Dict[str, Any]:
        """Compute comprehensive prediction errors."""
        try:
            from sklearn.metrics import (
                mean_squared_error, mean_absolute_error, r2_score,
                accuracy_score, precision_score, recall_score, f1_score,
                classification_report, confusion_matrix
            )

            errors = {
                'dataset_type': dataset_type,
                'n_samples': len(y_true)
            }

            # Detect problem type
            if pd.api.types.is_numeric_dtype(y_true) and len(y_true.unique()) > 10:
                # Regression metrics
                errors.update({
                    'problem_type': 'regression',
                    'mse': float(mean_squared_error(y_true, y_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                    'mae': float(mean_absolute_error(y_true, y_pred)),
                    'r2_score': float(r2_score(y_true, y_pred)),
                    'residuals': (y_true - y_pred).tolist()
                })

                # Additional regression diagnostics
                residuals = y_true - y_pred
                errors.update({
                    'residual_mean': float(np.mean(residuals)),
                    'residual_std': float(np.std(residuals)),
                    'residual_min': float(np.min(residuals)),
                    'residual_max': float(np.max(residuals))
                })

            else:
                # Classification metrics
                errors.update({
                    'problem_type': 'classification',
                    'accuracy': float(accuracy_score(y_true, y_pred))
                })

                # Multi-class vs binary classification
                unique_classes = len(np.unique(y_true))
                if unique_classes == 2:
                    errors.update({
                        'precision': float(precision_score(y_true, y_pred)),
                        'recall': float(recall_score(y_true, y_pred)),
                        'f1_score': float(f1_score(y_true, y_pred))
                    })
                else:
                    errors.update({
                        'precision_macro': float(precision_score(y_true, y_pred, average='macro')),
                        'recall_macro': float(recall_score(y_true, y_pred, average='macro')),
                        'f1_score_macro': float(f1_score(y_true, y_pred, average='macro')),
                        'precision_weighted': float(precision_score(y_true, y_pred, average='weighted')),
                        'recall_weighted': float(recall_score(y_true, y_pred, average='weighted')),
                        'f1_score_weighted': float(f1_score(y_true, y_pred, average='weighted'))
                    })

                # Confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                errors['confusion_matrix'] = cm.tolist()

                # Classification report
                try:
                    class_report = classification_report(y_true, y_pred, output_dict=True)
                    errors['classification_report'] = class_report
                except:
                    pass

            return errors

        except Exception as e:
            return {'error': str(e)}

    def _extract_model_parameters(self, model) -> Dict[str, Any]:
        """Extract model parameters and coefficients."""
        try:
            params = {
                'model_class': model.__class__.__name__,
                'sklearn_params': model.get_params()
            }

            # Linear model coefficients
            if hasattr(model, 'coef_'):
                if len(model.coef_.shape) == 1:
                    params['coefficients'] = model.coef_.tolist()
                else:
                    params['coefficients'] = model.coef_.tolist()

            if hasattr(model, 'intercept_'):
                if np.isscalar(model.intercept_):
                    params['intercept'] = float(model.intercept_)
                else:
                    params['intercept'] = model.intercept_.tolist()

            # Tree-based model parameters
            if hasattr(model, 'feature_importances_'):
                params['feature_importances'] = model.feature_importances_.tolist()

            if hasattr(model, 'n_estimators'):
                params['n_estimators'] = model.n_estimators

            if hasattr(model, 'max_depth'):
                params['max_depth'] = model.max_depth

            # SVM parameters
            if hasattr(model, 'support_vectors_'):
                params['n_support_vectors'] = len(model.support_vectors_)

            if hasattr(model, 'dual_coef_'):
                params['dual_coefficients'] = model.dual_coef_.tolist()

            return params

        except Exception as e:
            return {'error': str(e)}

    def _extract_feature_importance(self, model, feature_names) -> Optional[Dict[str, Any]]:
        """Extract feature importance with ranking."""
        try:
            importance_info = None

            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                importance_info = {
                    'method': 'feature_importances_',
                    'importances': importances.tolist()
                }

            elif hasattr(model, 'coef_'):
                if len(model.coef_.shape) == 1:
                    importances = np.abs(model.coef_)
                else:
                    importances = np.mean(np.abs(model.coef_), axis=0)

                importance_info = {
                    'method': 'coefficients',
                    'importances': importances.tolist()
                }

            if importance_info and feature_names is not None:
                # Create ranked features
                feature_importance_pairs = list(zip(feature_names, importance_info['importances']))
                feature_importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

                importance_info['ranked_features'] = [
                    {'feature': feature, 'importance': float(importance)}
                    for feature, importance in feature_importance_pairs
                ]

                # Top features
                importance_info['top_10_features'] = importance_info['ranked_features'][:10]

            return importance_info

        except Exception as e:
            return None

    def compute_shap_values(self, model_name: str, X_sample: pd.DataFrame,
                          max_samples: int = 100) -> Dict[str, Any]:
        """
        Compute SHAP values for feature importance analysis.

        Parameters:
        -----------
        model_name : str
            Name of fitted model
        X_sample : pd.DataFrame
            Sample data for SHAP analysis
        max_samples : int
            Maximum samples to analyze

        Returns:
        --------
        Dict with SHAP analysis results
        """
        try:
            try:
                import shap
            except ImportError:
                return {'error': 'SHAP library not installed. Install with: pip install shap'}

            if model_name not in self.fitted_models:
                return {'error': f'Model {model_name} not found'}

            model = self.fitted_models[model_name]['model']

            # Limit sample size for computational efficiency
            if len(X_sample) > max_samples:
                X_sample = X_sample.sample(n=max_samples, random_state=42)

            # Choose appropriate explainer
            shap_results = {
                'model_name': model_name,
                'n_samples': len(X_sample),
                'feature_names': list(X_sample.columns),
                'analysis_date': datetime.now()
            }

            try:
                # Try TreeExplainer for tree-based models
                if hasattr(model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)

                    shap_results['explainer_type'] = 'TreeExplainer'

                    # Handle multi-class output
                    if isinstance(shap_values, list):
                        shap_results['shap_values'] = [sv.tolist() for sv in shap_values]
                        shap_results['is_multiclass'] = True
                        shap_results['n_classes'] = len(shap_values)
                    else:
                        shap_results['shap_values'] = shap_values.tolist()
                        shap_results['is_multiclass'] = False

                else:
                    # Use LinearExplainer for linear models
                    explainer = shap.LinearExplainer(model, X_sample)
                    shap_values = explainer.shap_values(X_sample)

                    shap_results['explainer_type'] = 'LinearExplainer'
                    shap_results['shap_values'] = shap_values.tolist()
                    shap_results['is_multiclass'] = False

                # Compute feature importance from SHAP values
                if not shap_results['is_multiclass']:
                    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                    feature_importance = list(zip(X_sample.columns, mean_abs_shap))
                    feature_importance.sort(key=lambda x: x[1], reverse=True)

                    shap_results['feature_importance'] = [
                        {'feature': feat, 'importance': float(imp)}
                        for feat, imp in feature_importance
                    ]

                return shap_results

            except Exception as shap_error:
                # Fallback to KernelExplainer (slower but more general)
                try:
                    explainer = shap.KernelExplainer(model.predict, X_sample.sample(min(50, len(X_sample))))
                    shap_values = explainer.shap_values(X_sample.sample(min(20, len(X_sample))))

                    shap_results['explainer_type'] = 'KernelExplainer'
                    shap_results['shap_values'] = shap_values.tolist()
                    shap_results['note'] = 'Used slower KernelExplainer due to model compatibility'

                    return shap_results

                except Exception as kernel_error:
                    return {
                        'error': f'SHAP analysis failed: {shap_error}',
                        'fallback_error': str(kernel_error)
                    }

        except Exception as e:
            return {'error': str(e)}

    def hyperparameter_tuning(self, model_class, X_train: pd.DataFrame,
                            y_train: pd.Series, X_test: pd.DataFrame = None,
                            y_test: pd.Series = None,
                            search_type: str = 'grid',
                            cv_folds: int = 5,
                            n_jobs: int = -1,
                            scoring: str = 'auto') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for tree-based models.

        Parameters:
        -----------
        model_class : sklearn model class
            Model class to tune (e.g., RandomForestClassifier)
        X_train, y_train : training data
        X_test, y_test : test data (optional)
        search_type : str
            'grid', 'random', or 'bayesian'
        cv_folds : int
            Number of cross-validation folds
        n_jobs : int
            Number of parallel jobs
        scoring : str
            Scoring metric ('auto', 'accuracy', 'f1', 'roc_auc', 'r2', etc.)

        Returns:
        --------
        Dict with tuning results and best model
        """
        try:
            from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
            from sklearn.metrics import make_scorer

            # Auto-detect problem type and scoring
            if scoring == 'auto':
                if pd.api.types.is_numeric_dtype(y_train) and len(y_train.unique()) > 10:
                    scoring = 'r2'
                else:
                    scoring = 'accuracy'

            # Define parameter grids based on model type
            param_grids = self._get_hyperparameter_grids(model_class)

            if not param_grids:
                return {'error': f'No parameter grid defined for {model_class.__name__}'}

            results = {
                'model_class': model_class.__name__,
                'search_type': search_type,
                'cv_folds': cv_folds,
                'scoring': scoring,
                'tuning_date': datetime.now()
            }

            # Initialize base model
            base_model = model_class(random_state=42)

            # Choose search strategy
            if search_type == 'grid':
                search = GridSearchCV(
                    base_model,
                    param_grids,
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    verbose=1
                )
            elif search_type == 'random':
                search = RandomizedSearchCV(
                    base_model,
                    param_grids,
                    n_iter=50,  # Number of parameter settings sampled
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    random_state=42,
                    verbose=1
                )
            else:
                # Bayesian optimization (requires scikit-optimize)
                try:
                    from skopt import BayesSearchCV
                    search = BayesSearchCV(
                        base_model,
                        param_grids,
                        n_iter=50,
                        cv=cv_folds,
                        scoring=scoring,
                        n_jobs=n_jobs,
                        random_state=42
                    )
                except ImportError:
                    return {'error': 'Bayesian search requires scikit-optimize: pip install scikit-optimize'}

            # Perform hyperparameter search
            tuning_start = datetime.now()
            search.fit(X_train, y_train)
            tuning_time = datetime.now() - tuning_start

            # Extract results
            results.update({
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'best_estimator': search.best_estimator_,
                'tuning_time': tuning_time.total_seconds(),
                'n_parameter_combinations': len(search.cv_results_['params'])
            })

            # Cross-validation results
            cv_results = {
                'mean_test_scores': search.cv_results_['mean_test_score'].tolist(),
                'std_test_scores': search.cv_results_['std_test_score'].tolist(),
                'params': search.cv_results_['params']
            }
            results['cv_results'] = cv_results

            # Test set evaluation if provided
            if X_test is not None and y_test is not None:
                test_predictions = search.best_estimator_.predict(X_test)
                test_errors = self._compute_prediction_errors(
                    y_test, test_predictions, 'testing'
                )
                results['test_performance'] = test_errors

            # Store best model
            model_name = f'{model_class.__name__}_tuned'
            self.fitted_models[model_name] = {
                'model': search.best_estimator_,
                'training_data_shape': X_train.shape,
                'feature_names': list(X_train.columns) if hasattr(X_train, 'columns') else None,
                'fit_date': tuning_start,
                'hyperparameter_search': search
            }

            results['tuned_model_name'] = model_name

            return results

        except Exception as e:
            return {'error': str(e)}

    def _get_hyperparameter_grids(self, model_class) -> Dict[str, Any]:
        """Get hyperparameter grids for different model types."""
        model_name = model_class.__name__

        if 'RandomForest' in model_name:
            return {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'colsample_bytree': [0.8, 0.9, 1.0]  # Note: sklearn uses max_features
            }

        elif 'GradientBoosting' in model_name:
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

        elif 'XGB' in model_name or 'LGB' in model_name:
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'subsample': [0.8, 0.9, 1.0]
            }

        elif 'DecisionTree' in model_name:
            return {
                'max_depth': [3, 5, 10, 15, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10],
                'criterion': ['gini', 'entropy'] if 'Classifier' in model_name else ['mse', 'mae']
            }

        elif 'SVC' in model_name or 'SVR' in model_name:
            return {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            }

        else:
            return {}

    def extract_best_hyperparameters(self, model_name: str) -> Dict[str, Any]:
        """Extract best hyperparameters from tuned model."""
        try:
            if model_name not in self.fitted_models:
                return {'error': f'Model {model_name} not found'}

            model_info = self.fitted_models[model_name]

            if 'hyperparameter_search' not in model_info:
                return {'error': f'No hyperparameter search results for {model_name}'}

            search = model_info['hyperparameter_search']

            best_params_info = {
                'model_name': model_name,
                'best_parameters': search.best_params_,
                'best_cross_val_score': search.best_score_,
                'scoring_metric': search.scoring,
                'cv_folds': search.cv,
                'extraction_date': datetime.now()
            }

            # Add parameter importance analysis
            cv_results = search.cv_results_
            param_importance = self._analyze_parameter_importance(cv_results)
            best_params_info['parameter_importance'] = param_importance

            return best_params_info

        except Exception as e:
            return {'error': str(e)}

    def _analyze_parameter_importance(self, cv_results: Dict) -> Dict[str, Any]:
        """Analyze which parameters have the most impact on performance."""
        try:
            params_list = cv_results['params']
            scores = cv_results['mean_test_score']

            # Get all parameter names
            all_param_names = set()
            for param_dict in params_list:
                all_param_names.update(param_dict.keys())

            param_importance = {}

            for param_name in all_param_names:
                # Group scores by parameter value
                param_scores = {}
                for i, param_dict in enumerate(params_list):
                    if param_name in param_dict:
                        param_value = param_dict[param_name]
                        if param_value not in param_scores:
                            param_scores[param_value] = []
                        param_scores[param_value].append(scores[i])

                # Calculate mean score for each parameter value
                param_means = {value: np.mean(score_list)
                             for value, score_list in param_scores.items()}

                # Calculate importance as range of mean scores
                if len(param_means) > 1:
                    importance = max(param_means.values()) - min(param_means.values())
                    best_value = max(param_means, key=param_means.get)

                    param_importance[param_name] = {
                        'importance_score': float(importance),
                        'best_value': best_value,
                        'value_scores': {str(k): float(v) for k, v in param_means.items()}
                    }

            # Sort by importance
            sorted_importance = sorted(
                param_importance.items(),
                key=lambda x: x[1]['importance_score'],
                reverse=True
            )

            return {
                'ranked_parameters': [
                    {'parameter': param, **info}
                    for param, info in sorted_importance
                ],
                'most_important_parameter': sorted_importance[0][0] if sorted_importance else None
            }

        except Exception as e:
            return {'error': str(e)}


class ModelVisualizer:
    """
    Model visualization including decision trees and dendrograms.
    """

    def __init__(self, config: FeatureEngineeringConfig = None):
        self.config = config or FeatureEngineeringConfig()

    def plot_decision_tree(self, model, feature_names: List[str] = None,
                          class_names: List[str] = None,
                          max_depth: int = 3,
                          save_path: str = None) -> Dict[str, Any]:
        """
        Create decision tree visualization.

        Parameters:
        -----------
        model : sklearn decision tree model
            Fitted decision tree model
        feature_names : List[str]
            Names of features
        class_names : List[str]
            Names of classes (for classification)
        max_depth : int
            Maximum depth to visualize
        save_path : str
            Path to save the plot

        Returns:
        --------
        Dict with plot information
        """
        try:
            from sklearn.tree import plot_tree, export_text
            import matplotlib.pyplot as plt

            # Check if model is a tree-based model
            if not hasattr(model, 'tree_'):
                return {'error': 'Model is not a decision tree'}

            plot_info = {
                'model_class': model.__class__.__name__,
                'max_depth_visualized': max_depth,
                'visualization_date': datetime.now()
            }

            # Create the plot
            plt.figure(figsize=(20, 10))
            plot_tree(
                model,
                feature_names=feature_names,
                class_names=class_names,
                filled=True,
                rounded=True,
                fontsize=10,
                max_depth=max_depth
            )
            plt.title(f'Decision Tree Visualization (max_depth={max_depth})')

            # Save plot if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plot_info['saved_plot_path'] = save_path
            else:
                # Save to temporary file
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                plt.savefig(temp_file.name, dpi=300, bbox_inches='tight')
                plot_info['temp_plot_path'] = temp_file.name
                temp_file.close()

            plt.close()

            # Generate text representation
            tree_text = export_text(
                model,
                feature_names=feature_names,
                max_depth=max_depth
            )
            plot_info['tree_text'] = tree_text

            # Tree statistics
            plot_info['tree_statistics'] = {
                'n_nodes': model.tree_.node_count,
                'n_leaves': model.tree_.n_leaves,
                'max_depth': model.tree_.max_depth,
                'n_features': model.tree_.n_features,
                'n_classes': model.tree_.n_classes[0] if hasattr(model.tree_, 'n_classes') else None
            }

            return plot_info

        except Exception as e:
            return {'error': str(e)}

    def plot_feature_importance(self, model, feature_names: List[str],
                              top_n: int = 20,
                              save_path: str = None) -> Dict[str, Any]:
        """
        Create feature importance visualization.

        Parameters:
        -----------
        model : sklearn model
            Fitted model with feature_importances_ or coef_
        feature_names : List[str]
            Names of features
        top_n : int
            Number of top features to show
        save_path : str
            Path to save the plot

        Returns:
        --------
        Dict with plot information
        """
        try:
            import matplotlib.pyplot as plt

            plot_info = {
                'model_class': model.__class__.__name__,
                'top_n_features': top_n,
                'visualization_date': datetime.now()
            }

            # Extract feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                importance_type = 'Feature Importances'
            elif hasattr(model, 'coef_'):
                if len(model.coef_.shape) == 1:
                    importances = np.abs(model.coef_)
                else:
                    importances = np.mean(np.abs(model.coef_), axis=0)
                importance_type = 'Coefficient Magnitudes'
            else:
                return {'error': 'Model does not have feature importance or coefficients'}

            # Create feature importance pairs and sort
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(len(importances))]

            feature_importance_pairs = list(zip(feature_names, importances))
            feature_importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

            # Take top N features
            top_features = feature_importance_pairs[:top_n]
            top_feature_names = [pair[0] for pair in top_features]
            top_importances = [pair[1] for pair in top_features]

            # Create horizontal bar plot
            plt.figure(figsize=(10, max(6, len(top_features) * 0.4)))
            y_pos = np.arange(len(top_feature_names))

            plt.barh(y_pos, top_importances)
            plt.yticks(y_pos, top_feature_names)
            plt.xlabel(importance_type)
            plt.title(f'Top {len(top_features)} {importance_type}')
            plt.gca().invert_yaxis()

            # Add value labels on bars
            for i, v in enumerate(top_importances):
                plt.text(v + max(top_importances) * 0.01, i, f'{v:.4f}',
                        va='center', fontsize=8)

            plt.tight_layout()

            # Save plot
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plot_info['saved_plot_path'] = save_path
            else:
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                plt.savefig(temp_file.name, dpi=300, bbox_inches='tight')
                plot_info['temp_plot_path'] = temp_file.name
                temp_file.close()

            plt.close()

            plot_info['feature_importance_data'] = [
                {'feature': name, 'importance': float(imp)}
                for name, imp in top_features
            ]

            return plot_info

        except Exception as e:
            return {'error': str(e)}

    def plot_confusion_matrix(self, y_true, y_pred, class_names: List[str] = None,
                            normalize: bool = False,
                            save_path: str = None) -> Dict[str, Any]:
        """
        Create confusion matrix visualization.

        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        class_names : List[str]
            Names of classes
        normalize : bool
            Whether to normalize the confusion matrix
        save_path : str
            Path to save the plot

        Returns:
        --------
        Dict with plot information
        """
        try:
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            plot_info = {
                'confusion_matrix': cm.tolist(),
                'normalize': normalize,
                'visualization_date': datetime.now()
            }

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                plot_info['confusion_matrix_normalized'] = cm.tolist()

            # Create plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt='.2f' if normalize else 'd',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names
            )
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
            plt.title(title)

            # Save plot
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plot_info['saved_plot_path'] = save_path
            else:
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                plt.savefig(temp_file.name, dpi=300, bbox_inches='tight')
                plot_info['temp_plot_path'] = temp_file.name
                temp_file.close()

            plt.close()

            return plot_info

        except Exception as e:
            return {'error': str(e)}


class ModelPersistence:
    """
    Model saving and loading functionality.
    """

    def __init__(self, config: FeatureEngineeringConfig = None):
        self.config = config or FeatureEngineeringConfig()

    def save_model(self, model, file_path: str,
                  metadata: Dict[str, Any] = None,
                  include_preprocessing: bool = True) -> Dict[str, Any]:
        """
        Save model to file with metadata.

        Parameters:
        -----------
        model : sklearn model or pipeline
            Model to save
        file_path : str
            Path to save the model
        metadata : Dict
            Additional metadata to save
        include_preprocessing : bool
            Whether to include preprocessing steps

        Returns:
        --------
        Dict with save information
        """
        try:
            import joblib
            import json
            import os

            save_info = {
                'model_class': model.__class__.__name__,
                'file_path': file_path,
                'save_date': datetime.now()
            }

            # Prepare model data
            model_data = {
                'model': model,
                'model_class': model.__class__.__name__,
                'save_date': datetime.now().isoformat(),
                'sklearn_version': self._get_sklearn_version()
            }

            # Add metadata if provided
            if metadata:
                model_data['metadata'] = metadata

            # Extract model information
            if hasattr(model, 'get_params'):
                model_data['model_params'] = model.get_params()

            # Save using joblib
            joblib.dump(model_data, file_path)

            # Create accompanying metadata file
            metadata_path = file_path.replace('.pkl', '_metadata.json')
            metadata_info = {
                'model_class': model_data['model_class'],
                'save_date': model_data['save_date'],
                'sklearn_version': model_data['sklearn_version'],
                'file_size_bytes': os.path.getsize(file_path),
                'model_params': model_data.get('model_params', {}),
                'custom_metadata': metadata or {}
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata_info, f, indent=2, default=str)

            save_info.update({
                'metadata_path': metadata_path,
                'file_size_bytes': os.path.getsize(file_path),
                'success': True
            })

            return save_info

        except Exception as e:
            return {'error': str(e), 'file_path': file_path}

    def load_model(self, file_path: str) -> Dict[str, Any]:
        """
        Load model from file.

        Parameters:
        -----------
        file_path : str
            Path to the saved model

        Returns:
        --------
        Dict with loaded model and information
        """
        try:
            import joblib
            import json
            import os

            if not os.path.exists(file_path):
                return {'error': f'File not found: {file_path}'}

            # Load model data
            model_data = joblib.load(file_path)

            load_info = {
                'file_path': file_path,
                'load_date': datetime.now(),
                'model': model_data['model'],
                'model_class': model_data['model_class'],
                'save_date': model_data['save_date'],
                'sklearn_version': model_data.get('sklearn_version'),
                'current_sklearn_version': self._get_sklearn_version()
            }

            # Load metadata if available
            metadata_path = file_path.replace('.pkl', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                load_info['metadata'] = metadata

            # Check sklearn version compatibility
            saved_version = model_data.get('sklearn_version')
            current_version = self._get_sklearn_version()
            if saved_version != current_version:
                load_info['version_warning'] = (
                    f'Model saved with sklearn {saved_version}, '
                    f'current version is {current_version}'
                )

            load_info['success'] = True

            return load_info

        except Exception as e:
            return {'error': str(e), 'file_path': file_path}

    def save_model_specifications(self, model, file_path: str) -> Dict[str, Any]:
        """
        Save detailed model specifications without the actual model.

        Parameters:
        -----------
        model : sklearn model
            Model to extract specifications from
        file_path : str
            Path to save specifications

        Returns:
        --------
        Dict with save information
        """
        try:
            import json

            specs = {
                'model_class': model.__class__.__name__,
                'model_module': model.__class__.__module__,
                'model_params': model.get_params() if hasattr(model, 'get_params') else {},
                'extraction_date': datetime.now().isoformat(),
                'sklearn_version': self._get_sklearn_version()
            }

            # Add model-specific attributes
            if hasattr(model, 'feature_importances_'):
                specs['feature_importances'] = model.feature_importances_.tolist()

            if hasattr(model, 'coef_'):
                specs['coefficients'] = model.coef_.tolist()

            if hasattr(model, 'intercept_'):
                if np.isscalar(model.intercept_):
                    specs['intercept'] = float(model.intercept_)
                else:
                    specs['intercept'] = model.intercept_.tolist()

            if hasattr(model, 'classes_'):
                specs['classes'] = model.classes_.tolist()

            if hasattr(model, 'n_features_in_'):
                specs['n_features_in'] = int(model.n_features_in_)

            # Save specifications
            with open(file_path, 'w') as f:
                json.dump(specs, f, indent=2, default=str)

            return {
                'file_path': file_path,
                'model_class': specs['model_class'],
                'save_date': datetime.now(),
                'success': True
            }

        except Exception as e:
            return {'error': str(e), 'file_path': file_path}

    def _get_sklearn_version(self) -> str:
        """Get current sklearn version."""
        try:
            import sklearn
            return sklearn.__version__
        except:
            return 'unknown'
