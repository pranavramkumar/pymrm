"""
ValidationMetrics: Comprehensive Validation and Evaluation Metrics Module

This module provides extensive validation metrics for machine learning models including:
- Regression metrics (MAE, MSE, RMSE, R², MAPE, etc.)
- Classification metrics (accuracy, precision, recall, F1, etc.)
- Binary classification specific metrics (odds ratio, risk ratio, etc.)
- Clustering metrics (inertia, silhouette score, etc.)
- Advanced metrics and statistical tests

Author: Claude AI
Created: 2025
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import scipy.stats as stats
from collections import Counter


@dataclass
class ValidationConfig:
    """Configuration class for validation metrics."""

    # General settings
    round_decimals: int = 4
    confidence_level: float = 0.95
    handle_nan: str = 'warn'  # 'warn', 'raise', 'ignore'

    # Classification settings
    average_method: str = 'weighted'  # 'macro', 'micro', 'weighted', 'binary'
    pos_label: Union[int, str] = 1
    zero_division: Union[str, int] = 'warn'

    # Clustering settings
    clustering_metric: str = 'euclidean'

    # Odds/Risk ratio settings
    continuity_correction: float = 0.5


class RegressionMetrics:
    """
    Comprehensive regression validation metrics.
    """

    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()

    def mean_absolute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error (MAE)."""
        try:
            y_true, y_pred = self._validate_inputs(y_true, y_pred)
            mae = np.mean(np.abs(y_true - y_pred))
            return round(mae, self.config.round_decimals)
        except Exception as e:
            return {'error': str(e)}

    def mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Squared Error (MSE)."""
        try:
            y_true, y_pred = self._validate_inputs(y_true, y_pred)
            mse = np.mean((y_true - y_pred) ** 2)
            return round(mse, self.config.round_decimals)
        except Exception as e:
            return {'error': str(e)}

    def root_mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error (RMSE)."""
        try:
            mse = self.mean_squared_error(y_true, y_pred)
            if isinstance(mse, dict):
                return mse
            rmse = np.sqrt(mse)
            return round(rmse, self.config.round_decimals)
        except Exception as e:
            return {'error': str(e)}

    def mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error (MAPE)."""
        try:
            y_true, y_pred = self._validate_inputs(y_true, y_pred)

            # Avoid division by zero
            mask = y_true != 0
            if not np.any(mask):
                return {'error': 'All true values are zero, cannot calculate MAPE'}

            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            return round(mape, self.config.round_decimals)
        except Exception as e:
            return {'error': str(e)}

    def symmetric_mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error (SMAPE)."""
        try:
            y_true, y_pred = self._validate_inputs(y_true, y_pred)

            denominator = np.abs(y_true) + np.abs(y_pred)
            mask = denominator != 0

            if not np.any(mask):
                return {'error': 'All denominators are zero, cannot calculate SMAPE'}

            smape = np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
            return round(smape, self.config.round_decimals)
        except Exception as e:
            return {'error': str(e)}

    def r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared (coefficient of determination)."""
        try:
            y_true, y_pred = self._validate_inputs(y_true, y_pred)

            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

            if ss_tot == 0:
                return {'error': 'Total sum of squares is zero, cannot calculate R²'}

            r2 = 1 - (ss_res / ss_tot)
            return round(r2, self.config.round_decimals)
        except Exception as e:
            return {'error': str(e)}

    def adjusted_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
        """Calculate Adjusted R-squared."""
        try:
            r2 = self.r_squared(y_true, y_pred)
            if isinstance(r2, dict):
                return r2

            n = len(y_true)
            if n <= n_features + 1:
                return {'error': 'Not enough samples for the number of features'}

            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
            return round(adj_r2, self.config.round_decimals)
        except Exception as e:
            return {'error': str(e)}

    def mean_squared_log_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Squared Logarithmic Error (MSLE)."""
        try:
            y_true, y_pred = self._validate_inputs(y_true, y_pred)

            if np.any(y_true < 0) or np.any(y_pred < 0):
                return {'error': 'MSLE requires non-negative values'}

            msle = np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)
            return round(msle, self.config.round_decimals)
        except Exception as e:
            return {'error': str(e)}

    def median_absolute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Median Absolute Error (MedAE)."""
        try:
            y_true, y_pred = self._validate_inputs(y_true, y_pred)
            medae = np.median(np.abs(y_true - y_pred))
            return round(medae, self.config.round_decimals)
        except Exception as e:
            return {'error': str(e)}

    def explained_variance_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Explained Variance Score."""
        try:
            y_true, y_pred = self._validate_inputs(y_true, y_pred)

            y_true_var = np.var(y_true)
            if y_true_var == 0:
                return {'error': 'True values have zero variance'}

            residual_var = np.var(y_true - y_pred)
            evs = 1 - (residual_var / y_true_var)
            return round(evs, self.config.round_decimals)
        except Exception as e:
            return {'error': str(e)}

    def max_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Maximum Error."""
        try:
            y_true, y_pred = self._validate_inputs(y_true, y_pred)
            max_err = np.max(np.abs(y_true - y_pred))
            return round(max_err, self.config.round_decimals)
        except Exception as e:
            return {'error': str(e)}

    def _validate_inputs(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and clean input arrays."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

        # Handle NaN values
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            if self.config.handle_nan == 'raise':
                raise ValueError("NaN values found in input")
            elif self.config.handle_nan == 'warn':
                warnings.warn("NaN values found, removing them")

            # Remove NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true = y_true[mask]
            y_pred = y_pred[mask]

            if len(y_true) == 0:
                raise ValueError("No valid data points after removing NaN values")

        return y_true, y_pred

    def comprehensive_regression_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      n_features: int = None) -> Dict[str, Any]:
        """Generate comprehensive regression metrics report."""
        try:
            report = {
                'analysis_date': datetime.now(),
                'n_samples': len(y_true),
                'metrics': {}
            }

            # Basic metrics
            report['metrics']['mae'] = self.mean_absolute_error(y_true, y_pred)
            report['metrics']['mse'] = self.mean_squared_error(y_true, y_pred)
            report['metrics']['rmse'] = self.root_mean_squared_error(y_true, y_pred)
            report['metrics']['r2'] = self.r_squared(y_true, y_pred)

            # Advanced metrics
            report['metrics']['mape'] = self.mean_absolute_percentage_error(y_true, y_pred)
            report['metrics']['smape'] = self.symmetric_mean_absolute_percentage_error(y_true, y_pred)
            report['metrics']['medae'] = self.median_absolute_error(y_true, y_pred)
            report['metrics']['evs'] = self.explained_variance_score(y_true, y_pred)
            report['metrics']['max_error'] = self.max_error(y_true, y_pred)

            # Conditional metrics
            if np.all(y_true >= 0) and np.all(y_pred >= 0):
                report['metrics']['msle'] = self.mean_squared_log_error(y_true, y_pred)

            if n_features is not None:
                report['metrics']['adj_r2'] = self.adjusted_r_squared(y_true, y_pred, n_features)

            # Residual analysis
            residuals = y_true - y_pred
            report['residual_analysis'] = {
                'mean': round(np.mean(residuals), self.config.round_decimals),
                'std': round(np.std(residuals), self.config.round_decimals),
                'min': round(np.min(residuals), self.config.round_decimals),
                'max': round(np.max(residuals), self.config.round_decimals),
                'q25': round(np.percentile(residuals, 25), self.config.round_decimals),
                'q50': round(np.percentile(residuals, 50), self.config.round_decimals),
                'q75': round(np.percentile(residuals, 75), self.config.round_decimals)
            }

            # Normality test for residuals
            try:
                shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])  # Limit for large datasets
                report['residual_analysis']['normality_test'] = {
                    'shapiro_wilk_statistic': round(shapiro_stat, self.config.round_decimals),
                    'shapiro_wilk_p_value': round(shapiro_p, self.config.round_decimals),
                    'is_normal': shapiro_p > (1 - self.config.confidence_level)
                }
            except Exception:
                pass

            return report

        except Exception as e:
            return {'error': str(e)}


class ClassificationMetrics:
    """
    Comprehensive classification validation metrics.
    """

    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()

    def accuracy_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy score."""
        try:
            y_true, y_pred = self._validate_inputs(y_true, y_pred)
            accuracy = np.mean(y_true == y_pred)
            return round(accuracy, self.config.round_decimals)
        except Exception as e:
            return {'error': str(e)}

    def precision_score(self, y_true: np.ndarray, y_pred: np.ndarray,
                       average: str = None, labels: List = None) -> Union[float, np.ndarray]:
        """Calculate precision score."""
        try:
            from sklearn.metrics import precision_score as sk_precision

            average = average or self.config.average_method
            precision = sk_precision(
                y_true, y_pred,
                average=average,
                labels=labels,
                pos_label=self.config.pos_label,
                zero_division=self.config.zero_division
            )

            if isinstance(precision, np.ndarray):
                return np.round(precision, self.config.round_decimals)
            return round(precision, self.config.round_decimals)

        except Exception as e:
            return {'error': str(e)}

    def recall_score(self, y_true: np.ndarray, y_pred: np.ndarray,
                    average: str = None, labels: List = None) -> Union[float, np.ndarray]:
        """Calculate recall score."""
        try:
            from sklearn.metrics import recall_score as sk_recall

            average = average or self.config.average_method
            recall = sk_recall(
                y_true, y_pred,
                average=average,
                labels=labels,
                pos_label=self.config.pos_label,
                zero_division=self.config.zero_division
            )

            if isinstance(recall, np.ndarray):
                return np.round(recall, self.config.round_decimals)
            return round(recall, self.config.round_decimals)

        except Exception as e:
            return {'error': str(e)}

    def f1_score(self, y_true: np.ndarray, y_pred: np.ndarray,
                average: str = None, labels: List = None) -> Union[float, np.ndarray]:
        """Calculate F1 score."""
        try:
            from sklearn.metrics import f1_score as sk_f1

            average = average or self.config.average_method
            f1 = sk_f1(
                y_true, y_pred,
                average=average,
                labels=labels,
                pos_label=self.config.pos_label,
                zero_division=self.config.zero_division
            )

            if isinstance(f1, np.ndarray):
                return np.round(f1, self.config.round_decimals)
            return round(f1, self.config.round_decimals)

        except Exception as e:
            return {'error': str(e)}

    def fbeta_score(self, y_true: np.ndarray, y_pred: np.ndarray, beta: float = 1.0,
                   average: str = None, labels: List = None) -> Union[float, np.ndarray]:
        """Calculate F-beta score."""
        try:
            from sklearn.metrics import fbeta_score as sk_fbeta

            average = average or self.config.average_method
            fbeta = sk_fbeta(
                y_true, y_pred, beta=beta,
                average=average,
                labels=labels,
                pos_label=self.config.pos_label,
                zero_division=self.config.zero_division
            )

            if isinstance(fbeta, np.ndarray):
                return np.round(fbeta, self.config.round_decimals)
            return round(fbeta, self.config.round_decimals)

        except Exception as e:
            return {'error': str(e)}

    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                        labels: List = None) -> Dict[str, Any]:
        """Calculate confusion matrix with detailed analysis."""
        try:
            from sklearn.metrics import confusion_matrix as sk_confusion_matrix

            cm = sk_confusion_matrix(y_true, y_pred, labels=labels)

            # Get unique labels
            if labels is None:
                labels = sorted(list(set(y_true) | set(y_pred)))

            result = {
                'confusion_matrix': cm.tolist(),
                'labels': labels,
                'shape': cm.shape
            }

            # Add normalized confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            result['confusion_matrix_normalized'] = np.round(cm_normalized, self.config.round_decimals).tolist()

            # Per-class metrics
            if cm.shape[0] == cm.shape[1]:  # Square matrix
                n_classes = cm.shape[0]
                per_class_metrics = {}

                for i, label in enumerate(labels):
                    tp = cm[i, i]
                    fp = cm[:, i].sum() - tp
                    fn = cm[i, :].sum() - tp
                    tn = cm.sum() - tp - fp - fn

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                    per_class_metrics[str(label)] = {
                        'true_positives': int(tp),
                        'false_positives': int(fp),
                        'false_negatives': int(fn),
                        'true_negatives': int(tn),
                        'precision': round(precision, self.config.round_decimals),
                        'recall': round(recall, self.config.round_decimals),
                        'f1_score': round(f1, self.config.round_decimals)
                    }

                result['per_class_metrics'] = per_class_metrics

            return result

        except Exception as e:
            return {'error': str(e)}

    def classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                            labels: List = None, target_names: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive classification report."""
        try:
            from sklearn.metrics import classification_report as sk_classification_report

            # Get detailed classification report
            report_dict = sk_classification_report(
                y_true, y_pred,
                labels=labels,
                target_names=target_names,
                output_dict=True,
                zero_division=self.config.zero_division
            )

            # Round numerical values
            for key, value in report_dict.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            report_dict[key][sub_key] = round(sub_value, self.config.round_decimals)
                elif isinstance(value, (int, float)):
                    report_dict[key] = round(value, self.config.round_decimals)

            # Add confusion matrix
            cm_result = self.confusion_matrix(y_true, y_pred, labels)
            if 'error' not in cm_result:
                report_dict['confusion_matrix'] = cm_result['confusion_matrix']
                report_dict['confusion_matrix_normalized'] = cm_result['confusion_matrix_normalized']

            # Add overall accuracy
            report_dict['overall_accuracy'] = self.accuracy_score(y_true, y_pred)

            return report_dict

        except Exception as e:
            return {'error': str(e)}

    def roc_auc_score(self, y_true: np.ndarray, y_score: np.ndarray,
                     average: str = 'macro', multi_class: str = 'raise') -> float:
        """Calculate ROC AUC score."""
        try:
            from sklearn.metrics import roc_auc_score as sk_roc_auc

            auc = sk_roc_auc(y_true, y_score, average=average, multi_class=multi_class)
            return round(auc, self.config.round_decimals)

        except Exception as e:
            return {'error': str(e)}

    def log_loss(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate logarithmic loss."""
        try:
            from sklearn.metrics import log_loss as sk_log_loss

            loss = sk_log_loss(y_true, y_prob)
            return round(loss, self.config.round_decimals)

        except Exception as e:
            return {'error': str(e)}

    def _validate_inputs(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and clean input arrays."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

        return y_true, y_pred


class BinaryClassificationMetrics:
    """
    Specialized metrics for binary classification including odds ratios and risk ratios.
    """

    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()

    def odds_ratio(self, y_true: np.ndarray, y_pred: np.ndarray,
                  continuity_correction: bool = True) -> Dict[str, Any]:
        """
        Calculate odds ratio for binary classification.

        Parameters:
        -----------
        y_true : array-like
            True binary labels (0, 1)
        y_pred : array-like
            Predicted binary labels (0, 1)
        continuity_correction : bool
            Whether to apply continuity correction for zero cells

        Returns:
        --------
        Dict containing odds ratio, confidence interval, and statistics
        """
        try:
            y_true, y_pred = self._validate_binary_inputs(y_true, y_pred)

            # Create 2x2 contingency table
            cm = self._create_contingency_table(y_true, y_pred)

            # Extract values from confusion matrix
            # Format: [[TN, FP], [FN, TP]]
            tn, fp = cm[0, 0], cm[0, 1]
            fn, tp = cm[1, 0], cm[1, 1]

            # Apply continuity correction if needed
            if continuity_correction and (tn == 0 or fp == 0 or fn == 0 or tp == 0):
                correction = self.config.continuity_correction
                tn += correction
                fp += correction
                fn += correction
                tp += correction

            # Calculate odds ratio
            if (fp == 0 and fn == 0) or (tn == 0 and tp == 0):
                return {'error': 'Cannot calculate odds ratio: zero denominator'}

            odds_ratio = (tp * tn) / (fp * fn)

            # Calculate confidence interval using log transformation
            log_or = np.log(odds_ratio)
            se_log_or = np.sqrt(1/tn + 1/fp + 1/fn + 1/tp)

            alpha = 1 - self.config.confidence_level
            z_score = stats.norm.ppf(1 - alpha/2)

            ci_lower = np.exp(log_or - z_score * se_log_or)
            ci_upper = np.exp(log_or + z_score * se_log_or)

            result = {
                'odds_ratio': round(odds_ratio, self.config.round_decimals),
                'log_odds_ratio': round(log_or, self.config.round_decimals),
                'confidence_interval': {
                    'lower': round(ci_lower, self.config.round_decimals),
                    'upper': round(ci_upper, self.config.round_decimals),
                    'confidence_level': self.config.confidence_level
                },
                'standard_error_log_or': round(se_log_or, self.config.round_decimals),
                'contingency_table': {
                    'true_negative': int(tn),
                    'false_positive': int(fp),
                    'false_negative': int(fn),
                    'true_positive': int(tp)
                },
                'continuity_correction_applied': continuity_correction
            }

            return result

        except Exception as e:
            return {'error': str(e)}

    def risk_ratio(self, y_true: np.ndarray, y_pred: np.ndarray,
                  continuity_correction: bool = True) -> Dict[str, Any]:
        """
        Calculate risk ratio (relative risk) for binary classification.

        Parameters:
        -----------
        y_true : array-like
            True binary labels (0, 1)
        y_pred : array-like
            Predicted binary labels (0, 1)
        continuity_correction : bool
            Whether to apply continuity correction

        Returns:
        --------
        Dict containing risk ratio, confidence interval, and statistics
        """
        try:
            y_true, y_pred = self._validate_binary_inputs(y_true, y_pred)

            # Create 2x2 contingency table
            cm = self._create_contingency_table(y_true, y_pred)

            # Extract values
            tn, fp = cm[0, 0], cm[0, 1]
            fn, tp = cm[1, 0], cm[1, 1]

            # Apply continuity correction if needed
            if continuity_correction and (tn == 0 or fp == 0 or fn == 0 or tp == 0):
                correction = self.config.continuity_correction
                tn += correction
                fp += correction
                fn += correction
                tp += correction

            # Calculate risks
            risk_exposed = tp / (tp + fn)  # Risk in exposed group (pred=1)
            risk_unexposed = fp / (fp + tn)  # Risk in unexposed group (pred=0)

            if risk_unexposed == 0:
                return {'error': 'Cannot calculate risk ratio: zero risk in unexposed group'}

            risk_ratio = risk_exposed / risk_unexposed

            # Calculate confidence interval using log transformation
            log_rr = np.log(risk_ratio)
            se_log_rr = np.sqrt(1/tp - 1/(tp + fn) + 1/fp - 1/(fp + tn))

            alpha = 1 - self.config.confidence_level
            z_score = stats.norm.ppf(1 - alpha/2)

            ci_lower = np.exp(log_rr - z_score * se_log_rr)
            ci_upper = np.exp(log_rr + z_score * se_log_rr)

            result = {
                'risk_ratio': round(risk_ratio, self.config.round_decimals),
                'log_risk_ratio': round(log_rr, self.config.round_decimals),
                'risk_exposed': round(risk_exposed, self.config.round_decimals),
                'risk_unexposed': round(risk_unexposed, self.config.round_decimals),
                'risk_difference': round(risk_exposed - risk_unexposed, self.config.round_decimals),
                'confidence_interval': {
                    'lower': round(ci_lower, self.config.round_decimals),
                    'upper': round(ci_upper, self.config.round_decimals),
                    'confidence_level': self.config.confidence_level
                },
                'standard_error_log_rr': round(se_log_rr, self.config.round_decimals),
                'contingency_table': {
                    'true_negative': int(tn),
                    'false_positive': int(fp),
                    'false_negative': int(fn),
                    'true_positive': int(tp)
                },
                'continuity_correction_applied': continuity_correction
            }

            return result

        except Exception as e:
            return {'error': str(e)}

    def sensitivity_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate sensitivity and specificity."""
        try:
            y_true, y_pred = self._validate_binary_inputs(y_true, y_pred)

            cm = self._create_contingency_table(y_true, y_pred)
            tn, fp = cm[0, 0], cm[0, 1]
            fn, tp = cm[1, 0], cm[1, 1]

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate

            return {
                'sensitivity': round(sensitivity, self.config.round_decimals),
                'specificity': round(specificity, self.config.round_decimals),
                'true_positive_rate': round(sensitivity, self.config.round_decimals),
                'true_negative_rate': round(specificity, self.config.round_decimals),
                'false_positive_rate': round(1 - specificity, self.config.round_decimals),
                'false_negative_rate': round(1 - sensitivity, self.config.round_decimals)
            }

        except Exception as e:
            return {'error': str(e)}

    def positive_predictive_value(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate positive and negative predictive values."""
        try:
            y_true, y_pred = self._validate_binary_inputs(y_true, y_pred)

            cm = self._create_contingency_table(y_true, y_pred)
            tn, fp = cm[0, 0], cm[0, 1]
            fn, tp = cm[1, 0], cm[1, 1]

            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

            return {
                'positive_predictive_value': round(ppv, self.config.round_decimals),
                'negative_predictive_value': round(npv, self.config.round_decimals),
                'precision': round(ppv, self.config.round_decimals)  # PPV is same as precision
            }

        except Exception as e:
            return {'error': str(e)}

    def likelihood_ratios(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate positive and negative likelihood ratios."""
        try:
            sens_spec = self.sensitivity_specificity(y_true, y_pred)

            if 'error' in sens_spec:
                return sens_spec

            sensitivity = sens_spec['sensitivity']
            specificity = sens_spec['specificity']

            # Positive likelihood ratio
            plr = sensitivity / (1 - specificity) if specificity != 1 else float('inf')

            # Negative likelihood ratio
            nlr = (1 - sensitivity) / specificity if specificity != 0 else float('inf')

            return {
                'positive_likelihood_ratio': round(plr, self.config.round_decimals) if plr != float('inf') else 'inf',
                'negative_likelihood_ratio': round(nlr, self.config.round_decimals) if nlr != float('inf') else 'inf'
            }

        except Exception as e:
            return {'error': str(e)}

    def _validate_binary_inputs(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate inputs for binary classification."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

        # Check if binary
        unique_true = set(y_true)
        unique_pred = set(y_pred)

        if not unique_true.issubset({0, 1}) or not unique_pred.issubset({0, 1}):
            raise ValueError("Binary classification metrics require labels to be 0 or 1")

        return y_true, y_pred

    def _create_contingency_table(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Create 2x2 contingency table."""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        return cm


class ClusteringMetrics:
    """
    Clustering validation metrics.
    """

    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()

    def silhouette_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering."""
        try:
            from sklearn.metrics import silhouette_score as sk_silhouette

            if len(set(labels)) < 2:
                return {'error': 'Silhouette score requires at least 2 clusters'}

            score = sk_silhouette(X, labels, metric=self.config.clustering_metric)
            return round(score, self.config.round_decimals)

        except Exception as e:
            return {'error': str(e)}

    def calinski_harabasz_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Calinski-Harabasz score (variance ratio criterion)."""
        try:
            from sklearn.metrics import calinski_harabasz_score as sk_calinski

            if len(set(labels)) < 2:
                return {'error': 'Calinski-Harabasz score requires at least 2 clusters'}

            score = sk_calinski(X, labels)
            return round(score, self.config.round_decimals)

        except Exception as e:
            return {'error': str(e)}

    def davies_bouldin_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Davies-Bouldin score."""
        try:
            from sklearn.metrics import davies_bouldin_score as sk_davies_bouldin

            if len(set(labels)) < 2:
                return {'error': 'Davies-Bouldin score requires at least 2 clusters'}

            score = sk_davies_bouldin(X, labels)
            return round(score, self.config.round_decimals)

        except Exception as e:
            return {'error': str(e)}

    def inertia(self, X: np.ndarray, labels: np.ndarray, cluster_centers: np.ndarray = None) -> float:
        """Calculate within-cluster sum of squares (inertia)."""
        try:
            X = np.asarray(X)
            labels = np.asarray(labels)

            if cluster_centers is None:
                # Calculate centroids
                unique_labels = np.unique(labels)
                cluster_centers = np.array([X[labels == label].mean(axis=0) for label in unique_labels])

            inertia = 0
            for i, label in enumerate(np.unique(labels)):
                cluster_points = X[labels == label]
                centroid = cluster_centers[i]
                inertia += np.sum((cluster_points - centroid) ** 2)

            return round(inertia, self.config.round_decimals)

        except Exception as e:
            return {'error': str(e)}

    def dunn_index(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Dunn index for clustering validation."""
        try:
            from scipy.spatial.distance import cdist

            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return {'error': 'Dunn index requires at least 2 clusters'}

            # Calculate inter-cluster distances (minimum)
            min_inter_cluster_dist = float('inf')
            for i in range(len(unique_labels)):
                for j in range(i + 1, len(unique_labels)):
                    cluster_i = X[labels == unique_labels[i]]
                    cluster_j = X[labels == unique_labels[j]]

                    # Minimum distance between any two points in different clusters
                    distances = cdist(cluster_i, cluster_j, metric=self.config.clustering_metric)
                    min_dist = np.min(distances)
                    min_inter_cluster_dist = min(min_inter_cluster_dist, min_dist)

            # Calculate intra-cluster distances (maximum)
            max_intra_cluster_dist = 0
            for label in unique_labels:
                cluster_points = X[labels == label]
                if len(cluster_points) > 1:
                    distances = cdist(cluster_points, cluster_points, metric=self.config.clustering_metric)
                    max_dist = np.max(distances)
                    max_intra_cluster_dist = max(max_intra_cluster_dist, max_dist)

            if max_intra_cluster_dist == 0:
                return {'error': 'Cannot calculate Dunn index: zero intra-cluster distance'}

            dunn = min_inter_cluster_dist / max_intra_cluster_dist
            return round(dunn, self.config.round_decimals)

        except Exception as e:
            return {'error': str(e)}

    def comprehensive_clustering_report(self, X: np.ndarray, labels: np.ndarray,
                                      cluster_centers: np.ndarray = None) -> Dict[str, Any]:
        """Generate comprehensive clustering validation report."""
        try:
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)

            report = {
                'analysis_date': datetime.now(),
                'n_samples': len(X),
                'n_features': X.shape[1] if X.ndim > 1 else 1,
                'n_clusters': n_clusters,
                'cluster_labels': unique_labels.tolist(),
                'metrics': {}
            }

            # Calculate various metrics
            if n_clusters >= 2:
                report['metrics']['silhouette_score'] = self.silhouette_score(X, labels)
                report['metrics']['calinski_harabasz_score'] = self.calinski_harabasz_score(X, labels)
                report['metrics']['davies_bouldin_score'] = self.davies_bouldin_score(X, labels)
                report['metrics']['dunn_index'] = self.dunn_index(X, labels)

            report['metrics']['inertia'] = self.inertia(X, labels, cluster_centers)

            # Cluster size distribution
            cluster_sizes = Counter(labels)
            report['cluster_distribution'] = {
                'sizes': dict(cluster_sizes),
                'size_statistics': {
                    'min_size': min(cluster_sizes.values()),
                    'max_size': max(cluster_sizes.values()),
                    'mean_size': round(np.mean(list(cluster_sizes.values())), self.config.round_decimals),
                    'std_size': round(np.std(list(cluster_sizes.values())), self.config.round_decimals)
                }
            }

            return report

        except Exception as e:
            return {'error': str(e)}


class AdvancedMetrics:
    """
    Advanced validation metrics and statistical tests.
    """

    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()

    def gini_coefficient(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Gini coefficient for model discrimination."""
        try:
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)

            # Sort by predicted probabilities
            sorted_indices = np.argsort(y_pred)
            y_true_sorted = y_true[sorted_indices]

            n = len(y_true)
            cumsum = np.cumsum(y_true_sorted)

            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
            return round(gini, self.config.round_decimals)

        except Exception as e:
            return {'error': str(e)}

    def feature_importance_gini(self, y: np.ndarray, X: np.ndarray) -> Dict[str, Any]:
        """Calculate Gini importance for features."""
        try:
            from sklearn.tree import DecisionTreeClassifier

            # Fit a decision tree to calculate Gini importance
            dt = DecisionTreeClassifier(random_state=42)
            dt.fit(X, y)

            importances = dt.feature_importances_

            # Create feature importance ranking
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            importance_pairs = list(zip(feature_names, importances))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)

            return {
                'gini_importances': np.round(importances, self.config.round_decimals).tolist(),
                'ranked_features': [
                    {'feature': feat, 'importance': round(imp, self.config.round_decimals)}
                    for feat, imp in importance_pairs
                ],
                'total_importance': round(np.sum(importances), self.config.round_decimals)
            }

        except Exception as e:
            return {'error': str(e)}

    def concordance_index(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate concordance index (C-index)."""
        try:
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)

            n = len(y_true)
            concordant = 0
            total_pairs = 0

            for i in range(n):
                for j in range(i + 1, n):
                    if y_true[i] != y_true[j]:  # Only consider different outcomes
                        total_pairs += 1
                        if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                           (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                            concordant += 1
                        elif y_pred[i] == y_pred[j]:  # Tie in predictions
                            concordant += 0.5

            if total_pairs == 0:
                return {'error': 'No valid pairs for concordance calculation'}

            c_index = concordant / total_pairs
            return round(c_index, self.config.round_decimals)

        except Exception as e:
            return {'error': str(e)}

    def kolmogorov_smirnov_test(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test for distribution comparison."""
        try:
            statistic, p_value = stats.ks_2samp(y_true, y_pred)

            return {
                'ks_statistic': round(statistic, self.config.round_decimals),
                'p_value': round(p_value, self.config.round_decimals),
                'reject_null': p_value < (1 - self.config.confidence_level),
                'interpretation': 'Distributions are different' if p_value < (1 - self.config.confidence_level) else 'Distributions are similar'
            }

        except Exception as e:
            return {'error': str(e)}

    def anderson_darling_test(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Perform Anderson-Darling test for normality."""
        try:
            statistic, critical_values, significance_levels = stats.anderson(residuals, dist='norm')

            # Find significance level
            sig_level = None
            for i, cv in enumerate(critical_values):
                if statistic > cv:
                    sig_level = significance_levels[i]
                else:
                    break

            return {
                'ad_statistic': round(statistic, self.config.round_decimals),
                'critical_values': np.round(critical_values, self.config.round_decimals).tolist(),
                'significance_levels': significance_levels.tolist(),
                'reject_normality': sig_level is not None,
                'significance_level': sig_level
            }

        except Exception as e:
            return {'error': str(e)}

    def ljung_box_test(self, residuals: np.ndarray, lags: int = 10) -> Dict[str, Any]:
        """Perform Ljung-Box test for autocorrelation in residuals."""
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox

            result = acorr_ljungbox(residuals, lags=lags, return_df=True)

            return {
                'ljung_box_statistics': result['lb_stat'].round(self.config.round_decimals).tolist(),
                'p_values': result['lb_pvalue'].round(self.config.round_decimals).tolist(),
                'lags': list(range(1, lags + 1)),
                'reject_independence': (result['lb_pvalue'] < (1 - self.config.confidence_level)).any()
            }

        except Exception as e:
            return {'error': str(e)}


class ValidationMetrics:
    """
    Main class that combines all validation metrics.
    """

    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.regression = RegressionMetrics(self.config)
        self.classification = ClassificationMetrics(self.config)
        self.binary_classification = BinaryClassificationMetrics(self.config)
        self.clustering = ClusteringMetrics(self.config)
        self.advanced = AdvancedMetrics(self.config)

    def auto_evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                     problem_type: str = 'auto', **kwargs) -> Dict[str, Any]:
        """
        Automatically determine problem type and generate appropriate metrics.

        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        problem_type : str
            'auto', 'regression', 'classification', 'binary_classification'
        **kwargs : additional arguments for specific metrics

        Returns:
        --------
        Dict with comprehensive evaluation results
        """
        try:
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)

            # Auto-detect problem type
            if problem_type == 'auto':
                unique_true = len(np.unique(y_true))
                unique_pred = len(np.unique(y_pred))

                if unique_true == 2 and unique_pred == 2 and set(y_true).issubset({0, 1}) and set(y_pred).issubset({0, 1}):
                    problem_type = 'binary_classification'
                elif unique_true <= 20 and unique_pred <= 20:  # Arbitrary threshold
                    problem_type = 'classification'
                else:
                    problem_type = 'regression'

            result = {
                'problem_type': problem_type,
                'detected_automatically': kwargs.get('problem_type', 'auto') == 'auto',
                'analysis_date': datetime.now(),
                'data_info': {
                    'n_samples': len(y_true),
                    'unique_true_values': len(np.unique(y_true)),
                    'unique_pred_values': len(np.unique(y_pred))
                }
            }

            # Generate appropriate metrics
            if problem_type == 'regression':
                result['metrics'] = self.regression.comprehensive_regression_report(
                    y_true, y_pred, kwargs.get('n_features')
                )

            elif problem_type == 'binary_classification':
                result['metrics'] = {}
                result['metrics'].update(self.binary_classification.odds_ratio(y_true, y_pred))
                result['metrics'].update(self.binary_classification.risk_ratio(y_true, y_pred))
                result['metrics'].update(self.binary_classification.sensitivity_specificity(y_true, y_pred))
                result['metrics']['classification_report'] = self.classification.classification_report(y_true, y_pred)

            elif problem_type == 'classification':
                result['metrics'] = self.classification.classification_report(y_true, y_pred)

            return result

        except Exception as e:
            return {'error': str(e)}

    def compare_models(self, y_true: np.ndarray, predictions_dict: Dict[str, np.ndarray],
                      problem_type: str = 'auto') -> Dict[str, Any]:
        """
        Compare multiple models using appropriate metrics.

        Parameters:
        -----------
        y_true : array-like
            True values
        predictions_dict : Dict[str, array-like]
            Dictionary of model names and their predictions
        problem_type : str
            Problem type ('auto', 'regression', 'classification')

        Returns:
        --------
        Dict with model comparison results
        """
        try:
            comparison_results = {
                'comparison_date': datetime.now(),
                'problem_type': problem_type,
                'models_compared': list(predictions_dict.keys()),
                'model_metrics': {},
                'ranking': {}
            }

            # Evaluate each model
            for model_name, y_pred in predictions_dict.items():
                model_result = self.auto_evaluate(y_true, y_pred, problem_type)
                comparison_results['model_metrics'][model_name] = model_result

            # Create rankings based on key metrics
            if problem_type in ['auto', 'regression']:
                # Rank by R² (higher is better)
                r2_scores = {}
                for model_name, result in comparison_results['model_metrics'].items():
                    if 'metrics' in result and 'metrics' in result['metrics'] and 'r2' in result['metrics']['metrics']:
                        r2_scores[model_name] = result['metrics']['metrics']['r2']

                if r2_scores:
                    comparison_results['ranking']['by_r2'] = sorted(
                        r2_scores.items(), key=lambda x: x[1], reverse=True
                    )

            elif problem_type in ['classification', 'binary_classification']:
                # Rank by accuracy (higher is better)
                accuracy_scores = {}
                for model_name, result in comparison_results['model_metrics'].items():
                    if 'metrics' in result:
                        if 'overall_accuracy' in result['metrics']:
                            accuracy_scores[model_name] = result['metrics']['overall_accuracy']
                        elif 'classification_report' in result['metrics'] and 'overall_accuracy' in result['metrics']['classification_report']:
                            accuracy_scores[model_name] = result['metrics']['classification_report']['overall_accuracy']

                if accuracy_scores:
                    comparison_results['ranking']['by_accuracy'] = sorted(
                        accuracy_scores.items(), key=lambda x: x[1], reverse=True
                    )

            return comparison_results

        except Exception as e:
            return {'error': str(e)}