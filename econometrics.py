"""
Econometrics Analysis Module

This module provides comprehensive econometric analysis capabilities including
OLS regression, assumption testing, robust regression methods, regularized
regression, non-parametric regression, and difference-in-differences analysis.

Key Features:
- OLS assumptions testing (linearity, multicollinearity, homoskedasticity)
- Standard and robust OLS regression with diagnostic statistics
- Outlier and leverage point analysis using Cook's distance
- Weighted least squares and robust regression (M-estimation)
- Regularized regression (Ridge, Lasso, Elastic-net) with cross-validation
- Non-parametric regression (LOESS, local averaging)
- Difference-in-differences analysis for causal inference
- Endogeneity testing (Wu-Hausman test)
- Matching methods (nearest neighbor, Mahalanobis, propensity score)
- Instrumental variables and Two-Stage Least Squares (2SLS)
- Panel data and fixed effects regression
- Regression discontinuity design
- Autocorrelation testing (Durbin-Watson)
- Granger causality testing
- Spurious regression detection

Classes:
    OLSAnalyzer: Comprehensive OLS regression and diagnostics
    RobustRegression: Robust regression methods (Huber, Bisquare, etc.)
    RegularizedRegression: Penalized regression with cross-validation
    NonParametricRegression: Local and LOESS regression
    DifferenceInDifferences: Pre-post causal analysis
    EndogeneityTester: Wu-Hausman and other endogeneity tests
    MatchingMethods: Nearest neighbor, Mahalanobis, propensity score matching
    InstrumentalVariables: IV and 2SLS regression with instrument tests
    PanelDataAnalyzer: Fixed effects and panel data methods
    RegressionDiscontinuity: RDD analysis with bandwidth selection
    TimeSeriesEconometrics: Autocorrelation and Granger causality tests

Dependencies:
    - pandas
    - numpy
    - scipy
    - scikit-learn
    - statsmodels
    - matplotlib
    - seaborn (optional)
    - warnings

Author: Financial Data Analysis Toolkit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Union, Tuple, Optional, List, Dict, Any
from scipy import stats
from scipy.spatial.distance import pdist, squareform

# Core dependencies
try:
    import statsmodels.api as sm
    import statsmodels.stats.api as sms
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.regression.linear_model import WLS
    from statsmodels.robust import norms
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("Statsmodels not available. Core econometric functions will be limited.")

try:
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("Scikit-learn not available. Some regression methods will be limited.")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    warnings.warn("Seaborn not available. Some visualization features will be limited.")

class OLSAnalyzer:
    """
    Comprehensive Ordinary Least Squares regression analysis with assumption testing
    and diagnostic capabilities.
    """

    def __init__(self):
        """Initialize the OLS analyzer."""
        self.model = None
        self.results = None
        self.X = None
        self.y = None
        self.fitted_values = None
        self.residuals = None
        self.diagnostics = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, add_constant: bool = True) -> 'OLSAnalyzer':
        """
        Fit OLS regression model.

        Parameters:
        -----------
        X : pd.DataFrame
            Independent variables
        y : pd.Series
            Dependent variable
        add_constant : bool, default True
            Whether to add intercept term

        Returns:
        --------
        self : OLSAnalyzer
            Fitted analyzer instance
        """
        if not HAS_STATSMODELS:
            raise ImportError("Statsmodels required for OLS analysis")

        self.X = X.copy()
        self.y = y.copy()

        # Add constant if requested
        if add_constant:
            X_with_const = sm.add_constant(X)
        else:
            X_with_const = X

        # Fit model
        self.model = sm.OLS(y, X_with_const)
        self.results = self.model.fit()

        # Store predictions and residuals
        self.fitted_values = self.results.fittedvalues
        self.residuals = self.results.resid

        return self

    def test_linearity(self, plot: bool = True) -> Dict[str, Any]:
        """
        Test linear functional form using residual plots and RESET test.

        Parameters:
        -----------
        plot : bool, default True
            Whether to create diagnostic plots

        Returns:
        --------
        dict
            Linearity test results
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")

        linearity_results = {}

        # RESET test for functional form
        try:
            reset_test = sms.linear_reset(self.results)
            linearity_results['reset_test'] = {
                'statistic': reset_test[0],
                'p_value': reset_test[1],
                'conclusion': 'Linear form OK' if reset_test[1] > 0.05 else 'Non-linear relationship detected'
            }
        except:
            linearity_results['reset_test'] = {'error': 'Could not perform RESET test'}

        # Residual vs fitted plot analysis
        correlation_resid_fitted = np.corrcoef(self.fitted_values, self.residuals)[0, 1]
        linearity_results['residual_correlation'] = {
            'correlation': correlation_resid_fitted,
            'conclusion': 'Good linearity' if abs(correlation_resid_fitted) < 0.1 else 'Potential non-linearity'
        }

        if plot:
            self._plot_linearity_diagnostics()

        self.diagnostics['linearity'] = linearity_results
        return linearity_results

    def test_multicollinearity(self) -> Dict[str, Any]:
        """
        Test for multicollinearity using VIF and condition number.

        Returns:
        --------
        dict
            Multicollinearity test results
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")

        multicollinearity_results = {}

        # Calculate VIF for each variable
        X_with_const = self.results.model.exog
        vif_data = []

        # Skip constant term for VIF calculation
        start_idx = 1 if self.results.model.exog.shape[1] > len(self.X.columns) else 0

        for i in range(start_idx, X_with_const.shape[1]):
            try:
                vif_value = variance_inflation_factor(X_with_const, i)
                var_name = self.X.columns[i - start_idx] if start_idx == 1 else self.X.columns[i]
                vif_data.append({
                    'Variable': var_name,
                    'VIF': vif_value,
                    'Status': 'OK' if vif_value < 5 else ('Moderate' if vif_value < 10 else 'High multicollinearity')
                })
            except:
                continue

        multicollinearity_results['vif_analysis'] = vif_data

        # Condition number
        eigenvals = np.linalg.eigvals(np.dot(X_with_const.T, X_with_const))
        condition_number = np.sqrt(eigenvals.max() / eigenvals.min())
        multicollinearity_results['condition_number'] = {
            'value': condition_number,
            'status': 'OK' if condition_number < 15 else ('Moderate' if condition_number < 30 else 'High multicollinearity')
        }

        # Correlation matrix
        corr_matrix = self.X.corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append({
                        'Variable1': corr_matrix.columns[i],
                        'Variable2': corr_matrix.columns[j],
                        'Correlation': corr_val
                    })

        multicollinearity_results['high_correlations'] = high_corr_pairs

        self.diagnostics['multicollinearity'] = multicollinearity_results
        return multicollinearity_results

    def test_homoskedasticity(self, plot: bool = True) -> Dict[str, Any]:
        """
        Test for homoskedasticity using Breusch-Pagan and White tests.

        Parameters:
        -----------
        plot : bool, default True
            Whether to create diagnostic plots

        Returns:
        --------
        dict
            Heteroskedasticity test results
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")

        hetero_results = {}

        # Breusch-Pagan test
        try:
            bp_stat, bp_p, bp_f, bp_f_p = het_breuschpagan(self.residuals, self.results.model.exog)
            hetero_results['breusch_pagan'] = {
                'statistic': bp_stat,
                'p_value': bp_p,
                'f_statistic': bp_f,
                'f_p_value': bp_f_p,
                'conclusion': 'Homoskedastic' if bp_p > 0.05 else 'Heteroskedastic'
            }
        except Exception as e:
            hetero_results['breusch_pagan'] = {'error': f'Could not perform BP test: {str(e)}'}

        # White test
        try:
            white_stat, white_p, white_f, white_f_p = het_white(self.residuals, self.results.model.exog)
            hetero_results['white_test'] = {
                'statistic': white_stat,
                'p_value': white_p,
                'f_statistic': white_f,
                'f_p_value': white_f_p,
                'conclusion': 'Homoskedastic' if white_p > 0.05 else 'Heteroskedastic'
            }
        except Exception as e:
            hetero_results['white_test'] = {'error': f'Could not perform White test: {str(e)}'}

        if plot:
            self._plot_heteroskedasticity_diagnostics()

        self.diagnostics['heteroskedasticity'] = hetero_results
        return hetero_results

    def run_regression(self, X: pd.DataFrame, y: pd.Series, add_constant: bool = True,
                      test_assumptions: bool = True) -> Dict[str, Any]:
        """
        Run complete OLS regression with significance tests and diagnostics.

        Parameters:
        -----------
        X : pd.DataFrame
            Independent variables
        y : pd.Series
            Dependent variable
        add_constant : bool, default True
            Whether to add intercept
        test_assumptions : bool, default True
            Whether to run assumption tests

        Returns:
        --------
        dict
            Complete regression results
        """
        # Fit model
        self.fit(X, y, add_constant)

        regression_results = {
            'model_summary': {
                'r_squared': self.results.rsquared,
                'adj_r_squared': self.results.rsquared_adj,
                'f_statistic': self.results.fvalue,
                'f_p_value': self.results.f_pvalue,
                'aic': self.results.aic,
                'bic': self.results.bic,
                'n_observations': self.results.nobs
            },
            'coefficients': self._extract_coefficient_info(),
            'joint_significance': self._joint_f_test()
        }

        # Run assumption tests if requested
        if test_assumptions:
            regression_results['assumptions'] = {
                'linearity': self.test_linearity(plot=False),
                'multicollinearity': self.test_multicollinearity(),
                'homoskedasticity': self.test_homoskedasticity(plot=False)
            }

        return regression_results

    def joint_f_test(self, restrictions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform joint F-test for significance of coefficients.

        Parameters:
        -----------
        restrictions : list of str, optional
            List of restrictions to test. If None, tests all slope coefficients.

        Returns:
        --------
        dict
            F-test results
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")

        if restrictions is None:
            # Test all slope coefficients (exclude constant)
            param_names = list(self.results.params.index)
            if 'const' in param_names:
                restrictions = [f"{name} = 0" for name in param_names if name != 'const']
            else:
                restrictions = [f"{name} = 0" for name in param_names]

        try:
            f_test = self.results.f_test(restrictions)
            return {
                'f_statistic': f_test.fvalue[0, 0],
                'p_value': f_test.pvalue,
                'df_num': f_test.df_num,
                'df_denom': f_test.df_denom,
                'conclusion': 'Jointly significant' if f_test.pvalue < 0.05 else 'Not jointly significant',
                'restrictions': restrictions
            }
        except Exception as e:
            return {'error': f'Could not perform F-test: {str(e)}'}

    def analyze_outliers_leverage(self, plot: bool = True) -> Dict[str, Any]:
        """
        Analyze outliers and leverage points using Cook's distance and other measures.

        Parameters:
        -----------
        plot : bool, default True
            Whether to create diagnostic plots

        Returns:
        --------
        dict
            Outlier and leverage analysis results
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")

        # Get influence measures
        influence = self.results.get_influence()

        # Cook's distance
        cooks_d = influence.cooks_distance[0]
        cooks_threshold = 4 / len(self.y)

        # Leverage values (hat values)
        leverage = influence.hat_matrix_diag
        leverage_threshold = 2 * self.results.model.exog.shape[1] / len(self.y)

        # Standardized residuals
        std_resid = influence.resid_studentized_internal

        # DFFITS
        dffits = influence.dffits[0]
        dffits_threshold = 2 * np.sqrt(self.results.model.exog.shape[1] / len(self.y))

        # Identify problematic observations
        outliers = {
            'high_cooks_distance': np.where(cooks_d > cooks_threshold)[0].tolist(),
            'high_leverage': np.where(leverage > leverage_threshold)[0].tolist(),
            'high_standardized_residuals': np.where(np.abs(std_resid) > 3)[0].tolist(),
            'high_dffits': np.where(np.abs(dffits) > dffits_threshold)[0].tolist()
        }

        outlier_results = {
            'cooks_distance': {
                'values': cooks_d,
                'threshold': cooks_threshold,
                'outliers': outliers['high_cooks_distance']
            },
            'leverage': {
                'values': leverage,
                'threshold': leverage_threshold,
                'high_leverage_points': outliers['high_leverage']
            },
            'standardized_residuals': {
                'values': std_resid,
                'outliers': outliers['high_standardized_residuals']
            },
            'dffits': {
                'values': dffits,
                'threshold': dffits_threshold,
                'outliers': outliers['high_dffits']
            },
            'summary': self._summarize_influential_points(outliers)
        }

        if plot:
            self._plot_outlier_diagnostics(influence, cooks_d, leverage, std_resid)

        self.diagnostics['outliers_leverage'] = outlier_results
        return outlier_results

    def multivariate_ols(self, X: pd.DataFrame, Y: pd.DataFrame,
                        add_constant: bool = True) -> Dict[str, Any]:
        """
        Run multivariate OLS regression (multiple dependent variables).

        Parameters:
        -----------
        X : pd.DataFrame
            Independent variables
        Y : pd.DataFrame
            Multiple dependent variables
        add_constant : bool, default True
            Whether to add intercept

        Returns:
        --------
        dict
            Multivariate regression results
        """
        if add_constant:
            X_with_const = sm.add_constant(X)
        else:
            X_with_const = X

        multivariate_results = {}

        # Fit separate models for each dependent variable
        for col in Y.columns:
            model = sm.OLS(Y[col], X_with_const)
            results = model.fit()

            multivariate_results[col] = {
                'coefficients': results.params.to_dict(),
                'std_errors': results.bse.to_dict(),
                't_values': results.tvalues.to_dict(),
                'p_values': results.pvalues.to_dict(),
                'r_squared': results.rsquared,
                'adj_r_squared': results.rsquared_adj,
                'f_statistic': results.fvalue,
                'f_p_value': results.f_pvalue
            }

        # Overall multivariate tests could be added here
        return multivariate_results

    def weighted_least_squares(self, X: pd.DataFrame, y: pd.Series, weights: pd.Series,
                              add_constant: bool = True) -> Dict[str, Any]:
        """
        Run Weighted Least Squares regression.

        Parameters:
        -----------
        X : pd.DataFrame
            Independent variables
        y : pd.Series
            Dependent variable
        weights : pd.Series
            Weights for observations
        add_constant : bool, default True
            Whether to add intercept

        Returns:
        --------
        dict
            WLS regression results
        """
        if add_constant:
            X_with_const = sm.add_constant(X)
        else:
            X_with_const = X

        # Fit WLS model
        wls_model = WLS(y, X_with_const, weights=weights)
        wls_results = wls_model.fit()

        return {
            'coefficients': wls_results.params.to_dict(),
            'std_errors': wls_results.bse.to_dict(),
            't_values': wls_results.tvalues.to_dict(),
            'p_values': wls_results.pvalues.to_dict(),
            'r_squared': wls_results.rsquared,
            'adj_r_squared': wls_results.rsquared_adj,
            'f_statistic': wls_results.fvalue,
            'f_p_value': wls_results.f_pvalue,
            'weighted_residuals': wls_results.wresid,
            'summary': str(wls_results.summary())
        }

    def _extract_coefficient_info(self) -> pd.DataFrame:
        """Extract coefficient information with significance tests."""
        coef_df = pd.DataFrame({
            'Coefficient': self.results.params,
            'Std_Error': self.results.bse,
            't_statistic': self.results.tvalues,
            'p_value': self.results.pvalues,
            'CI_lower': self.results.conf_int()[0],
            'CI_upper': self.results.conf_int()[1]
        })

        # Add significance indicators
        coef_df['Significant'] = coef_df['p_value'] < 0.05
        coef_df['Significance_Level'] = pd.cut(coef_df['p_value'],
                                              bins=[0, 0.01, 0.05, 0.1, 1],
                                              labels=['***', '**', '*', ''])

        return coef_df

    def _joint_f_test(self) -> Dict[str, Any]:
        """Perform joint F-test for all slope coefficients."""
        return {
            'f_statistic': self.results.fvalue,
            'p_value': self.results.f_pvalue,
            'df_num': self.results.df_model,
            'df_denom': self.results.df_resid,
            'conclusion': 'Model significant' if self.results.f_pvalue < 0.05 else 'Model not significant'
        }

    def _summarize_influential_points(self, outliers: Dict) -> str:
        """Summarize influential points analysis."""
        total_outliers = set()
        for outlier_list in outliers.values():
            total_outliers.update(outlier_list)

        summary = f"Total potentially influential observations: {len(total_outliers)}\n"
        summary += f"High Cook's distance: {len(outliers['high_cooks_distance'])}\n"
        summary += f"High leverage: {len(outliers['high_leverage'])}\n"
        summary += f"High standardized residuals: {len(outliers['high_standardized_residuals'])}\n"
        summary += f"High DFFITS: {len(outliers['high_dffits'])}\n"

        if total_outliers:
            summary += f"Observation indices to investigate: {sorted(list(total_outliers))}"

        return summary

    def _plot_linearity_diagnostics(self):
        """Create linearity diagnostic plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Linearity Diagnostics', fontsize=16, fontweight='bold')

        # Residuals vs Fitted
        axes[0, 0].scatter(self.fitted_values, self.residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        axes[0, 0].grid(True, alpha=0.3)

        # Q-Q plot of residuals
        stats.probplot(self.residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot of Residuals')
        axes[0, 1].grid(True, alpha=0.3)

        # Scale-Location plot
        sqrt_abs_resid = np.sqrt(np.abs(self.residuals))
        axes[1, 0].scatter(self.fitted_values, sqrt_abs_resid, alpha=0.6)
        axes[1, 0].set_xlabel('Fitted Values')
        axes[1, 0].set_ylabel('√|Residuals|')
        axes[1, 0].set_title('Scale-Location Plot')
        axes[1, 0].grid(True, alpha=0.3)

        # Residuals histogram
        axes[1, 1].hist(self.residuals, bins=20, density=True, alpha=0.7)
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Residuals Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _plot_heteroskedasticity_diagnostics(self):
        """Create heteroskedasticity diagnostic plots."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Heteroskedasticity Diagnostics', fontsize=16, fontweight='bold')

        # Residuals vs Fitted
        axes[0].scatter(self.fitted_values, self.residuals, alpha=0.6)
        axes[0].axhline(y=0, color='red', linestyle='--')
        axes[0].set_xlabel('Fitted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Fitted Values')
        axes[0].grid(True, alpha=0.3)

        # Absolute residuals vs fitted
        abs_resid = np.abs(self.residuals)
        axes[1].scatter(self.fitted_values, abs_resid, alpha=0.6)
        axes[1].set_xlabel('Fitted Values')
        axes[1].set_ylabel('|Residuals|')
        axes[1].set_title('Absolute Residuals vs Fitted')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _plot_outlier_diagnostics(self, influence, cooks_d, leverage, std_resid):
        """Create outlier and leverage diagnostic plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Outlier and Leverage Diagnostics', fontsize=16, fontweight='bold')

        # Cook's distance
        axes[0, 0].stem(range(len(cooks_d)), cooks_d, linefmt='b-', markerfmt='bo', basefmt=' ')
        axes[0, 0].axhline(y=4/len(self.y), color='red', linestyle='--', label='Threshold')
        axes[0, 0].set_xlabel('Observation')
        axes[0, 0].set_ylabel("Cook's Distance")
        axes[0, 0].set_title("Cook's Distance")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Leverage vs Residuals
        axes[0, 1].scatter(leverage, std_resid, alpha=0.6)
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].axvline(x=2*self.results.model.exog.shape[1]/len(self.y), color='red', linestyle='--')
        axes[0, 1].set_xlabel('Leverage')
        axes[0, 1].set_ylabel('Standardized Residuals')
        axes[0, 1].set_title('Leverage vs Standardized Residuals')
        axes[0, 1].grid(True, alpha=0.3)

        # Leverage
        axes[1, 0].stem(range(len(leverage)), leverage, linefmt='g-', markerfmt='go', basefmt=' ')
        axes[1, 0].axhline(y=2*self.results.model.exog.shape[1]/len(self.y), color='red', linestyle='--')
        axes[1, 0].set_xlabel('Observation')
        axes[1, 0].set_ylabel('Leverage')
        axes[1, 0].set_title('Leverage Values')
        axes[1, 0].grid(True, alpha=0.3)

        # Standardized Residuals
        axes[1, 1].stem(range(len(std_resid)), std_resid, linefmt='r-', markerfmt='ro', basefmt=' ')
        axes[1, 1].axhline(y=3, color='red', linestyle='--')
        axes[1, 1].axhline(y=-3, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Observation')
        axes[1, 1].set_ylabel('Standardized Residuals')
        axes[1, 1].set_title('Standardized Residuals')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

class RobustRegression:
    """
    Robust regression methods including M-estimation with various loss functions.
    """

    def __init__(self):
        """Initialize the robust regression analyzer."""
        self.models = {}
        self.results = {}

    def irls_regression(self, X: pd.DataFrame, y: pd.Series, norm_func: str = 'huber',
                       max_iter: int = 50, tol: float = 1e-6, add_constant: bool = True) -> Dict[str, Any]:
        """
        Iteratively Reweighted Least Squares (IRLS) regression.

        Parameters:
        -----------
        X : pd.DataFrame
            Independent variables
        y : pd.Series
            Dependent variable
        norm_func : str, default 'huber'
            Loss function ('huber', 'ramsay_novick', 'trimmed_mean', 'least_squares')
        max_iter : int, default 50
            Maximum iterations
        tol : float, default 1e-6
            Convergence tolerance
        add_constant : bool, default True
            Whether to add intercept

        Returns:
        --------
        dict
            IRLS regression results
        """
        if not HAS_STATSMODELS:
            raise ImportError("Statsmodels required for IRLS regression")

        if add_constant:
            X_with_const = sm.add_constant(X)
        else:
            X_with_const = X

        # Get norm function
        norm_map = {
            'huber': norms.HuberT(),
            'ramsay_novick': norms.RamsayE(),
            'trimmed_mean': norms.TrimmedMean(),
            'least_squares': norms.LeastSquares()
        }

        if norm_func not in norm_map:
            raise ValueError(f"Unknown norm function: {norm_func}")

        # Fit robust model
        rlm_model = sm.RLM(y, X_with_const, M=norm_map[norm_func])
        rlm_results = rlm_model.fit(maxiter=max_iter, tol=tol)

        results = {
            'method': f'IRLS ({norm_func})',
            'coefficients': rlm_results.params.to_dict(),
            'std_errors': rlm_results.bse.to_dict(),
            't_values': rlm_results.tvalues.to_dict(),
            'p_values': rlm_results.pvalues.to_dict(),
            'scale': rlm_results.scale,
            'weights': rlm_results.weights,
            'converged': rlm_results.converged,
            'iterations': rlm_results.niter,
            'fitted_values': rlm_results.fittedvalues,
            'residuals': rlm_results.resid
        }

        self.results[f'irls_{norm_func}'] = results
        return results

    def huber_regression(self, X: pd.DataFrame, y: pd.Series, epsilon: float = 1.35,
                        add_constant: bool = True) -> Dict[str, Any]:
        """
        Huber robust regression using sklearn.

        Parameters:
        -----------
        X : pd.DataFrame
            Independent variables
        y : pd.Series
            Dependent variable
        epsilon : float, default 1.35
            Huber parameter
        add_constant : bool, default True
            Whether to add intercept

        Returns:
        --------
        dict
            Huber regression results
        """
        if not HAS_SKLEARN:
            raise ImportError("Scikit-learn required for Huber regression")

        # Fit Huber regression
        huber = HuberRegressor(epsilon=epsilon, fit_intercept=add_constant)
        huber.fit(X, y)

        # Predictions
        y_pred = huber.predict(X)
        residuals = y - y_pred

        results = {
            'method': 'Huber regression',
            'coefficients': dict(zip(X.columns, huber.coef_)),
            'intercept': huber.intercept_ if add_constant else 0,
            'epsilon': epsilon,
            'n_iter': huber.n_iter_,
            'fitted_values': y_pred,
            'residuals': residuals,
            'outliers': huber.outliers_,
            'r_squared': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred)
        }

        if add_constant:
            results['coefficients']['intercept'] = huber.intercept_

        self.results['huber'] = results
        return results

    def bisquare_regression(self, X: pd.DataFrame, y: pd.Series, c: float = 4.685,
                           add_constant: bool = True) -> Dict[str, Any]:
        """
        Bisquare (Tukey's biweight) robust regression.

        Parameters:
        -----------
        X : pd.DataFrame
            Independent variables
        y : pd.Series
            Dependent variable
        c : float, default 4.685
            Tuning parameter
        add_constant : bool, default True
            Whether to add intercept

        Returns:
        --------
        dict
            Bisquare regression results
        """
        if not HAS_STATSMODELS:
            raise ImportError("Statsmodels required for bisquare regression")

        if add_constant:
            X_with_const = sm.add_constant(X)
        else:
            X_with_const = X

        # Create bisquare norm
        bisquare_norm = norms.TukeyBiweight(c=c)

        # Fit model
        rlm_model = sm.RLM(y, X_with_const, M=bisquare_norm)
        rlm_results = rlm_model.fit()

        results = {
            'method': 'Bisquare (Tukey biweight)',
            'coefficients': rlm_results.params.to_dict(),
            'std_errors': rlm_results.bse.to_dict(),
            't_values': rlm_results.tvalues.to_dict(),
            'p_values': rlm_results.pvalues.to_dict(),
            'tuning_parameter_c': c,
            'scale': rlm_results.scale,
            'weights': rlm_results.weights,
            'fitted_values': rlm_results.fittedvalues,
            'residuals': rlm_results.resid
        }

        self.results['bisquare'] = results
        return results

    def andrews_sine_regression(self, X: pd.DataFrame, y: pd.Series, a: float = 1.339,
                               add_constant: bool = True) -> Dict[str, Any]:
        """
        Andrew's sine robust regression.

        Parameters:
        -----------
        X : pd.DataFrame
            Independent variables
        y : pd.Series
            Dependent variable
        a : float, default 1.339
            Tuning parameter
        add_constant : bool, default True
            Whether to add intercept

        Returns:
        --------
        dict
            Andrew's sine regression results
        """
        if not HAS_STATSMODELS:
            raise ImportError("Statsmodels required for Andrew's sine regression")

        if add_constant:
            X_with_const = sm.add_constant(X)
        else:
            X_with_const = X

        # Create Andrew's norm
        andrews_norm = norms.AndrewWave(a=a)

        # Fit model
        rlm_model = sm.RLM(y, X_with_const, M=andrews_norm)
        rlm_results = rlm_model.fit()

        results = {
            'method': "Andrew's sine wave",
            'coefficients': rlm_results.params.to_dict(),
            'std_errors': rlm_results.bse.to_dict(),
            't_values': rlm_results.tvalues.to_dict(),
            'p_values': rlm_results.pvalues.to_dict(),
            'tuning_parameter_a': a,
            'scale': rlm_results.scale,
            'weights': rlm_results.weights,
            'fitted_values': rlm_results.fittedvalues,
            'residuals': rlm_results.resid
        }

        self.results['andrews_sine'] = results
        return results

    def compare_robust_methods(self, X: pd.DataFrame, y: pd.Series,
                              methods: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple robust regression methods.

        Parameters:
        -----------
        X : pd.DataFrame
            Independent variables
        y : pd.Series
            Dependent variable
        methods : list of str, optional
            List of methods to compare

        Returns:
        --------
        pd.DataFrame
            Comparison of robust methods
        """
        if methods is None:
            methods = ['huber', 'bisquare', 'andrews_sine', 'irls_huber']

        comparison_results = []

        for method in methods:
            try:
                if method == 'huber':
                    result = self.huber_regression(X, y)
                elif method == 'bisquare':
                    result = self.bisquare_regression(X, y)
                elif method == 'andrews_sine':
                    result = self.andrews_sine_regression(X, y)
                elif method == 'irls_huber':
                    result = self.irls_regression(X, y, norm_func='huber')
                else:
                    continue

                # Calculate performance metrics
                y_pred = result['fitted_values']
                residuals = result['residuals']

                comparison_results.append({
                    'Method': result['method'],
                    'R_squared': r2_score(y, y_pred) if 'r_squared' not in result else result['r_squared'],
                    'MSE': mean_squared_error(y, y_pred) if 'mse' not in result else result['mse'],
                    'MAE': np.mean(np.abs(residuals)),
                    'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
                    'Robust_Scale': result.get('scale', np.std(residuals))
                })

            except Exception as e:
                print(f"Failed to fit {method}: {str(e)}")

        return pd.DataFrame(comparison_results).sort_values('R_squared', ascending=False)

class RegularizedRegression:
    """
    Regularized regression methods with cross-validation for parameter selection.
    """

    def __init__(self):
        """Initialize the regularized regression analyzer."""
        self.models = {}
        self.cv_results = {}

    def ridge_regression(self, X: pd.DataFrame, y: pd.Series, alpha: Optional[float] = None,
                        cv_folds: int = 5, alpha_range: Tuple[float, float] = (0.01, 100),
                        n_alphas: int = 50) -> Dict[str, Any]:
        """
        Ridge regression with cross-validation for alpha selection.

        Parameters:
        -----------
        X : pd.DataFrame
            Independent variables
        y : pd.Series
            Dependent variable
        alpha : float, optional
            Regularization parameter. If None, uses CV to find optimal.
        cv_folds : int, default 5
            Number of cross-validation folds
        alpha_range : tuple, default (0.01, 100)
            Range for alpha search
        n_alphas : int, default 50
            Number of alphas to test

        Returns:
        --------
        dict
            Ridge regression results
        """
        if not HAS_SKLEARN:
            raise ImportError("Scikit-learn required for Ridge regression")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if alpha is None:
            # Cross-validation to find optimal alpha
            alphas = np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), n_alphas)
            ridge_cv = Ridge()

            cv_scores = []
            for alpha_val in alphas:
                ridge_cv.alpha = alpha_val
                scores = cross_val_score(ridge_cv, X_scaled, y, cv=cv_folds, scoring='r2')
                cv_scores.append(scores.mean())

            optimal_alpha = alphas[np.argmax(cv_scores)]
            self.cv_results['ridge'] = {
                'alphas': alphas,
                'cv_scores': cv_scores,
                'optimal_alpha': optimal_alpha
            }
        else:
            optimal_alpha = alpha

        # Fit final model
        ridge = Ridge(alpha=optimal_alpha)
        ridge.fit(X_scaled, y)

        # Predictions
        y_pred = ridge.predict(X_scaled)

        results = {
            'method': 'Ridge regression (L2)',
            'alpha': optimal_alpha,
            'coefficients': dict(zip(X.columns, ridge.coef_)),
            'intercept': ridge.intercept_,
            'r_squared': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'fitted_values': y_pred,
            'residuals': y - y_pred,
            'cv_used': alpha is None,
            'scaler': scaler
        }

        self.models['ridge'] = ridge
        return results

    def lasso_regression(self, X: pd.DataFrame, y: pd.Series, alpha: Optional[float] = None,
                        cv_folds: int = 5, alpha_range: Tuple[float, float] = (0.01, 10),
                        n_alphas: int = 50) -> Dict[str, Any]:
        """
        Lasso regression with cross-validation for alpha selection.

        Parameters:
        -----------
        X : pd.DataFrame
            Independent variables
        y : pd.Series
            Dependent variable
        alpha : float, optional
            Regularization parameter. If None, uses CV to find optimal.
        cv_folds : int, default 5
            Number of cross-validation folds
        alpha_range : tuple, default (0.01, 10)
            Range for alpha search
        n_alphas : int, default 50
            Number of alphas to test

        Returns:
        --------
        dict
            Lasso regression results
        """
        if not HAS_SKLEARN:
            raise ImportError("Scikit-learn required for Lasso regression")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if alpha is None:
            # Cross-validation to find optimal alpha
            alphas = np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), n_alphas)
            lasso_cv = Lasso(max_iter=2000)

            cv_scores = []
            for alpha_val in alphas:
                try:
                    lasso_cv.alpha = alpha_val
                    scores = cross_val_score(lasso_cv, X_scaled, y, cv=cv_folds, scoring='r2')
                    cv_scores.append(scores.mean())
                except:
                    cv_scores.append(-np.inf)

            optimal_alpha = alphas[np.argmax(cv_scores)]
            self.cv_results['lasso'] = {
                'alphas': alphas,
                'cv_scores': cv_scores,
                'optimal_alpha': optimal_alpha
            }
        else:
            optimal_alpha = alpha

        # Fit final model
        lasso = Lasso(alpha=optimal_alpha, max_iter=2000)
        lasso.fit(X_scaled, y)

        # Predictions
        y_pred = lasso.predict(X_scaled)

        # Identify selected features
        selected_features = X.columns[lasso.coef_ != 0].tolist()

        results = {
            'method': 'Lasso regression (L1)',
            'alpha': optimal_alpha,
            'coefficients': dict(zip(X.columns, lasso.coef_)),
            'intercept': lasso.intercept_,
            'r_squared': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'fitted_values': y_pred,
            'residuals': y - y_pred,
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'cv_used': alpha is None,
            'scaler': scaler
        }

        self.models['lasso'] = lasso
        return results

    def elastic_net_regression(self, X: pd.DataFrame, y: pd.Series, alpha: Optional[float] = None,
                              l1_ratio: Optional[float] = None, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Elastic Net regression with cross-validation for parameter selection.

        Parameters:
        -----------
        X : pd.DataFrame
            Independent variables
        y : pd.Series
            Dependent variable
        alpha : float, optional
            Regularization strength. If None, uses CV to find optimal.
        l1_ratio : float, optional
            L1 penalty ratio. If None, uses CV to find optimal.
        cv_folds : int, default 5
            Number of cross-validation folds

        Returns:
        --------
        dict
            Elastic Net regression results
        """
        if not HAS_SKLEARN:
            raise ImportError("Scikit-learn required for Elastic Net regression")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if alpha is None or l1_ratio is None:
            # Grid search for optimal parameters
            from sklearn.model_selection import GridSearchCV

            elastic_net = ElasticNet(max_iter=2000)
            param_grid = {
                'alpha': np.logspace(-3, 1, 20),
                'l1_ratio': np.linspace(0.1, 0.9, 9)
            }

            grid_search = GridSearchCV(elastic_net, param_grid, cv=cv_folds, scoring='r2')
            grid_search.fit(X_scaled, y)

            optimal_alpha = grid_search.best_params_['alpha']
            optimal_l1_ratio = grid_search.best_params_['l1_ratio']

            self.cv_results['elastic_net'] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
        else:
            optimal_alpha = alpha
            optimal_l1_ratio = l1_ratio

        # Fit final model
        elastic_net = ElasticNet(alpha=optimal_alpha, l1_ratio=optimal_l1_ratio, max_iter=2000)
        elastic_net.fit(X_scaled, y)

        # Predictions
        y_pred = elastic_net.predict(X_scaled)

        # Selected features
        selected_features = X.columns[elastic_net.coef_ != 0].tolist()

        results = {
            'method': 'Elastic Net regression',
            'alpha': optimal_alpha,
            'l1_ratio': optimal_l1_ratio,
            'coefficients': dict(zip(X.columns, elastic_net.coef_)),
            'intercept': elastic_net.intercept_,
            'r_squared': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'fitted_values': y_pred,
            'residuals': y - y_pred,
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'cv_used': alpha is None or l1_ratio is None,
            'scaler': scaler
        }

        self.models['elastic_net'] = elastic_net
        return results

    def compare_regularized_methods(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Compare Ridge, Lasso, and Elastic Net regression methods.

        Parameters:
        -----------
        X : pd.DataFrame
            Independent variables
        y : pd.Series
            Dependent variable

        Returns:
        --------
        pd.DataFrame
            Comparison of regularized methods
        """
        methods = ['ridge', 'lasso', 'elastic_net']
        comparison_results = []

        for method in methods:
            try:
                if method == 'ridge':
                    result = self.ridge_regression(X, y)
                elif method == 'lasso':
                    result = self.lasso_regression(X, y)
                elif method == 'elastic_net':
                    result = self.elastic_net_regression(X, y)

                comparison_results.append({
                    'Method': result['method'],
                    'Alpha': result['alpha'],
                    'L1_Ratio': result.get('l1_ratio', 0 if method == 'ridge' else (1 if method == 'lasso' else result.get('l1_ratio', 'N/A'))),
                    'R_squared': result['r_squared'],
                    'MSE': result['mse'],
                    'RMSE': np.sqrt(result['mse']),
                    'Selected_Features': result.get('n_selected', len(X.columns)),
                    'CV_Used': result['cv_used']
                })

            except Exception as e:
                print(f"Failed to fit {method}: {str(e)}")

        return pd.DataFrame(comparison_results)

    def plot_regularization_path(self, X: pd.DataFrame, y: pd.Series, method: str = 'lasso') -> None:
        """
        Plot regularization path showing coefficient evolution.

        Parameters:
        -----------
        X : pd.DataFrame
            Independent variables
        y : pd.Series
            Dependent variable
        method : str, default 'lasso'
            Regularization method ('ridge', 'lasso', 'elastic_net')
        """
        if not HAS_SKLEARN:
            raise ImportError("Scikit-learn required for regularization path plotting")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Define alpha range
        if method == 'ridge':
            alphas = np.logspace(-3, 3, 100)
            from sklearn.linear_model import ridge_path
            _, coefs = ridge_path(X_scaled, y, alphas=alphas)
        elif method == 'lasso':
            from sklearn.linear_model import lasso_path
            alphas, coefs, _ = lasso_path(X_scaled, y, max_iter=2000)
        else:
            raise ValueError("Method must be 'ridge' or 'lasso'")

        # Create plot
        plt.figure(figsize=(12, 8))
        for i, feature in enumerate(X.columns):
            plt.plot(alphas, coefs[i], label=feature)

        plt.xlabel('Alpha (log scale)')
        plt.ylabel('Coefficients')
        plt.title(f'{method.title()} Regularization Path')
        plt.xscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

class NonParametricRegression:
    """
    Non-parametric regression methods including local averaging and LOESS.
    """

    def __init__(self):
        """Initialize the non-parametric regression analyzer."""
        self.models = {}
        self.results = {}

    def local_averaging(self, X: pd.Series, y: pd.Series, span: float = 0.3,
                       weights: str = 'uniform') -> Dict[str, Any]:
        """
        Local averaging regression (simple case for univariate).

        Parameters:
        -----------
        X : pd.Series
            Independent variable (univariate)
        y : pd.Series
            Dependent variable
        span : float, default 0.3
            Proportion of data to use for each local fit
        weights : str, default 'uniform'
            Weighting scheme ('uniform', 'distance')

        Returns:
        --------
        dict
            Local averaging results
        """
        n = len(X)
        window_size = max(3, int(span * n))

        # Sort data
        sort_idx = np.argsort(X)
        X_sorted = X.iloc[sort_idx]
        y_sorted = y.iloc[sort_idx]

        fitted_values = np.zeros(n)

        for i in range(n):
            # Find local window
            start = max(0, i - window_size // 2)
            end = min(n, start + window_size)

            if end - start < window_size and start > 0:
                start = max(0, end - window_size)

            # Local data
            X_local = X_sorted.iloc[start:end]
            y_local = y_sorted.iloc[start:end]

            if weights == 'uniform':
                fitted_values[i] = y_local.mean()
            elif weights == 'distance':
                # Distance-based weights
                distances = np.abs(X_local - X_sorted.iloc[i])
                if distances.sum() == 0:
                    fitted_values[i] = y_local.mean()
                else:
                    w = 1 / (distances + 1e-10)
                    fitted_values[i] = np.average(y_local, weights=w)

        # Unsort fitted values
        fitted_unsorted = np.zeros(n)
        fitted_unsorted[sort_idx] = fitted_values

        residuals = y - fitted_unsorted

        results = {
            'method': 'Local averaging',
            'span': span,
            'weights': weights,
            'window_size': window_size,
            'fitted_values': fitted_unsorted,
            'residuals': residuals,
            'r_squared': r2_score(y, fitted_unsorted),
            'mse': mean_squared_error(y, fitted_unsorted)
        }

        self.results['local_averaging'] = results
        return results

    def loess_regression(self, X: pd.Series, y: pd.Series, frac: float = 0.3,
                        it: int = 3, delta: float = 0.0) -> Dict[str, Any]:
        """
        LOESS (LOcally WEighted Scatterplot Smoothing) regression.

        Parameters:
        -----------
        X : pd.Series
            Independent variable (univariate)
        y : pd.Series
            Dependent variable
        frac : float, default 0.3
            Fraction of data to use for each local regression
        it : int, default 3
            Number of robustifying iterations
        delta : float, default 0.0
            Distance parameter for local regression

        Returns:
        --------
        dict
            LOESS regression results
        """
        if not HAS_STATSMODELS:
            raise ImportError("Statsmodels required for LOESS regression")

        # Apply LOESS smoothing
        smoothed = lowess(y, X, frac=frac, it=it, delta=delta, return_sorted=False)

        # Calculate residuals
        residuals = y - smoothed

        results = {
            'method': 'LOESS',
            'frac': frac,
            'iterations': it,
            'delta': delta,
            'fitted_values': smoothed,
            'residuals': residuals,
            'r_squared': r2_score(y, smoothed),
            'mse': mean_squared_error(y, smoothed)
        }

        self.results['loess'] = results
        return results

    def optimal_span_selection(self, X: pd.Series, y: pd.Series,
                              span_range: Tuple[float, float] = (0.1, 0.8),
                              n_spans: int = 20, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Select optimal span for LOESS using cross-validation.

        Parameters:
        -----------
        X : pd.Series
            Independent variable
        y : pd.Series
            Dependent variable
        span_range : tuple, default (0.1, 0.8)
            Range of spans to test
        n_spans : int, default 20
            Number of spans to test
        cv_folds : int, default 5
            Number of cross-validation folds

        Returns:
        --------
        dict
            Optimal span selection results
        """
        spans = np.linspace(span_range[0], span_range[1], n_spans)
        cv_scores = []

        # Simple cross-validation
        n = len(X)
        fold_size = n // cv_folds

        for span in spans:
            fold_scores = []

            for fold in range(cv_folds):
                # Create train/test split
                test_start = fold * fold_size
                test_end = (fold + 1) * fold_size if fold < cv_folds - 1 else n

                train_idx = list(range(test_start)) + list(range(test_end, n))
                test_idx = list(range(test_start, test_end))

                if len(train_idx) == 0 or len(test_idx) == 0:
                    continue

                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

                try:
                    # Fit LOESS on training data
                    smoothed_train = lowess(y_train, X_train, frac=span, return_sorted=False)

                    # Predict on test data (simple interpolation)
                    # For simplicity, using nearest neighbor prediction
                    y_pred = []
                    for x_val in X_test:
                        nearest_idx = np.argmin(np.abs(X_train - x_val))
                        y_pred.append(smoothed_train[nearest_idx])

                    score = r2_score(y_test, y_pred)
                    fold_scores.append(score)
                except:
                    continue

            if fold_scores:
                cv_scores.append(np.mean(fold_scores))
            else:
                cv_scores.append(-np.inf)

        # Find optimal span
        optimal_idx = np.argmax(cv_scores)
        optimal_span = spans[optimal_idx]

        return {
            'spans': spans,
            'cv_scores': cv_scores,
            'optimal_span': optimal_span,
            'optimal_score': cv_scores[optimal_idx]
        }

    def plot_nonparametric_fit(self, X: pd.Series, y: pd.Series, method: str = 'loess',
                              **kwargs) -> None:
        """
        Plot non-parametric regression fit.

        Parameters:
        -----------
        X : pd.Series
            Independent variable
        y : pd.Series
            Dependent variable
        method : str, default 'loess'
            Method to use ('loess', 'local_averaging')
        **kwargs
            Additional arguments for the method
        """
        if method == 'loess':
            result = self.loess_regression(X, y, **kwargs)
        elif method == 'local_averaging':
            result = self.local_averaging(X, y, **kwargs)
        else:
            raise ValueError("Method must be 'loess' or 'local_averaging'")

        # Create plot
        plt.figure(figsize=(10, 6))

        # Original data
        plt.scatter(X, y, alpha=0.6, label='Data')

        # Fitted curve
        sort_idx = np.argsort(X)
        plt.plot(X.iloc[sort_idx], result['fitted_values'][sort_idx],
                'r-', linewidth=2, label=f'{method.upper()} fit')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'{method.upper()} Regression (R² = {result["r_squared"]:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

class DifferenceInDifferences:
    """
    Difference-in-Differences analysis for causal inference in pre-post studies.
    """

    def __init__(self):
        """Initialize the Difference-in-Differences analyzer."""
        self.results = {}

    def basic_diff_in_diff(self, data: pd.DataFrame, outcome_col: str,
                          treatment_col: str, time_col: str,
                          pre_period: Any, post_period: Any) -> Dict[str, Any]:
        """
        Basic difference-in-differences analysis.

        Parameters:
        -----------
        data : pd.DataFrame
            Panel data with treatment, control, pre, and post observations
        outcome_col : str
            Name of outcome variable
        treatment_col : str
            Name of treatment indicator (1 for treated, 0 for control)
        time_col : str
            Name of time/period indicator
        pre_period : Any
            Value indicating pre-treatment period
        post_period : Any
            Value indicating post-treatment period

        Returns:
        --------
        dict
            Difference-in-differences results
        """
        # Filter data to pre and post periods
        analysis_data = data[data[time_col].isin([pre_period, post_period])].copy()

        # Create time indicator (1 for post, 0 for pre)
        analysis_data['post'] = (analysis_data[time_col] == post_period).astype(int)
        analysis_data['treated'] = analysis_data[treatment_col]
        analysis_data['treated_post'] = analysis_data['treated'] * analysis_data['post']

        # Calculate group means
        group_means = analysis_data.groupby(['treated', 'post'])[outcome_col].mean().unstack()

        # Calculate differences
        treatment_diff = group_means.loc[1, 1] - group_means.loc[1, 0]  # Post - Pre for treated
        control_diff = group_means.loc[0, 1] - group_means.loc[0, 0]    # Post - Pre for control
        did_estimate = treatment_diff - control_diff

        # Run regression for statistical inference
        if HAS_STATSMODELS:
            formula = f"{outcome_col} ~ treated + post + treated_post"
            model = sm.OLS.from_formula(formula, data=analysis_data)
            reg_results = model.fit()

            did_coef = reg_results.params['treated_post']
            did_se = reg_results.bse['treated_post']
            did_pvalue = reg_results.pvalues['treated_post']
            did_ci = reg_results.conf_int().loc['treated_post']
        else:
            did_coef = did_estimate
            did_se = None
            did_pvalue = None
            did_ci = None
            reg_results = None

        results = {
            'method': 'Basic Difference-in-Differences',
            'group_means': group_means,
            'treatment_effect_estimate': did_estimate,
            'treatment_difference': treatment_diff,
            'control_difference': control_diff,
            'regression_coefficient': did_coef,
            'standard_error': did_se,
            'p_value': did_pvalue,
            'confidence_interval': did_ci,
            'regression_results': reg_results,
            'n_observations': len(analysis_data),
            'n_treated': analysis_data['treated'].sum(),
            'n_control': len(analysis_data) - analysis_data['treated'].sum()
        }

        self.results['basic_did'] = results
        return results

    def diff_in_diff_with_covariates(self, data: pd.DataFrame, outcome_col: str,
                                    treatment_col: str, time_col: str,
                                    covariate_cols: List[str],
                                    pre_period: Any, post_period: Any) -> Dict[str, Any]:
        """
        Difference-in-differences with additional covariates.

        Parameters:
        -----------
        data : pd.DataFrame
            Panel data
        outcome_col : str
            Outcome variable
        treatment_col : str
            Treatment indicator
        time_col : str
            Time indicator
        covariate_cols : list of str
            List of covariate columns
        pre_period : Any
            Pre-treatment period value
        post_period : Any
            Post-treatment period value

        Returns:
        --------
        dict
            Enhanced DiD results with covariates
        """
        if not HAS_STATSMODELS:
            raise ImportError("Statsmodels required for DiD with covariates")

        # Filter data
        analysis_data = data[data[time_col].isin([pre_period, post_period])].copy()

        # Create indicators
        analysis_data['post'] = (analysis_data[time_col] == post_period).astype(int)
        analysis_data['treated'] = analysis_data[treatment_col]
        analysis_data['treated_post'] = analysis_data['treated'] * analysis_data['post']

        # Build formula
        covariates_str = " + ".join(covariate_cols)
        formula = f"{outcome_col} ~ treated + post + treated_post + {covariates_str}"

        # Run regression
        model = sm.OLS.from_formula(formula, data=analysis_data)
        reg_results = model.fit()

        did_coef = reg_results.params['treated_post']
        did_se = reg_results.bse['treated_post']
        did_pvalue = reg_results.pvalues['treated_post']
        did_ci = reg_results.conf_int().loc['treated_post']

        results = {
            'method': 'Difference-in-Differences with Covariates',
            'treatment_effect_estimate': did_coef,
            'standard_error': did_se,
            'p_value': did_pvalue,
            'confidence_interval': did_ci,
            'regression_results': reg_results,
            'covariates_included': covariate_cols,
            'r_squared': reg_results.rsquared,
            'adj_r_squared': reg_results.rsquared_adj,
            'n_observations': len(analysis_data)
        }

        self.results['did_with_covariates'] = results
        return results

    def parallel_trends_test(self, data: pd.DataFrame, outcome_col: str,
                           treatment_col: str, time_col: str,
                           pre_periods: List[Any]) -> Dict[str, Any]:
        """
        Test parallel trends assumption using pre-treatment periods.

        Parameters:
        -----------
        data : pd.DataFrame
            Panel data
        outcome_col : str
            Outcome variable
        treatment_col : str
            Treatment indicator
        time_col : str
            Time indicator
        pre_periods : list
            List of pre-treatment period values

        Returns:
        --------
        dict
            Parallel trends test results
        """
        if not HAS_STATSMODELS:
            raise ImportError("Statsmodels required for parallel trends test")

        # Filter to pre-treatment periods only
        pre_data = data[data[time_col].isin(pre_periods)].copy()

        # Create time dummies for each period
        time_dummies = pd.get_dummies(pre_data[time_col], prefix='time')
        pre_data = pd.concat([pre_data, time_dummies], axis=1)

        # Create interaction terms (treatment × time)
        interaction_cols = []
        for col in time_dummies.columns:
            interaction_col = f"treated_x_{col}"
            pre_data[interaction_col] = pre_data[treatment_col] * pre_data[col]
            interaction_cols.append(interaction_col)

        # Run regression with interactions
        time_cols_str = " + ".join(time_dummies.columns[1:])  # Exclude first period (reference)
        interaction_cols_str = " + ".join(interaction_cols[1:])  # Exclude first interaction

        formula = f"{outcome_col} ~ {treatment_col} + {time_cols_str} + {interaction_cols_str}"

        model = sm.OLS.from_formula(formula, data=pre_data)
        reg_results = model.fit()

        # Test joint significance of interaction terms
        interaction_params = [param for param in reg_results.params.index if 'treated_x_time' in param]
        if interaction_params:
            f_test = reg_results.f_test([f"{param} = 0" for param in interaction_params])
            parallel_trends_pvalue = f_test.pvalue
        else:
            parallel_trends_pvalue = None

        results = {
            'method': 'Parallel Trends Test',
            'regression_results': reg_results,
            'interaction_coefficients': {param: reg_results.params[param]
                                       for param in interaction_params},
            'interaction_pvalues': {param: reg_results.pvalues[param]
                                  for param in interaction_params},
            'joint_f_test_pvalue': parallel_trends_pvalue,
            'parallel_trends_assumption': 'Satisfied' if parallel_trends_pvalue and parallel_trends_pvalue > 0.05 else 'Violated',
            'pre_periods_analyzed': pre_periods,
            'n_observations': len(pre_data)
        }

        self.results['parallel_trends'] = results
        return results

    def plot_did_trends(self, data: pd.DataFrame, outcome_col: str,
                       treatment_col: str, time_col: str) -> None:
        """
        Plot trends for treatment and control groups over time.

        Parameters:
        -----------
        data : pd.DataFrame
            Panel data
        outcome_col : str
            Outcome variable
        treatment_col : str
            Treatment indicator
        time_col : str
            Time indicator
        """
        # Calculate group means by time period
        group_trends = data.groupby([time_col, treatment_col])[outcome_col].mean().unstack()

        plt.figure(figsize=(10, 6))

        # Plot trends
        plt.plot(group_trends.index, group_trends[0], 'b-o', label='Control Group', linewidth=2)
        plt.plot(group_trends.index, group_trends[1], 'r-o', label='Treatment Group', linewidth=2)

        plt.xlabel('Time Period')
        plt.ylabel(f'Average {outcome_col}')
        plt.title('Difference-in-Differences: Group Trends Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

class EndogeneityTester:
    """
    Tests for endogeneity of explanatory variables using various statistical tests.
    """

    def __init__(self):
        """Initialize the endogeneity tester."""
        self.test_results = {}

    def wu_hausman_test(self, y: pd.Series, X_exog: pd.DataFrame, X_endog: pd.DataFrame,
                       instruments: pd.DataFrame, add_constant: bool = True) -> Dict[str, Any]:
        """
        Wu-Hausman test for endogeneity of explanatory variables.

        Parameters:
        -----------
        y : pd.Series
            Dependent variable
        X_exog : pd.DataFrame
            Exogenous explanatory variables
        X_endog : pd.DataFrame
            Potentially endogenous explanatory variables
        instruments : pd.DataFrame
            Instrumental variables
        add_constant : bool, default True
            Whether to add intercept

        Returns:
        --------
        dict
            Wu-Hausman test results
        """
        if not HAS_STATSMODELS:
            raise ImportError("Statsmodels required for Wu-Hausman test")

        # Combine exogenous variables and instruments
        X_all = pd.concat([X_exog, instruments], axis=1)
        if add_constant:
            X_all = sm.add_constant(X_all)

        # First stage: regress endogenous variables on instruments and exogenous variables
        first_stage_results = {}
        residuals_first_stage = pd.DataFrame()

        for endog_var in X_endog.columns:
            first_stage_model = sm.OLS(X_endog[endog_var], X_all)
            first_stage_fit = first_stage_model.fit()
            first_stage_results[endog_var] = first_stage_fit
            residuals_first_stage[f"{endog_var}_resid"] = first_stage_fit.resid

        # Second stage: add first-stage residuals to original regression
        X_original = pd.concat([X_exog, X_endog], axis=1)
        if add_constant:
            X_original = sm.add_constant(X_original)

        X_with_residuals = pd.concat([X_original, residuals_first_stage], axis=1)

        # Estimate augmented regression
        augmented_model = sm.OLS(y, X_with_residuals)
        augmented_results = augmented_model.fit()

        # Test significance of residual terms
        residual_params = [param for param in augmented_results.params.index if '_resid' in param]

        if residual_params:
            # F-test for joint significance of residuals
            restrictions = [f"{param} = 0" for param in residual_params]
            f_test = augmented_results.f_test(restrictions)

            wu_hausman_stat = f_test.fvalue[0, 0] * len(residual_params)
            wu_hausman_pvalue = f_test.pvalue
        else:
            wu_hausman_stat = None
            wu_hausman_pvalue = None

        results = {
            'test_name': 'Wu-Hausman Test',
            'statistic': wu_hausman_stat,
            'p_value': wu_hausman_pvalue,
            'conclusion': 'Endogenous' if wu_hausman_pvalue and wu_hausman_pvalue < 0.05 else 'Exogenous',
            'first_stage_results': first_stage_results,
            'augmented_regression': augmented_results,
            'tested_variables': list(X_endog.columns),
            'instruments_used': list(instruments.columns)
        }

        self.test_results['wu_hausman'] = results
        return results

    def durbin_wu_hausman_test(self, y: pd.Series, X: pd.DataFrame, instruments: pd.DataFrame,
                              suspected_endog: List[str], add_constant: bool = True) -> Dict[str, Any]:
        """
        Alternative implementation of endogeneity test.

        Parameters:
        -----------
        y : pd.Series
            Dependent variable
        X : pd.DataFrame
            All explanatory variables
        instruments : pd.DataFrame
            Instrumental variables
        suspected_endog : list of str
            Names of suspected endogenous variables
        add_constant : bool, default True
            Whether to add intercept

        Returns:
        --------
        dict
            Endogeneity test results
        """
        # Separate exogenous and endogenous variables
        X_endog = X[suspected_endog]
        X_exog = X.drop(columns=suspected_endog)

        return self.wu_hausman_test(y, X_exog, X_endog, instruments, add_constant)

class MatchingMethods:
    """
    Various matching methods for causal inference including nearest neighbor,
    Mahalanobis distance, and propensity score matching.
    """

    def __init__(self):
        """Initialize the matching methods analyzer."""
        self.matching_results = {}
        self.propensity_scores = None

    def nearest_neighbor_matching(self, data: pd.DataFrame, treatment_col: str,
                                 outcome_col: str, covariate_cols: List[str],
                                 n_neighbors: int = 1, with_replacement: bool = False) -> Dict[str, Any]:
        """
        Nearest neighbor matching on covariates.

        Parameters:
        -----------
        data : pd.DataFrame
            Dataset with treatment, outcome, and covariates
        treatment_col : str
            Name of treatment indicator column
        outcome_col : str
            Name of outcome variable column
        covariate_cols : list of str
            List of covariate column names
        n_neighbors : int, default 1
            Number of neighbors to match
        with_replacement : bool, default False
            Whether to sample with replacement

        Returns:
        --------
        dict
            Nearest neighbor matching results
        """
        if not HAS_SKLEARN:
            raise ImportError("Scikit-learn required for nearest neighbor matching")

        # Separate treated and control units
        treated = data[data[treatment_col] == 1].copy()
        control = data[data[treatment_col] == 0].copy()

        # Standardize covariates
        scaler = StandardScaler()
        control_covariates = scaler.fit_transform(control[covariate_cols])
        treated_covariates = scaler.transform(treated[covariate_cols])

        # Fit nearest neighbors
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        nn.fit(control_covariates)

        # Find matches for treated units
        distances, indices = nn.kneighbors(treated_covariates)

        # Create matched dataset
        matched_pairs = []
        used_controls = set()

        for i, treated_idx in enumerate(treated.index):
            for j in range(n_neighbors):
                control_idx = control.iloc[indices[i, j]].name

                if not with_replacement and control_idx in used_controls:
                    continue

                matched_pairs.append({
                    'treated_unit': treated_idx,
                    'control_unit': control_idx,
                    'distance': distances[i, j],
                    'treated_outcome': treated.loc[treated_idx, outcome_col],
                    'control_outcome': control.loc[control_idx, outcome_col]
                })

                if not with_replacement:
                    used_controls.add(control_idx)

        matched_df = pd.DataFrame(matched_pairs)

        # Calculate treatment effect
        if not matched_df.empty:
            treatment_effect = matched_df['treated_outcome'].mean() - matched_df['control_outcome'].mean()

            # Simple t-test
            t_stat, p_value = stats.ttest_rel(matched_df['treated_outcome'], matched_df['control_outcome'])
        else:
            treatment_effect = np.nan
            t_stat = np.nan
            p_value = np.nan

        results = {
            'method': 'Nearest Neighbor Matching',
            'n_neighbors': n_neighbors,
            'with_replacement': with_replacement,
            'n_matched_pairs': len(matched_pairs),
            'treatment_effect': treatment_effect,
            't_statistic': t_stat,
            'p_value': p_value,
            'matched_pairs': matched_df,
            'average_distance': matched_df['distance'].mean() if not matched_df.empty else np.nan
        }

        self.matching_results['nearest_neighbor'] = results
        return results

    def mahalanobis_matching(self, data: pd.DataFrame, treatment_col: str,
                           outcome_col: str, covariate_cols: List[str],
                           n_neighbors: int = 1) -> Dict[str, Any]:
        """
        Mahalanobis distance matching.

        Parameters:
        -----------
        data : pd.DataFrame
            Dataset with treatment, outcome, and covariates
        treatment_col : str
            Name of treatment indicator column
        outcome_col : str
            Name of outcome variable column
        covariate_cols : list of str
            List of covariate column names
        n_neighbors : int, default 1
            Number of neighbors to match

        Returns:
        --------
        dict
            Mahalanobis matching results
        """
        if not HAS_SCIPY_ADVANCED:
            raise ImportError("Advanced scipy functions required for Mahalanobis matching")

        # Separate treated and control units
        treated = data[data[treatment_col] == 1].copy()
        control = data[data[treatment_col] == 0].copy()

        # Calculate covariance matrix from control group
        control_cov = np.cov(control[covariate_cols].T)
        control_cov_inv = np.linalg.inv(control_cov)

        # Calculate Mahalanobis distances
        matched_pairs = []
        used_controls = set()

        for treated_idx in treated.index:
            treated_point = treated.loc[treated_idx, covariate_cols].values

            distances = []
            available_controls = []

            for control_idx in control.index:
                if control_idx in used_controls:
                    continue

                control_point = control.loc[control_idx, covariate_cols].values
                distance = mahalanobis(treated_point, control_point, control_cov_inv)
                distances.append(distance)
                available_controls.append(control_idx)

            if available_controls:
                # Find closest matches
                sorted_indices = np.argsort(distances)

                for i in range(min(n_neighbors, len(available_controls))):
                    best_match_idx = available_controls[sorted_indices[i]]

                    matched_pairs.append({
                        'treated_unit': treated_idx,
                        'control_unit': best_match_idx,
                        'mahalanobis_distance': distances[sorted_indices[i]],
                        'treated_outcome': treated.loc[treated_idx, outcome_col],
                        'control_outcome': control.loc[best_match_idx, outcome_col]
                    })

                    used_controls.add(best_match_idx)

        matched_df = pd.DataFrame(matched_pairs)

        # Calculate treatment effect
        if not matched_df.empty:
            treatment_effect = matched_df['treated_outcome'].mean() - matched_df['control_outcome'].mean()
            t_stat, p_value = stats.ttest_rel(matched_df['treated_outcome'], matched_df['control_outcome'])
        else:
            treatment_effect = np.nan
            t_stat = np.nan
            p_value = np.nan

        results = {
            'method': 'Mahalanobis Distance Matching',
            'n_neighbors': n_neighbors,
            'n_matched_pairs': len(matched_pairs),
            'treatment_effect': treatment_effect,
            't_statistic': t_stat,
            'p_value': p_value,
            'matched_pairs': matched_df,
            'average_distance': matched_df['mahalanobis_distance'].mean() if not matched_df.empty else np.nan
        }

        self.matching_results['mahalanobis'] = results
        return results

    def propensity_score_matching(self, data: pd.DataFrame, treatment_col: str,
                                outcome_col: str, covariate_cols: List[str],
                                caliper: Optional[float] = None, method: str = 'logit') -> Dict[str, Any]:
        """
        Propensity score matching.

        Parameters:
        -----------
        data : pd.DataFrame
            Dataset with treatment, outcome, and covariates
        treatment_col : str
            Name of treatment indicator column
        outcome_col : str
            Name of outcome variable column
        covariate_cols : list of str
            List of covariate column names
        caliper : float, optional
            Maximum distance for matching
        method : str, default 'logit'
            Method for propensity score estimation ('logit', 'rf')

        Returns:
        --------
        dict
            Propensity score matching results
        """
        if not HAS_SKLEARN:
            raise ImportError("Scikit-learn required for propensity score matching")

        # Estimate propensity scores
        X = data[covariate_cols]
        y = data[treatment_col]

        if method == 'logit':
            ps_model = LogisticRegression(random_state=42, max_iter=1000)
        elif method == 'rf':
            ps_model = RandomForestClassifier(random_state=42, n_estimators=100)
        else:
            raise ValueError("Method must be 'logit' or 'rf'")

        ps_model.fit(X, y)
        propensity_scores = ps_model.predict_proba(X)[:, 1]
        self.propensity_scores = propensity_scores

        # Add propensity scores to data
        data_with_ps = data.copy()
        data_with_ps['propensity_score'] = propensity_scores

        # Separate treated and control
        treated = data_with_ps[data_with_ps[treatment_col] == 1].copy()
        control = data_with_ps[data_with_ps[treatment_col] == 0].copy()

        # Match on propensity scores
        matched_pairs = []
        used_controls = set()

        for treated_idx in treated.index:
            treated_ps = treated.loc[treated_idx, 'propensity_score']

            best_match = None
            best_distance = float('inf')

            for control_idx in control.index:
                if control_idx in used_controls:
                    continue

                control_ps = control.loc[control_idx, 'propensity_score']
                distance = abs(treated_ps - control_ps)

                if caliper is None or distance <= caliper:
                    if distance < best_distance:
                        best_distance = distance
                        best_match = control_idx

            if best_match is not None:
                matched_pairs.append({
                    'treated_unit': treated_idx,
                    'control_unit': best_match,
                    'ps_distance': best_distance,
                    'treated_ps': treated_ps,
                    'control_ps': control.loc[best_match, 'propensity_score'],
                    'treated_outcome': treated.loc[treated_idx, outcome_col],
                    'control_outcome': control.loc[best_match, outcome_col]
                })

                used_controls.add(best_match)

        matched_df = pd.DataFrame(matched_pairs)

        # Calculate treatment effect
        if not matched_df.empty:
            treatment_effect = matched_df['treated_outcome'].mean() - matched_df['control_outcome'].mean()
            t_stat, p_value = stats.ttest_rel(matched_df['treated_outcome'], matched_df['control_outcome'])
        else:
            treatment_effect = np.nan
            t_stat = np.nan
            p_value = np.nan

        results = {
            'method': 'Propensity Score Matching',
            'ps_estimation_method': method,
            'caliper': caliper,
            'n_matched_pairs': len(matched_pairs),
            'treatment_effect': treatment_effect,
            't_statistic': t_stat,
            'p_value': p_value,
            'matched_pairs': matched_df,
            'average_ps_distance': matched_df['ps_distance'].mean() if not matched_df.empty else np.nan,
            'propensity_score_model': ps_model
        }

        self.matching_results['propensity_score'] = results
        return results

    def assess_balance(self, data: pd.DataFrame, treatment_col: str,
                      covariate_cols: List[str], matched_pairs: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess covariate balance after matching.

        Parameters:
        -----------
        data : pd.DataFrame
            Original dataset
        treatment_col : str
            Treatment indicator column
        covariate_cols : list of str
            Covariate columns
        matched_pairs : pd.DataFrame
            Matched pairs from matching procedure

        Returns:
        --------
        dict
            Balance assessment results
        """
        balance_results = {}

        # Before matching balance
        treated_before = data[data[treatment_col] == 1][covariate_cols]
        control_before = data[data[treatment_col] == 0][covariate_cols]

        # After matching balance
        treated_after_idx = matched_pairs['treated_unit']
        control_after_idx = matched_pairs['control_unit']

        treated_after = data.loc[treated_after_idx, covariate_cols]
        control_after = data.loc[control_after_idx, covariate_cols]

        for covar in covariate_cols:
            # Standardized mean differences
            smd_before = self._standardized_mean_difference(
                treated_before[covar], control_before[covar]
            )
            smd_after = self._standardized_mean_difference(
                treated_after[covar], control_after[covar]
            )

            # T-tests
            t_before, p_before = stats.ttest_ind(
                treated_before[covar], control_before[covar]
            )
            t_after, p_after = stats.ttest_ind(
                treated_after[covar], control_after[covar]
            )

            balance_results[covar] = {
                'smd_before': smd_before,
                'smd_after': smd_after,
                'smd_reduction': abs(smd_before) - abs(smd_after),
                't_stat_before': t_before,
                'p_value_before': p_before,
                't_stat_after': t_after,
                'p_value_after': p_after
            }

        return balance_results

    def _standardized_mean_difference(self, group1: pd.Series, group2: pd.Series) -> float:
        """
        Calculate standardized mean difference between two groups.
        """
        mean1, mean2 = group1.mean(), group2.mean()
        var1, var2 = group1.var(), group2.var()
        pooled_std = np.sqrt((var1 + var2) / 2)

        return (mean1 - mean2) / pooled_std if pooled_std != 0 else 0

class InstrumentalVariables:
    """
    Instrumental Variables and Two-Stage Least Squares regression with instrument testing.
    """

    def __init__(self):
        """Initialize the IV analyzer."""
        self.results = {}
        self.first_stage_results = {}
        self.iv_results = {}

    def two_stage_least_squares(self, y: pd.Series, X_exog: pd.DataFrame,
                               X_endog: pd.DataFrame, instruments: pd.DataFrame,
                               add_constant: bool = True) -> Dict[str, Any]:
        """
        Two-Stage Least Squares (2SLS) estimation.

        Parameters:
        -----------
        y : pd.Series
            Dependent variable
        X_exog : pd.DataFrame
            Exogenous explanatory variables
        X_endog : pd.DataFrame
            Endogenous explanatory variables
        instruments : pd.DataFrame
            Instrumental variables
        add_constant : bool, default True
            Whether to add intercept

        Returns:
        --------
        dict
            2SLS estimation results
        """
        if not HAS_STATSMODELS:
            raise ImportError("Statsmodels required for 2SLS")

        # Combine all exogenous variables (including instruments)
        X_all_exog = pd.concat([X_exog, instruments], axis=1)
        if add_constant:
            X_all_exog = sm.add_constant(X_all_exog)

        # First stage: regress endogenous variables on all exogenous variables
        first_stage_results = {}
        X_endog_fitted = pd.DataFrame(index=X_endog.index)

        for endog_var in X_endog.columns:
            first_stage_model = sm.OLS(X_endog[endog_var], X_all_exog)
            first_stage_fit = first_stage_model.fit()
            first_stage_results[endog_var] = first_stage_fit
            X_endog_fitted[endog_var] = first_stage_fit.fittedvalues

        # Second stage: regress y on exogenous variables and fitted endogenous variables
        X_exog_final = X_exog.copy()
        if add_constant:
            X_exog_final = sm.add_constant(X_exog_final)

        X_second_stage = pd.concat([X_exog_final, X_endog_fitted], axis=1)
        second_stage_model = sm.OLS(y, X_second_stage)
        second_stage_results = second_stage_model.fit()

        # Calculate correct standard errors (need to account for first stage uncertainty)
        # For now, using basic 2SLS implementation
        # In practice, would use statsmodels IV2SLS for correct standard errors

        results = {
            'method': 'Two-Stage Least Squares (2SLS)',
            'first_stage_results': first_stage_results,
            'second_stage_results': second_stage_results,
            'coefficients': second_stage_results.params.to_dict(),
            'std_errors': second_stage_results.bse.to_dict(),
            't_values': second_stage_results.tvalues.to_dict(),
            'p_values': second_stage_results.pvalues.to_dict(),
            'r_squared': second_stage_results.rsquared,
            'endogenous_variables': list(X_endog.columns),
            'instruments': list(instruments.columns),
            'n_observations': len(y)
        }

        self.first_stage_results = first_stage_results
        self.iv_results['2sls'] = results
        return results

    def test_instrument_strength(self, first_stage_results: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Test strength of instruments using F-statistics from first stage.

        Parameters:
        -----------
        first_stage_results : dict, optional
            First stage regression results. If None, uses stored results.

        Returns:
        --------
        dict
            Instrument strength test results
        """
        if first_stage_results is None:
            first_stage_results = self.first_stage_results

        if not first_stage_results:
            raise ValueError("No first stage results available")

        strength_results = {}

        for endog_var, results in first_stage_results.items():
            f_stat = results.fvalue
            f_pvalue = results.f_pvalue

            # Rule of thumb: F > 10 indicates strong instruments
            strength_assessment = "Strong" if f_stat > 10 else ("Moderate" if f_stat > 5 else "Weak")

            strength_results[endog_var] = {
                'f_statistic': f_stat,
                'f_pvalue': f_pvalue,
                'strength_assessment': strength_assessment,
                'r_squared': results.rsquared,
                'partial_r_squared': self._calculate_partial_r_squared(results)
            }

        return {
            'instrument_strength_results': strength_results,
            'overall_assessment': self._overall_strength_assessment(strength_results)
        }

    def test_instrument_validity(self, y: pd.Series, X_exog: pd.DataFrame,
                               X_endog: pd.DataFrame, instruments: pd.DataFrame,
                               overid_test: bool = True) -> Dict[str, Any]:
        """
        Test instrument validity using overidentification tests.

        Parameters:
        -----------
        y : pd.Series
            Dependent variable
        X_exog : pd.DataFrame
            Exogenous variables
        X_endog : pd.DataFrame
            Endogenous variables
        instruments : pd.DataFrame
            Instrumental variables
        overid_test : bool, default True
            Whether to perform overidentification test

        Returns:
        --------
        dict
            Instrument validity test results
        """
        validity_results = {}

        # Check identification condition
        n_endogenous = X_endog.shape[1]
        n_instruments = instruments.shape[1]
        n_excluded = n_instruments  # Assuming all instruments are excluded

        identification_status = {
            'n_endogenous': n_endogenous,
            'n_instruments': n_instruments,
            'n_excluded_instruments': n_excluded,
            'identification': 'Just identified' if n_excluded == n_endogenous else
                           ('Overidentified' if n_excluded > n_endogenous else 'Underidentified')
        }

        validity_results['identification'] = identification_status

        # Overidentification test (Hansen J-test) - simplified version
        if overid_test and n_excluded > n_endogenous:
            # Run 2SLS first
            tsls_results = self.two_stage_least_squares(y, X_exog, X_endog, instruments)

            # Get 2SLS residuals (would need proper implementation)
            # This is a simplified placeholder
            validity_results['overid_test'] = {
                'test_name': 'Hansen J-test (simplified)',
                'note': 'Full implementation requires specialized IV regression package'
            }

        return validity_results

    def _calculate_partial_r_squared(self, first_stage_results) -> float:
        """
        Calculate partial R-squared for instrument strength.
        Simplified implementation.
        """
        return first_stage_results.rsquared  # Placeholder

    def _overall_strength_assessment(self, strength_results: Dict) -> str:
        """
        Provide overall assessment of instrument strength.
        """
        assessments = [result['strength_assessment'] for result in strength_results.values()]

        if all(a == 'Strong' for a in assessments):
            return 'All instruments are strong'
        elif any(a == 'Weak' for a in assessments):
            return 'Some weak instruments detected'
        else:
            return 'Instruments have moderate to strong strength'

class PanelDataAnalyzer:
    """
    Panel data analysis including fixed effects and random effects models.
    """

    def __init__(self):
        """Initialize the panel data analyzer."""
        self.results = {}

    def fixed_effects_regression(self, data: pd.DataFrame, dependent_var: str,
                               independent_vars: List[str], entity_col: str,
                               time_col: Optional[str] = None,
                               entity_effects: bool = True,
                               time_effects: bool = False) -> Dict[str, Any]:
        """
        Fixed effects regression for panel data.

        Parameters:
        -----------
        data : pd.DataFrame
            Panel dataset
        dependent_var : str
            Dependent variable column name
        independent_vars : list of str
            Independent variable column names
        entity_col : str
            Entity identifier column
        time_col : str, optional
            Time identifier column
        entity_effects : bool, default True
            Include entity fixed effects
        time_effects : bool, default False
            Include time fixed effects

        Returns:
        --------
        dict
            Fixed effects regression results
        """
        if not HAS_STATSMODELS:
            raise ImportError("Statsmodels required for fixed effects regression")

        panel_data = data.copy()

        # Create dummy variables for fixed effects
        formula_parts = [dependent_var, '~'] + independent_vars

        if entity_effects:
            # Add entity dummies (drop one for identification)
            entity_dummies = pd.get_dummies(panel_data[entity_col], prefix='entity', drop_first=True)
            panel_data = pd.concat([panel_data, entity_dummies], axis=1)
            formula_parts.extend(entity_dummies.columns.tolist())

        if time_effects and time_col:
            # Add time dummies (drop one for identification)
            time_dummies = pd.get_dummies(panel_data[time_col], prefix='time', drop_first=True)
            panel_data = pd.concat([panel_data, time_dummies], axis=1)
            formula_parts.extend(time_dummies.columns.tolist())

        # Build formula
        formula = ' + '.join(formula_parts)

        # Estimate model
        model = smf.ols(formula, data=panel_data)
        results = model.fit()

        # Extract main coefficients (exclude fixed effects)
        main_coefficients = {}
        for var in independent_vars:
            if var in results.params.index:
                main_coefficients[var] = {
                    'coefficient': results.params[var],
                    'std_error': results.bse[var],
                    't_value': results.tvalues[var],
                    'p_value': results.pvalues[var]
                }

        fe_results = {
            'method': 'Fixed Effects Regression',
            'entity_effects': entity_effects,
            'time_effects': time_effects,
            'main_coefficients': main_coefficients,
            'full_results': results,
            'r_squared': results.rsquared,
            'adj_r_squared': results.rsquared_adj,
            'f_statistic': results.fvalue,
            'f_pvalue': results.f_pvalue,
            'n_observations': results.nobs,
            'n_entities': panel_data[entity_col].nunique(),
            'n_time_periods': panel_data[time_col].nunique() if time_col else None
        }

        self.results['fixed_effects'] = fe_results
        return fe_results

    def within_transformation(self, data: pd.DataFrame, variables: List[str],
                            entity_col: str) -> pd.DataFrame:
        """
        Apply within transformation (entity demeaning) to variables.

        Parameters:
        -----------
        data : pd.DataFrame
            Panel dataset
        variables : list of str
            Variables to transform
        entity_col : str
            Entity identifier column

        Returns:
        --------
        pd.DataFrame
            Data with within-transformed variables
        """
        transformed_data = data.copy()

        for var in variables:
            # Calculate entity means
            entity_means = data.groupby(entity_col)[var].mean()

            # Subtract entity means
            transformed_data[f"{var}_within"] = (
                data[var] - data[entity_col].map(entity_means)
            )

        return transformed_data

    def hausman_test_fe_re(self, data: pd.DataFrame, dependent_var: str,
                          independent_vars: List[str], entity_col: str) -> Dict[str, Any]:
        """
        Hausman test to choose between fixed effects and random effects.
        Simplified implementation.

        Parameters:
        -----------
        data : pd.DataFrame
            Panel dataset
        dependent_var : str
            Dependent variable
        independent_vars : list of str
            Independent variables
        entity_col : str
            Entity identifier

        Returns:
        --------
        dict
            Hausman test results
        """
        # Fixed effects estimation
        fe_results = self.fixed_effects_regression(
            data, dependent_var, independent_vars, entity_col
        )

        # Random effects estimation (simplified - using pooled OLS as approximation)
        y = data[dependent_var]
        X = data[independent_vars]
        X_with_const = sm.add_constant(X)

        pooled_model = sm.OLS(y, X_with_const)
        re_results = pooled_model.fit()  # This is a simplification

        # Extract coefficients for comparison
        fe_coefs = [fe_results['main_coefficients'][var]['coefficient']
                   for var in independent_vars
                   if var in fe_results['main_coefficients']]

        re_coefs = [re_results.params[var] for var in independent_vars
                   if var in re_results.params.index]

        # Simplified Hausman test (proper implementation needs covariance matrices)
        coef_diff = np.array(fe_coefs) - np.array(re_coefs[:len(fe_coefs)])
        hausman_stat = np.sum(coef_diff**2)  # Simplified

        return {
            'test_name': 'Hausman Test (FE vs RE)',
            'hausman_statistic': hausman_stat,
            'note': 'Simplified implementation. Use specialized panel data packages for full test.',
            'fe_coefficients': fe_coefs,
            're_coefficients': re_coefs[:len(fe_coefs)],
            'coefficient_differences': coef_diff.tolist()
        }

class RegressionDiscontinuity:
    """
    Regression Discontinuity Design analysis with bandwidth selection.
    """

    def __init__(self):
        """Initialize the RDD analyzer."""
        self.results = {}

    def sharp_rdd(self, data: pd.DataFrame, outcome_col: str, running_var_col: str,
                 cutoff: float, bandwidth: Optional[float] = None,
                 polynomial_order: int = 1, kernel: str = 'triangular') -> Dict[str, Any]:
        """
        Sharp Regression Discontinuity Design.

        Parameters:
        -----------
        data : pd.DataFrame
            Dataset with outcome and running variable
        outcome_col : str
            Outcome variable column name
        running_var_col : str
            Running variable (assignment variable) column name
        cutoff : float
            Cutoff value for treatment assignment
        bandwidth : float, optional
            Bandwidth around cutoff. If None, optimal bandwidth is selected.
        polynomial_order : int, default 1
            Order of polynomial fit
        kernel : str, default 'triangular'
            Kernel for weighting ('triangular', 'uniform')

        Returns:
        --------
        dict
            Sharp RDD results
        """
        rdd_data = data.copy()

        # Center running variable at cutoff
        rdd_data['running_centered'] = rdd_data[running_var_col] - cutoff

        # Create treatment indicator
        rdd_data['treatment'] = (rdd_data[running_var_col] >= cutoff).astype(int)

        # Select bandwidth if not provided
        if bandwidth is None:
            bandwidth = self._optimal_bandwidth_ik(rdd_data, outcome_col, 'running_centered')

        # Filter data within bandwidth
        rdd_subset = rdd_data[
            np.abs(rdd_data['running_centered']) <= bandwidth
        ].copy()

        if len(rdd_subset) == 0:
            raise ValueError("No observations within specified bandwidth")

        # Create polynomial terms
        for order in range(1, polynomial_order + 1):
            rdd_subset[f'running_poly_{order}'] = rdd_subset['running_centered'] ** order
            rdd_subset[f'treatment_running_poly_{order}'] = (
                rdd_subset['treatment'] * rdd_subset[f'running_poly_{order}']
            )

        # Create weights if using kernel
        if kernel == 'triangular':
            rdd_subset['weights'] = 1 - np.abs(rdd_subset['running_centered']) / bandwidth
        elif kernel == 'uniform':
            rdd_subset['weights'] = 1.0
        else:
            raise ValueError("Kernel must be 'triangular' or 'uniform'")

        # Build regression formula
        poly_terms = [f'running_poly_{i}' for i in range(1, polynomial_order + 1)]
        interaction_terms = [f'treatment_running_poly_{i}' for i in range(1, polynomial_order + 1)]

        formula_terms = ['treatment'] + poly_terms + interaction_terms
        formula = f"{outcome_col} ~ {' + '.join(formula_terms)}"

        # Estimate RDD model
        if HAS_STATSMODELS:
            model = smf.wls(formula, data=rdd_subset, weights=rdd_subset['weights'])
            results = model.fit()

            # Treatment effect is the coefficient on treatment dummy
            treatment_effect = results.params['treatment']
            treatment_se = results.bse['treatment']
            treatment_pvalue = results.pvalues['treatment']

        else:
            # Fallback: simple difference in means at cutoff
            left_mean = rdd_subset[rdd_subset['treatment'] == 0][outcome_col].mean()
            right_mean = rdd_subset[rdd_subset['treatment'] == 1][outcome_col].mean()
            treatment_effect = right_mean - left_mean
            treatment_se = None
            treatment_pvalue = None
            results = None

        rdd_results = {
            'method': 'Sharp Regression Discontinuity',
            'treatment_effect': treatment_effect,
            'standard_error': treatment_se,
            'p_value': treatment_pvalue,
            'bandwidth': bandwidth,
            'cutoff': cutoff,
            'polynomial_order': polynomial_order,
            'kernel': kernel,
            'n_observations': len(rdd_subset),
            'n_treated': rdd_subset['treatment'].sum(),
            'n_control': len(rdd_subset) - rdd_subset['treatment'].sum(),
            'regression_results': results
        }

        self.results['sharp_rdd'] = rdd_results
        return rdd_results

    def _optimal_bandwidth_ik(self, data: pd.DataFrame, outcome_col: str,
                             running_var_col: str) -> float:
        """
        Imbens-Kalyanaraman optimal bandwidth selection (simplified).
        """
        # Simplified implementation - in practice would use more sophisticated methods
        running_var = data[running_var_col]

        # Rule of thumb: use standard deviation of running variable
        bandwidth = running_var.std() * 0.5

        return bandwidth

    def plot_rdd(self, data: pd.DataFrame, outcome_col: str, running_var_col: str,
                cutoff: float, bandwidth: float, bins: int = 50) -> None:
        """
        Plot RDD visualization.

        Parameters:
        -----------
        data : pd.DataFrame
            Dataset
        outcome_col : str
            Outcome variable
        running_var_col : str
            Running variable
        cutoff : float
            Cutoff value
        bandwidth : float
            Bandwidth for analysis
        bins : int, default 50
            Number of bins for local averages
        """
        # Filter data within bandwidth
        plot_data = data[
            np.abs(data[running_var_col] - cutoff) <= bandwidth
        ].copy()

        if len(plot_data) == 0:
            print("No data within bandwidth for plotting")
            return

        plt.figure(figsize=(10, 6))

        # Create bins for local averages
        plot_data['bin'] = pd.cut(plot_data[running_var_col], bins=bins)
        bin_means = plot_data.groupby('bin').agg({
            running_var_col: 'mean',
            outcome_col: 'mean'
        }).dropna()

        # Separate left and right of cutoff
        left_bins = bin_means[bin_means[running_var_col] < cutoff]
        right_bins = bin_means[bin_means[running_var_col] >= cutoff]

        # Plot local averages
        plt.scatter(left_bins[running_var_col], left_bins[outcome_col],
                   alpha=0.7, color='blue', label='Control')
        plt.scatter(right_bins[running_var_col], right_bins[outcome_col],
                   alpha=0.7, color='red', label='Treatment')

        # Add cutoff line
        plt.axvline(x=cutoff, color='black', linestyle='--', alpha=0.8, label='Cutoff')

        plt.xlabel(running_var_col)
        plt.ylabel(outcome_col)
        plt.title('Regression Discontinuity Design')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

class TimeSeriesEconometrics:
    """
    Time series econometric methods including autocorrelation tests and Granger causality.
    """

    def __init__(self):
        """Initialize the time series econometrics analyzer."""
        self.results = {}

    def durbin_watson_test(self, residuals: pd.Series) -> Dict[str, Any]:
        """
        Durbin-Watson test for autocorrelation in residuals.

        Parameters:
        -----------
        residuals : pd.Series
            Regression residuals

        Returns:
        --------
        dict
            Durbin-Watson test results
        """
        if not HAS_STATSMODELS:
            raise ImportError("Statsmodels required for Durbin-Watson test")

        dw_statistic = durbin_watson(residuals)

        # Interpretation guidelines
        if dw_statistic < 1.5:
            interpretation = "Positive autocorrelation likely"
        elif dw_statistic > 2.5:
            interpretation = "Negative autocorrelation likely"
        else:
            interpretation = "No strong evidence of autocorrelation"

        results = {
            'test_name': 'Durbin-Watson Test',
            'statistic': dw_statistic,
            'interpretation': interpretation,
            'note': 'DW ≈ 2 indicates no autocorrelation, DW < 2 suggests positive autocorrelation'
        }

        self.results['durbin_watson'] = results
        return results

    def breusch_godfrey_test(self, model_results, lags: int = 1) -> Dict[str, Any]:
        """
        Breusch-Godfrey test for serial correlation.

        Parameters:
        -----------
        model_results
            Fitted regression model results
        lags : int, default 1
            Number of lags to test

        Returns:
        --------
        dict
            Breusch-Godfrey test results
        """
        if not HAS_STATSMODELS:
            raise ImportError("Statsmodels required for Breusch-Godfrey test")

        try:
            bg_test = acorr_breusch_godfrey(model_results, nlags=lags)

            results = {
                'test_name': 'Breusch-Godfrey Test',
                'lm_statistic': bg_test[0],
                'p_value': bg_test[1],
                'f_statistic': bg_test[2],
                'f_p_value': bg_test[3],
                'lags_tested': lags,
                'conclusion': 'No serial correlation' if bg_test[1] > 0.05 else 'Serial correlation detected'
            }
        except Exception as e:
            results = {
                'test_name': 'Breusch-Godfrey Test',
                'error': f"Could not perform test: {str(e)}"
            }

        self.results['breusch_godfrey'] = results
        return results

    def granger_causality_test(self, data: pd.DataFrame, caused_var: str,
                              causing_var: str, max_lags: int = 4,
                              test: str = 'ssr_ftest') -> Dict[str, Any]:
        """
        Granger causality test.

        Parameters:
        -----------
        data : pd.DataFrame
            Time series data
        caused_var : str
            Variable being caused (dependent)
        causing_var : str
            Variable doing the causing
        max_lags : int, default 4
            Maximum number of lags to test
        test : str, default 'ssr_ftest'
            Test statistic to use

        Returns:
        --------
        dict
            Granger causality test results
        """
        if not HAS_STATSMODELS:
            raise ImportError("Statsmodels required for Granger causality test")

        # Prepare data for Granger test
        test_data = data[[caused_var, causing_var]].dropna()

        try:
            # Perform Granger causality test
            gc_results = grangercausalitytests(
                test_data[[caused_var, causing_var]],
                maxlag=max_lags,
                verbose=False
            )

            # Extract results for each lag
            lag_results = {}
            for lag in range(1, max_lags + 1):
                if lag in gc_results:
                    test_result = gc_results[lag][0][test]
                    lag_results[lag] = {
                        'f_statistic': test_result[0],
                        'p_value': test_result[1],
                        'conclusion': 'Granger causes' if test_result[1] < 0.05 else 'Does not Granger cause'
                    }

            # Find best lag (lowest p-value)
            if lag_results:
                best_lag = min(lag_results.keys(), key=lambda x: lag_results[x]['p_value'])
                best_result = lag_results[best_lag]
            else:
                best_lag = None
                best_result = None

            results = {
                'test_name': 'Granger Causality Test',
                'caused_variable': caused_var,
                'causing_variable': causing_var,
                'max_lags_tested': max_lags,
                'test_statistic': test,
                'lag_results': lag_results,
                'best_lag': best_lag,
                'best_result': best_result,
                'overall_conclusion': best_result['conclusion'] if best_result else 'Inconclusive'
            }

        except Exception as e:
            results = {
                'test_name': 'Granger Causality Test',
                'error': f"Could not perform test: {str(e)}"
            }

        self.results['granger_causality'] = results
        return results

    def spurious_regression_test(self, y: pd.Series, X: pd.DataFrame,
                               test_stationarity: bool = True) -> Dict[str, Any]:
        """
        Test for spurious regression by checking stationarity and R-squared vs DW ratio.

        Parameters:
        -----------
        y : pd.Series
            Dependent variable
        X : pd.DataFrame
            Independent variables
        test_stationarity : bool, default True
            Whether to test stationarity of variables

        Returns:
        --------
        dict
            Spurious regression test results
        """
        results = {
            'test_name': 'Spurious Regression Test'
        }

        # Run basic regression
        if HAS_STATSMODELS:
            X_with_const = sm.add_constant(X)
            model = sm.OLS(y, X_with_const)
            reg_results = model.fit()

            r_squared = reg_results.rsquared
            dw_stat = durbin_watson(reg_results.resid)

            # Rule of thumb: R² > DW suggests spurious regression
            spurious_indicator = r_squared > dw_stat

            results.update({
                'r_squared': r_squared,
                'durbin_watson': dw_stat,
                'r2_vs_dw_ratio': r_squared / dw_stat if dw_stat != 0 else np.inf,
                'spurious_indicator': spurious_indicator,
                'rule_of_thumb': 'R² > DW suggests spurious regression'
            })

        # Test stationarity if requested
        if test_stationarity and HAS_STATSMODELS:
            stationarity_results = {}

            # Test dependent variable
            try:
                adf_result_y = adfuller(y.dropna())
                stationarity_results['dependent_var'] = {
                    'adf_statistic': adf_result_y[0],
                    'p_value': adf_result_y[1],
                    'stationary': adf_result_y[1] < 0.05
                }
            except:
                stationarity_results['dependent_var'] = {'error': 'Could not test stationarity'}

            # Test independent variables
            for col in X.columns:
                try:
                    adf_result_x = adfuller(X[col].dropna())
                    stationarity_results[col] = {
                        'adf_statistic': adf_result_x[0],
                        'p_value': adf_result_x[1],
                        'stationary': adf_result_x[1] < 0.05
                    }
                except:
                    stationarity_results[col] = {'error': 'Could not test stationarity'}

            results['stationarity_tests'] = stationarity_results

            # Check if any variables are non-stationary
            non_stationary_vars = [
                var for var, test_result in stationarity_results.items()
                if isinstance(test_result, dict) and not test_result.get('stationary', True)
            ]

            results['non_stationary_variables'] = non_stationary_vars
            results['spurious_risk'] = len(non_stationary_vars) > 0

        self.results['spurious_regression'] = results
        return results

def demonstrate_econometrics():
    """
    Demonstrate the usage of the econometrics module with sample data.
    """
    print("=== Econometrics Analysis Demo ===\n")

    # Create sample data
    np.random.seed(42)
    n = 200

    # Independent variables
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)
    X3 = 0.5 * X1 + np.random.normal(0, 0.5, n)  # Some multicollinearity

    # Dependent variable with some heteroskedasticity
    error = np.random.normal(0, 1 + 0.3 * np.abs(X1), n)
    y = 2 + 1.5 * X1 + 0.8 * X2 + 0.3 * X3 + error

    # Create DataFrame
    df = pd.DataFrame({
        'y': y,
        'X1': X1,
        'X2': X2,
        'X3': X3
    })

    print("1. OLS Regression Analysis:")
    ols = OLSAnalyzer()
    X = df[['X1', 'X2', 'X3']]
    y_series = df['y']

    # Fit model
    ols.fit(X, y_series)
    print(f"   R-squared: {ols.results.rsquared:.3f}")
    print(f"   F-statistic: {ols.results.fvalue:.3f}")

    print("\n2. Testing OLS Assumptions:")
    # Test multicollinearity
    multicollinearity_results = ols.test_multicollinearity()
    print("   VIF Analysis:")
    for vif_info in multicollinearity_results['vif_analysis']:
        print(f"     {vif_info['Variable']}: VIF = {vif_info['VIF']:.2f} ({vif_info['Status']})")

    # Test heteroskedasticity
    hetero_results = ols.test_homoskedasticity(plot=False)
    if 'breusch_pagan' in hetero_results:
        bp_result = hetero_results['breusch_pagan']
        if 'p_value' in bp_result:
            print(f"   Breusch-Pagan test p-value: {bp_result['p_value']:.3f} ({bp_result['conclusion']})")

    print("\n3. Outlier Analysis:")
    outlier_results = ols.analyze_outliers_leverage(plot=False)
    print(f"   High Cook's distance observations: {len(outlier_results['cooks_distance']['outliers'])}")
    print(f"   High leverage observations: {len(outlier_results['leverage']['high_leverage_points'])}")

    print("\n4. Endogeneity Testing:")
    # Create simple instrument (for demo)
    instrument = np.random.normal(0, 1, n)
    df['instrument'] = instrument

    endogeneity = EndogeneityTester()
    try:
        wu_hausman = endogeneity.wu_hausman_test(
            df['y'], df[['X1']], df[['X2']], df[['instrument']]
        )
        print(f"   Wu-Hausman test p-value: {wu_hausman['p_value']:.3f} ({wu_hausman['conclusion']})")
    except Exception as e:
        print(f"   Wu-Hausman test failed: {str(e)}")

    print("\n5. Matching Methods (simulated treatment data):")
    # Create treatment assignment
    df['treatment'] = (df['X1'] + np.random.normal(0, 0.5, n) > 0).astype(int)
    df['outcome'] = df['y'] + 2 * df['treatment']  # Add treatment effect

    matching = MatchingMethods()
    try:
        ps_matching = matching.propensity_score_matching(
            df, 'treatment', 'outcome', ['X1', 'X2', 'X3']
        )
        print(f"   Propensity score matching treatment effect: {ps_matching['treatment_effect']:.3f}")
        print(f"   Number of matched pairs: {ps_matching['n_matched_pairs']}")
    except Exception as e:
        print(f"   Propensity score matching failed: {str(e)}")

    print("\n6. Instrumental Variables:")
    iv_analyzer = InstrumentalVariables()
    try:
        iv_results = iv_analyzer.two_stage_least_squares(
            df['y'], df[['X1']], df[['X2']], df[['instrument']]
        )
        print(f"   2SLS estimation completed")
        strength_test = iv_analyzer.test_instrument_strength()
        for var, result in strength_test['instrument_strength_results'].items():
            print(f"   Instrument strength for {var}: {result['strength_assessment']} (F = {result['f_statistic']:.3f})")
    except Exception as e:
        print(f"   IV analysis failed: {str(e)}")

    print("\n7. Time Series Tests:")
    ts_econometrics = TimeSeriesEconometrics()

    # Durbin-Watson test
    dw_result = ts_econometrics.durbin_watson_test(ols.residuals)
    print(f"   Durbin-Watson statistic: {dw_result['statistic']:.3f} ({dw_result['interpretation']})")

    # Create time series data for Granger causality
    np.random.seed(456)
    t = 100
    ts_data = pd.DataFrame({
        'x': np.random.normal(0, 1, t),
        'y': np.random.normal(0, 1, t)
    })
    # Add some lagged relationship
    for i in range(1, t):
        ts_data.loc[i, 'y'] += 0.3 * ts_data.loc[i-1, 'x']

    try:
        granger_result = ts_econometrics.granger_causality_test(ts_data, 'y', 'x', max_lags=2)
        print(f"   Granger causality (x -> y): {granger_result['overall_conclusion']}")
    except Exception as e:
        print(f"   Granger causality test failed: {str(e)}")

    print("\n8. Panel Data Analysis (simulated data):")
    # Create panel data
    np.random.seed(789)
    n_entities = 20
    n_periods = 5
    panel_data = []

    for entity in range(n_entities):
        entity_effect = np.random.normal(0, 1)
        for period in range(n_periods):
            x_val = np.random.normal(0, 1)
            y_val = 2 + entity_effect + 0.5 * x_val + np.random.normal(0, 0.5)

            panel_data.append({
                'entity': entity,
                'period': period,
                'x': x_val,
                'y': y_val
            })

    panel_df = pd.DataFrame(panel_data)

    panel_analyzer = PanelDataAnalyzer()
    try:
        fe_results = panel_analyzer.fixed_effects_regression(
            panel_df, 'y', ['x'], 'entity'
        )
        print(f"   Fixed effects R-squared: {fe_results['r_squared']:.3f}")
        print(f"   Number of entities: {fe_results['n_entities']}")
    except Exception as e:
        print(f"   Panel data analysis failed: {str(e)}")

    print("\n9. Robust Regression:")
    robust = RobustRegression()
    huber_results = robust.huber_regression(X, y_series)
    print(f"   Huber regression R-squared: {huber_results['r_squared']:.3f}")
    print(f"   Number of outliers detected: {huber_results['outliers'].sum()}")

    print("\n10. Regularized Regression:")
    regularized = RegularizedRegression()
    ridge_results = regularized.ridge_regression(X, y_series)
    print(f"   Ridge regression optimal alpha: {ridge_results['alpha']:.3f}")
    print(f"   Ridge regression R-squared: {ridge_results['r_squared']:.3f}")

    print("\n11. Non-parametric Regression (univariate example):")
    # Use X1 for univariate example
    nonparam = NonParametricRegression()
    loess_results = nonparam.loess_regression(df['X1'], df['y'])
    print(f"   LOESS R-squared: {loess_results['r_squared']:.3f}")

    print("\n12. Difference-in-Differences (simulated data):")
    # Create DiD sample data
    np.random.seed(123)
    n_units = 50

    did_data = []
    for unit in range(n_units):
        treated = 1 if unit >= n_units // 2 else 0
        for period in ['pre', 'post']:
            outcome = 10 + 2 * treated + (3 if period == 'post' else 0) + \
                     (5 if treated and period == 'post' else 0) + np.random.normal(0, 1)

            did_data.append({
                'unit': unit,
                'period': period,
                'treated': treated,
                'outcome': outcome
            })

    did_df = pd.DataFrame(did_data)

    did = DifferenceInDifferences()
    did_results = did.basic_diff_in_diff(did_df, 'outcome', 'treated', 'period', 'pre', 'post')
    print(f"   DiD treatment effect: {did_results['treatment_effect_estimate']:.3f}")
    if did_results['p_value']:
        print(f"   P-value: {did_results['p_value']:.3f}")

    print("\n13. Regression Discontinuity (simulated data):")
    # Create RDD data
    np.random.seed(999)
    rdd_n = 200
    running_var = np.random.uniform(-2, 2, rdd_n)
    cutoff = 0
    treatment_rdd = (running_var >= cutoff).astype(int)
    outcome_rdd = 2 + 0.5 * running_var + 3 * treatment_rdd + np.random.normal(0, 1, rdd_n)

    rdd_data = pd.DataFrame({
        'outcome': outcome_rdd,
        'running_var': running_var
    })

    rdd = RegressionDiscontinuity()
    try:
        rdd_results = rdd.sharp_rdd(rdd_data, 'outcome', 'running_var', cutoff=0)
        print(f"   RDD treatment effect: {rdd_results['treatment_effect']:.3f}")
        print(f"   Bandwidth used: {rdd_results['bandwidth']:.3f}")
    except Exception as e:
        print(f"   RDD analysis failed: {str(e)}")

    print("\nDemo completed successfully!")

if __name__ == "__main__":
    demonstrate_econometrics()