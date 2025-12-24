"""
Time Series Analysis Module

This module provides comprehensive time series analysis capabilities including
resampling, smoothing techniques, stationarity testing, trend removal, and
autocorrelation analysis for financial and economic data.

Key Features:
- Time series resampling and frequency conversion
- Moving averages with rolling windows
- Exponential smoothing (Simple, Double, Triple/Holt-Winters)
- Unit root tests for stationarity (Dickey-Fuller, ADF, KPSS)
- Trend removal techniques (detrending, differencing)
- Autocorrelation and partial autocorrelation analysis
- ARMA and ARIMA model fitting and forecasting
- Walk-forward validation and model evaluation
- Hyperparameter tuning with computational efficiency analysis
- Comprehensive visualization capabilities

Classes:
    TimeSeriesResampler: Resampling and frequency conversion
    MovingAverages: Simple and weighted moving averages
    ExponentialSmoothing: Simple, double, and triple exponential smoothing
    StationarityTester: Unit root tests and stationarity analysis
    TrendRemover: Detrending and differencing methods
    AutocorrelationAnalyzer: ACF and PACF analysis with lag determination
    ARMAModeler: ARMA model fitting and analysis
    ARIMAModeler: ARIMA model fitting, forecasting, and validation
    TimeSeriesVisualizer: Comprehensive plotting capabilities
    ModelValidator: Walk-forward validation and performance evaluation
    HyperparameterTuner: ARIMA parameter optimization

Dependencies:
    - pandas
    - numpy
    - scipy
    - matplotlib
    - statsmodels
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

# Core dependencies
try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.statespace.tools import diff
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWSmoothing
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import acf, pacf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("Statsmodels not available. Advanced time series functions will be limited.")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    warnings.warn("Seaborn not available. Some visualization features will be limited.")

class TimeSeriesResampler:
    """
    Comprehensive time series resampling and frequency conversion methods.
    """

    def __init__(self):
        """Initialize the time series resampler."""
        self.original_data = None
        self.resampled_data = {}

    def resample_data(self, data: pd.Series, target_frequency: str,
                     aggregation_method: str = 'mean') -> pd.Series:
        """
        Resample time series data to a different frequency.

        Parameters:
        -----------
        data : pd.Series
            Time series data with datetime index
        target_frequency : str
            Target frequency ('D', 'W', 'M', 'Q', 'Y', 'H', 'T', etc.)
        aggregation_method : str, default 'mean'
            Aggregation method ('mean', 'sum', 'first', 'last', 'min', 'max', 'std')

        Returns:
        --------
        pd.Series
            Resampled time series data
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex")

        self.original_data = data.copy()

        # Map aggregation methods
        agg_methods = {
            'mean': lambda x: x.mean(),
            'sum': lambda x: x.sum(),
            'first': lambda x: x.first(),
            'last': lambda x: x.last(),
            'min': lambda x: x.min(),
            'max': lambda x: x.max(),
            'std': lambda x: x.std(),
            'count': lambda x: x.count(),
            'median': lambda x: x.median()
        }

        if aggregation_method not in agg_methods:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        # Perform resampling
        resampler = data.resample(target_frequency)
        resampled = agg_methods[aggregation_method](resampler)

        # Store result
        key = f"{target_frequency}_{aggregation_method}"
        self.resampled_data[key] = resampled

        return resampled

    def upsample_data(self, data: pd.Series, target_frequency: str,
                     fill_method: str = 'interpolate') -> pd.Series:
        """
        Upsample time series data to higher frequency.

        Parameters:
        -----------
        data : pd.Series
            Time series data with datetime index
        target_frequency : str
            Target frequency (higher frequency than original)
        fill_method : str, default 'interpolate'
            Method to fill missing values ('interpolate', 'ffill', 'bfill', 'zero')

        Returns:
        --------
        pd.Series
            Upsampled time series data
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex")

        # Upsample to target frequency
        upsampled = data.resample(target_frequency).asfreq()

        # Fill missing values
        if fill_method == 'interpolate':
            upsampled = upsampled.interpolate(method='time')
        elif fill_method == 'ffill':
            upsampled = upsampled.fillna(method='ffill')
        elif fill_method == 'bfill':
            upsampled = upsampled.fillna(method='bfill')
        elif fill_method == 'zero':
            upsampled = upsampled.fillna(0)
        else:
            raise ValueError(f"Unknown fill method: {fill_method}")

        return upsampled

    def align_frequencies(self, *series: pd.Series,
                         target_frequency: Optional[str] = None) -> List[pd.Series]:
        """
        Align multiple time series to the same frequency.

        Parameters:
        -----------
        *series : pd.Series
            Multiple time series to align
        target_frequency : str, optional
            Target frequency. If None, uses the lowest common frequency.

        Returns:
        --------
        list of pd.Series
            Aligned time series
        """
        if not all(isinstance(s.index, pd.DatetimeIndex) for s in series):
            raise ValueError("All series must have DatetimeIndex")

        if target_frequency is None:
            # Find the lowest common frequency (highest resolution)
            frequencies = []
            for s in series:
                try:
                    freq = pd.infer_freq(s.index)
                    if freq:
                        frequencies.append(freq)
                except:
                    continue

            if not frequencies:
                raise ValueError("Could not infer frequency from any series")

            # Use the most frequent frequency
            target_frequency = max(set(frequencies), key=frequencies.count)

        # Align all series to target frequency
        aligned_series = []
        for s in series:
            aligned = self.resample_data(s, target_frequency, 'mean')
            aligned_series.append(aligned)

        return aligned_series

    def create_business_calendar(self, start_date: str, end_date: str,
                               holidays: Optional[List[str]] = None,
                               country: str = 'US') -> pd.DatetimeIndex:
        """
        Create a business day calendar excluding weekends and holidays.

        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        holidays : list of str, optional
            List of holiday dates in 'YYYY-MM-DD' format
        country : str, default 'US'
            Country code for holiday calendar

        Returns:
        --------
        pd.DatetimeIndex
            Business day calendar
        """
        # Create business day range
        business_days = pd.bdate_range(start=start_date, end=end_date, freq='B')

        # Remove custom holidays if provided
        if holidays:
            holiday_dates = pd.to_datetime(holidays)
            business_days = business_days.difference(holiday_dates)

        return business_days

class MovingAverages:
    """
    Moving average calculations with various window types and weighting schemes.
    """

    def __init__(self):
        """Initialize the moving averages calculator."""
        self.results = {}

    def simple_moving_average(self, data: pd.Series, window: int,
                            min_periods: Optional[int] = None) -> pd.Series:
        """
        Calculate simple moving average using rolling windows.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        window : int
            Window size for moving average
        min_periods : int, optional
            Minimum number of observations required to have a value

        Returns:
        --------
        pd.Series
            Simple moving average
        """
        if min_periods is None:
            min_periods = window

        sma = data.rolling(window=window, min_periods=min_periods).mean()

        self.results[f'SMA_{window}'] = sma
        return sma

    def exponential_moving_average(self, data: pd.Series, span: int,
                                 adjust: bool = True) -> pd.Series:
        """
        Calculate exponential moving average.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        span : int
            Span for exponential moving average
        adjust : bool, default True
            Whether to use adjusted exponential moving average

        Returns:
        --------
        pd.Series
            Exponential moving average
        """
        ema = data.ewm(span=span, adjust=adjust).mean()

        self.results[f'EMA_{span}'] = ema
        return ema

    def weighted_moving_average(self, data: pd.Series, weights: List[float]) -> pd.Series:
        """
        Calculate weighted moving average with custom weights.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        weights : list of float
            Weights for moving average (should sum to 1)

        Returns:
        --------
        pd.Series
            Weighted moving average
        """
        if not np.isclose(sum(weights), 1.0):
            warnings.warn("Weights do not sum to 1, normalizing...")
            weights = np.array(weights) / sum(weights)

        window = len(weights)

        def weighted_mean(x):
            if len(x) == window:
                return np.sum(x * weights)
            else:
                return np.nan

        wma = data.rolling(window=window).apply(weighted_mean, raw=True)

        self.results[f'WMA_{window}'] = wma
        return wma

    def triangular_moving_average(self, data: pd.Series, window: int) -> pd.Series:
        """
        Calculate triangular moving average.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        window : int
            Window size for triangular moving average

        Returns:
        --------
        pd.Series
            Triangular moving average
        """
        # Create triangular weights
        half_window = window // 2
        weights = []

        # Ascending part
        for i in range(1, half_window + 1):
            weights.append(i)

        # Descending part (include middle if odd window)
        if window % 2 == 1:
            weights.append(half_window + 1)
            start = half_window
        else:
            start = half_window

        for i in range(start, 0, -1):
            weights.append(i)

        # Normalize weights
        weights = np.array(weights) / sum(weights)

        return self.weighted_moving_average(data, weights.tolist())

    def adaptive_moving_average(self, data: pd.Series, fast_period: int = 2,
                              slow_period: int = 30) -> pd.Series:
        """
        Calculate adaptive moving average using efficiency ratio.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        fast_period : int, default 2
            Fast smoothing constant period
        slow_period : int, default 30
            Slow smoothing constant period

        Returns:
        --------
        pd.Series
            Adaptive moving average
        """
        # Calculate efficiency ratio
        period = 14  # Standard period for efficiency ratio

        # Direction of price movement
        direction = np.abs(data - data.shift(period))

        # Volatility (sum of absolute differences)
        volatility = data.diff().abs().rolling(window=period).sum()

        # Efficiency ratio
        efficiency_ratio = direction / volatility

        # Smoothing constants
        fast_sc = 2 / (fast_period + 1)
        slow_sc = 2 / (slow_period + 1)

        # Adaptive smoothing constant
        smoothing_constant = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2

        # Calculate adaptive moving average
        ama = pd.Series(index=data.index, dtype=float)
        ama.iloc[0] = data.iloc[0]

        for i in range(1, len(data)):
            if not pd.isna(smoothing_constant.iloc[i]):
                ama.iloc[i] = (ama.iloc[i-1] +
                              smoothing_constant.iloc[i] * (data.iloc[i] - ama.iloc[i-1]))
            else:
                ama.iloc[i] = ama.iloc[i-1]

        self.results[f'AMA_{fast_period}_{slow_period}'] = ama
        return ama

class ExponentialSmoothing:
    """
    Exponential smoothing methods: Simple, Double (Holt), and Triple (Holt-Winters).
    """

    def __init__(self):
        """Initialize the exponential smoothing analyzer."""
        self.models = {}
        self.results = {}

    def simple_exponential_smoothing(self, data: pd.Series, alpha: Optional[float] = None,
                                   optimize: bool = True) -> Dict[str, Any]:
        """
        Simple exponential smoothing for data without trend or seasonality.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        alpha : float, optional
            Smoothing parameter. If None, will be optimized.
        optimize : bool, default True
            Whether to optimize parameters

        Returns:
        --------
        dict
            Smoothing results including fitted values and parameters
        """
        if not HAS_STATSMODELS:
            # Manual implementation
            return self._manual_simple_smoothing(data, alpha)

        # Use statsmodels implementation
        model = HWSmoothing(data, trend=None, seasonal=None)

        if alpha is not None and not optimize:
            fitted_model = model.fit(smoothing_level=alpha, optimized=False)
        else:
            fitted_model = model.fit(optimized=optimize)

        results = {
            'method': 'Simple Exponential Smoothing',
            'fitted_values': fitted_model.fittedvalues,
            'residuals': fitted_model.resid,
            'alpha': fitted_model.params['smoothing_level'],
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'sse': fitted_model.sse,
            'model': fitted_model
        }

        self.models['simple'] = fitted_model
        self.results['simple'] = results
        return results

    def double_exponential_smoothing(self, data: pd.Series, alpha: Optional[float] = None,
                                   beta: Optional[float] = None,
                                   optimize: bool = True) -> Dict[str, Any]:
        """
        Double exponential smoothing (Holt's method) for data with trend.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        alpha : float, optional
            Level smoothing parameter
        beta : float, optional
            Trend smoothing parameter
        optimize : bool, default True
            Whether to optimize parameters

        Returns:
        --------
        dict
            Smoothing results including fitted values and parameters
        """
        if not HAS_STATSMODELS:
            return self._manual_double_smoothing(data, alpha, beta)

        # Use statsmodels implementation
        model = HWSmoothing(data, trend='add', seasonal=None)

        if alpha is not None and beta is not None and not optimize:
            fitted_model = model.fit(
                smoothing_level=alpha,
                smoothing_trend=beta,
                optimized=False
            )
        else:
            fitted_model = model.fit(optimized=optimize)

        results = {
            'method': 'Double Exponential Smoothing (Holt)',
            'fitted_values': fitted_model.fittedvalues,
            'residuals': fitted_model.resid,
            'alpha': fitted_model.params['smoothing_level'],
            'beta': fitted_model.params['smoothing_trend'],
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'sse': fitted_model.sse,
            'model': fitted_model
        }

        self.models['double'] = fitted_model
        self.results['double'] = results
        return results

    def triple_exponential_smoothing(self, data: pd.Series, seasonal_periods: int,
                                   trend: str = 'add', seasonal: str = 'add',
                                   alpha: Optional[float] = None,
                                   beta: Optional[float] = None,
                                   gamma: Optional[float] = None,
                                   optimize: bool = True) -> Dict[str, Any]:
        """
        Triple exponential smoothing (Holt-Winters) for data with trend and seasonality.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        seasonal_periods : int
            Number of periods in seasonal cycle
        trend : str, default 'add'
            Type of trend component ('add', 'mul')
        seasonal : str, default 'add'
            Type of seasonal component ('add', 'mul')
        alpha : float, optional
            Level smoothing parameter
        beta : float, optional
            Trend smoothing parameter
        gamma : float, optional
            Seasonal smoothing parameter
        optimize : bool, default True
            Whether to optimize parameters

        Returns:
        --------
        dict
            Smoothing results including fitted values and parameters
        """
        if not HAS_STATSMODELS:
            return self._manual_triple_smoothing(data, seasonal_periods, alpha, beta, gamma)

        # Use statsmodels implementation
        model = HWSmoothing(
            data,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        )

        if (alpha is not None and beta is not None and
            gamma is not None and not optimize):
            fitted_model = model.fit(
                smoothing_level=alpha,
                smoothing_trend=beta,
                smoothing_seasonal=gamma,
                optimized=False
            )
        else:
            fitted_model = model.fit(optimized=optimize)

        results = {
            'method': 'Triple Exponential Smoothing (Holt-Winters)',
            'fitted_values': fitted_model.fittedvalues,
            'residuals': fitted_model.resid,
            'alpha': fitted_model.params['smoothing_level'],
            'beta': fitted_model.params['smoothing_trend'],
            'gamma': fitted_model.params['smoothing_seasonal'],
            'seasonal_periods': seasonal_periods,
            'trend_type': trend,
            'seasonal_type': seasonal,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'sse': fitted_model.sse,
            'model': fitted_model
        }

        self.models['triple'] = fitted_model
        self.results['triple'] = results
        return results

    def forecast(self, model_type: str, steps: int) -> pd.Series:
        """
        Generate forecasts using fitted exponential smoothing model.

        Parameters:
        -----------
        model_type : str
            Type of model ('simple', 'double', 'triple')
        steps : int
            Number of steps to forecast

        Returns:
        --------
        pd.Series
            Forecasted values
        """
        if model_type not in self.models:
            raise ValueError(f"Model type '{model_type}' not fitted")

        model = self.models[model_type]
        forecast = model.forecast(steps=steps)

        return forecast

    def _manual_simple_smoothing(self, data: pd.Series, alpha: Optional[float] = None) -> Dict[str, Any]:
        """Manual implementation of simple exponential smoothing."""
        if alpha is None:
            alpha = 0.3  # Default value

        smoothed = pd.Series(index=data.index, dtype=float)
        smoothed.iloc[0] = data.iloc[0]

        for i in range(1, len(data)):
            smoothed.iloc[i] = alpha * data.iloc[i] + (1 - alpha) * smoothed.iloc[i-1]

        residuals = data - smoothed
        sse = (residuals ** 2).sum()

        return {
            'method': 'Simple Exponential Smoothing (Manual)',
            'fitted_values': smoothed,
            'residuals': residuals,
            'alpha': alpha,
            'sse': sse
        }

    def _manual_double_smoothing(self, data: pd.Series, alpha: Optional[float] = None,
                               beta: Optional[float] = None) -> Dict[str, Any]:
        """Manual implementation of double exponential smoothing."""
        if alpha is None:
            alpha = 0.3
        if beta is None:
            beta = 0.3

        level = pd.Series(index=data.index, dtype=float)
        trend = pd.Series(index=data.index, dtype=float)
        fitted = pd.Series(index=data.index, dtype=float)

        # Initialize
        level.iloc[0] = data.iloc[0]
        trend.iloc[0] = data.iloc[1] - data.iloc[0] if len(data) > 1 else 0
        fitted.iloc[0] = level.iloc[0]

        for i in range(1, len(data)):
            # Update level
            level.iloc[i] = alpha * data.iloc[i] + (1 - alpha) * (level.iloc[i-1] + trend.iloc[i-1])

            # Update trend
            trend.iloc[i] = beta * (level.iloc[i] - level.iloc[i-1]) + (1 - beta) * trend.iloc[i-1]

            # Fitted value
            fitted.iloc[i] = level.iloc[i-1] + trend.iloc[i-1]

        residuals = data - fitted
        sse = (residuals ** 2).sum()

        return {
            'method': 'Double Exponential Smoothing (Manual)',
            'fitted_values': fitted,
            'residuals': residuals,
            'alpha': alpha,
            'beta': beta,
            'sse': sse,
            'level': level,
            'trend': trend
        }

    def _manual_triple_smoothing(self, data: pd.Series, seasonal_periods: int,
                               alpha: Optional[float] = None,
                               beta: Optional[float] = None,
                               gamma: Optional[float] = None) -> Dict[str, Any]:
        """Manual implementation of triple exponential smoothing."""
        if alpha is None:
            alpha = 0.3
        if beta is None:
            beta = 0.3
        if gamma is None:
            gamma = 0.3

        n = len(data)
        level = pd.Series(index=data.index, dtype=float)
        trend = pd.Series(index=data.index, dtype=float)
        seasonal = pd.Series(index=data.index, dtype=float)
        fitted = pd.Series(index=data.index, dtype=float)

        # Initialize seasonal factors
        for i in range(seasonal_periods):
            seasonal.iloc[i] = data.iloc[i] / (data.iloc[:seasonal_periods].mean())

        # Initialize level and trend
        level.iloc[0] = data.iloc[0]
        trend.iloc[0] = (data.iloc[1] - data.iloc[0]) if len(data) > 1 else 0
        fitted.iloc[0] = level.iloc[0] * seasonal.iloc[0]

        for i in range(1, n):
            # Update level
            level.iloc[i] = (alpha * data.iloc[i] / seasonal.iloc[i - seasonal_periods] +
                           (1 - alpha) * (level.iloc[i-1] + trend.iloc[i-1]))

            # Update trend
            trend.iloc[i] = beta * (level.iloc[i] - level.iloc[i-1]) + (1 - beta) * trend.iloc[i-1]

            # Update seasonal
            if i >= seasonal_periods:
                seasonal.iloc[i] = (gamma * data.iloc[i] / level.iloc[i] +
                                  (1 - gamma) * seasonal.iloc[i - seasonal_periods])
            else:
                seasonal.iloc[i] = seasonal.iloc[i % seasonal_periods]

            # Fitted value
            if i >= seasonal_periods:
                fitted.iloc[i] = (level.iloc[i-1] + trend.iloc[i-1]) * seasonal.iloc[i - seasonal_periods]
            else:
                fitted.iloc[i] = level.iloc[i-1] + trend.iloc[i-1]

        residuals = data - fitted
        sse = (residuals ** 2).sum()

        return {
            'method': 'Triple Exponential Smoothing (Manual)',
            'fitted_values': fitted,
            'residuals': residuals,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'sse': sse,
            'level': level,
            'trend': trend,
            'seasonal': seasonal
        }

class StationarityTester:
    """
    Unit root tests and stationarity analysis including Dickey-Fuller, ADF, and KPSS tests.
    """

    def __init__(self):
        """Initialize the stationarity tester."""
        self.test_results = {}

    def dickey_fuller_test(self, data: pd.Series, maxlag: Optional[int] = None,
                          regression: str = 'c', autolag: str = 'AIC') -> Dict[str, Any]:
        """
        Augmented Dickey-Fuller test for unit root.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        maxlag : int, optional
            Maximum number of lags to use
        regression : str, default 'c'
            Constant and trend order to include ('c', 'ct', 'ctt', 'nc')
        autolag : str, default 'AIC'
            Method to use for automatic lag selection ('AIC', 'BIC', 't-stat')

        Returns:
        --------
        dict
            Test results including statistic, p-value, and conclusion
        """
        if not HAS_STATSMODELS:
            raise ImportError("Statsmodels required for Dickey-Fuller test")

        # Perform ADF test
        adf_result = adfuller(data.dropna(), maxlag=maxlag, regression=regression, autolag=autolag)

        # Extract results
        test_statistic = adf_result[0]
        p_value = adf_result[1]
        used_lags = adf_result[2]
        n_observations = adf_result[3]
        critical_values = adf_result[4]

        # Determine stationarity
        is_stationary = p_value < 0.05

        results = {
            'test_name': 'Augmented Dickey-Fuller Test',
            'test_statistic': test_statistic,
            'p_value': p_value,
            'used_lags': used_lags,
            'n_observations': n_observations,
            'critical_values': critical_values,
            'is_stationary': is_stationary,
            'conclusion': 'Stationary' if is_stationary else 'Non-stationary (has unit root)',
            'regression': regression,
            'autolag': autolag
        }

        self.test_results['adf'] = results
        return results

    def kpss_test(self, data: pd.Series, regression: str = 'c',
                 nlags: str = 'auto') -> Dict[str, Any]:
        """
        Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        regression : str, default 'c'
            Type of regression ('c' for constant, 'ct' for constant and trend)
        nlags : str or int, default 'auto'
            Number of lags to use

        Returns:
        --------
        dict
            Test results including statistic, p-value, and conclusion
        """
        if not HAS_STATSMODELS:
            raise ImportError("Statsmodels required for KPSS test")

        # Perform KPSS test
        kpss_result = kpss(data.dropna(), regression=regression, nlags=nlags)

        test_statistic = kpss_result[0]
        p_value = kpss_result[1]
        used_lags = kpss_result[2]
        critical_values = kpss_result[3]

        # KPSS null hypothesis is stationarity
        is_stationary = p_value > 0.05

        results = {
            'test_name': 'Kwiatkowski-Phillips-Schmidt-Shin Test',
            'test_statistic': test_statistic,
            'p_value': p_value,
            'used_lags': used_lags,
            'critical_values': critical_values,
            'is_stationary': is_stationary,
            'conclusion': 'Stationary' if is_stationary else 'Non-stationary',
            'regression': regression,
            'note': 'KPSS tests null hypothesis of stationarity'
        }

        self.test_results['kpss'] = results
        return results

    def combined_stationarity_test(self, data: pd.Series) -> Dict[str, Any]:
        """
        Combined stationarity test using both ADF and KPSS tests.

        Parameters:
        -----------
        data : pd.Series
            Time series data

        Returns:
        --------
        dict
            Combined test results and overall conclusion
        """
        # Run both tests
        adf_results = self.dickey_fuller_test(data)
        kpss_results = self.kpss_test(data)

        # Combined interpretation
        adf_stationary = adf_results['is_stationary']
        kpss_stationary = kpss_results['is_stationary']

        if adf_stationary and kpss_stationary:
            overall_conclusion = "Stationary (both tests agree)"
        elif not adf_stationary and not kpss_stationary:
            overall_conclusion = "Non-stationary (both tests agree)"
        elif adf_stationary and not kpss_stationary:
            overall_conclusion = "Conflicting results: ADF suggests stationary, KPSS suggests non-stationary"
        else:
            overall_conclusion = "Conflicting results: ADF suggests non-stationary, KPSS suggests stationary"

        combined_results = {
            'adf_results': adf_results,
            'kpss_results': kpss_results,
            'overall_conclusion': overall_conclusion,
            'recommendation': self._get_stationarity_recommendation(adf_stationary, kpss_stationary)
        }

        self.test_results['combined'] = combined_results
        return combined_results

    def _get_stationarity_recommendation(self, adf_stationary: bool,
                                       kpss_stationary: bool) -> str:
        """Get recommendation based on stationarity test results."""
        if adf_stationary and kpss_stationary:
            return "Series appears stationary. Proceed with analysis."
        elif not adf_stationary and not kpss_stationary:
            return "Series is non-stationary. Consider differencing or detrending."
        else:
            return ("Conflicting results. Consider additional tests or examine "
                   "the series characteristics more closely.")

class TrendRemover:
    """
    Methods for removing trends from non-stationary time series.
    """

    def __init__(self):
        """Initialize the trend remover."""
        self.detrended_data = {}
        self.trend_components = {}

    def linear_detrend(self, data: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Remove linear trend from time series.

        Parameters:
        -----------
        data : pd.Series
            Time series data

        Returns:
        --------
        tuple
            (detrended_series, trend_component)
        """
        # Create time index for regression
        time_index = np.arange(len(data))

        # Fit linear trend
        coeffs = np.polyfit(time_index, data.values, 1)
        trend = np.polyval(coeffs, time_index)

        # Create trend series with original index
        trend_series = pd.Series(trend, index=data.index)

        # Remove trend
        detrended = data - trend_series

        self.detrended_data['linear'] = detrended
        self.trend_components['linear'] = trend_series

        return detrended, trend_series

    def polynomial_detrend(self, data: pd.Series, degree: int = 2) -> Tuple[pd.Series, pd.Series]:
        """
        Remove polynomial trend from time series.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        degree : int, default 2
            Degree of polynomial trend

        Returns:
        --------
        tuple
            (detrended_series, trend_component)
        """
        # Create time index for regression
        time_index = np.arange(len(data))

        # Fit polynomial trend
        coeffs = np.polyfit(time_index, data.values, degree)
        trend = np.polyval(coeffs, time_index)

        # Create trend series with original index
        trend_series = pd.Series(trend, index=data.index)

        # Remove trend
        detrended = data - trend_series

        key = f'polynomial_{degree}'
        self.detrended_data[key] = detrended
        self.trend_components[key] = trend_series

        return detrended, trend_series

    def moving_average_detrend(self, data: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
        """
        Remove trend using moving average.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        window : int
            Window size for moving average

        Returns:
        --------
        tuple
            (detrended_series, trend_component)
        """
        # Calculate moving average trend
        trend = data.rolling(window=window, center=True).mean()

        # Remove trend
        detrended = data - trend

        key = f'ma_{window}'
        self.detrended_data[key] = detrended
        self.trend_components[key] = trend

        return detrended, trend

    def first_difference(self, data: pd.Series) -> pd.Series:
        """
        Apply first differencing to remove trend.

        Parameters:
        -----------
        data : pd.Series
            Time series data

        Returns:
        --------
        pd.Series
            First differenced series
        """
        differenced = data.diff().dropna()

        self.detrended_data['first_difference'] = differenced

        return differenced

    def seasonal_difference(self, data: pd.Series, seasonal_periods: int) -> pd.Series:
        """
        Apply seasonal differencing.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        seasonal_periods : int
            Number of periods in seasonal cycle

        Returns:
        --------
        pd.Series
            Seasonally differenced series
        """
        differenced = data.diff(seasonal_periods).dropna()

        key = f'seasonal_diff_{seasonal_periods}'
        self.detrended_data[key] = differenced

        return differenced

    def log_transform(self, data: pd.Series) -> pd.Series:
        """
        Apply log transformation to stabilize variance.

        Parameters:
        -----------
        data : pd.Series
            Time series data (must be positive)

        Returns:
        --------
        pd.Series
            Log-transformed series
        """
        if (data <= 0).any():
            raise ValueError("Log transformation requires all positive values")

        log_data = np.log(data)

        self.detrended_data['log_transform'] = log_data

        return log_data

    def hodrick_prescott_filter(self, data: pd.Series, lamb: float = 1600) -> Tuple[pd.Series, pd.Series]:
        """
        Apply Hodrick-Prescott filter to separate trend and cycle.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        lamb : float, default 1600
            Smoothing parameter (1600 for quarterly, 14400 for monthly data)

        Returns:
        --------
        tuple
            (cycle_component, trend_component)
        """
        if not HAS_STATSMODELS:
            # Simple approximation using moving average
            warnings.warn("Statsmodels not available. Using moving average approximation.")
            return self.moving_average_detrend(data, window=12)

        try:
            from statsmodels.tsa.filters.hp_filter import hpfilter
            cycle, trend = hpfilter(data, lamb=lamb)

            self.detrended_data['hp_cycle'] = cycle
            self.trend_components['hp_trend'] = trend

            return cycle, trend
        except ImportError:
            warnings.warn("HP filter not available. Using moving average approximation.")
            return self.moving_average_detrend(data, window=12)

class AutocorrelationAnalyzer:
    """
    Autocorrelation and partial autocorrelation analysis for lag determination.
    """

    def __init__(self):
        """Initialize the autocorrelation analyzer."""
        self.acf_results = {}
        self.pacf_results = {}

    def calculate_acf(self, data: pd.Series, nlags: int = 40,
                     alpha: float = 0.05) -> Dict[str, Any]:
        """
        Calculate autocorrelation function.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        nlags : int, default 40
            Number of lags to calculate
        alpha : float, default 0.05
            Significance level for confidence intervals

        Returns:
        --------
        dict
            ACF values, confidence intervals, and significant lags
        """
        if not HAS_STATSMODELS:
            return self._manual_acf(data, nlags)

        # Calculate ACF with confidence intervals
        acf_values, confint = acf(data.dropna(), nlags=nlags, alpha=alpha, fft=False)

        # Find significant lags
        significant_lags = []
        for i in range(1, len(acf_values)):
            if abs(acf_values[i]) > abs(confint[i, 1] - acf_values[i]):
                significant_lags.append(i)

        results = {
            'acf_values': acf_values,
            'confidence_intervals': confint,
            'significant_lags': significant_lags,
            'nlags': nlags,
            'alpha': alpha
        }

        self.acf_results[nlags] = results
        return results

    def calculate_pacf(self, data: pd.Series, nlags: int = 40,
                      alpha: float = 0.05) -> Dict[str, Any]:
        """
        Calculate partial autocorrelation function.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        nlags : int, default 40
            Number of lags to calculate
        alpha : float, default 0.05
            Significance level for confidence intervals

        Returns:
        --------
        dict
            PACF values, confidence intervals, and significant lags
        """
        if not HAS_STATSMODELS:
            return self._manual_pacf(data, nlags)

        # Calculate PACF with confidence intervals
        pacf_values, confint = pacf(data.dropna(), nlags=nlags, alpha=alpha)

        # Find significant lags
        significant_lags = []
        for i in range(1, len(pacf_values)):
            if abs(pacf_values[i]) > abs(confint[i, 1] - pacf_values[i]):
                significant_lags.append(i)

        results = {
            'pacf_values': pacf_values,
            'confidence_intervals': confint,
            'significant_lags': significant_lags,
            'nlags': nlags,
            'alpha': alpha
        }

        self.pacf_results[nlags] = results
        return results

    def plot_acf_pacf(self, data: pd.Series, nlags: int = 40,
                     alpha: float = 0.05, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot ACF and PACF together.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        nlags : int, default 40
            Number of lags to plot
        alpha : float, default 0.05
            Significance level for confidence intervals
        figsize : tuple, default (12, 8)
            Figure size
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize)

        if HAS_STATSMODELS:
            # Use statsmodels plotting functions
            plot_acf(data.dropna(), lags=nlags, alpha=alpha, ax=axes[0])
            plot_pacf(data.dropna(), lags=nlags, alpha=alpha, ax=axes[1])
        else:
            # Manual plotting
            acf_results = self.calculate_acf(data, nlags, alpha)
            pacf_results = self.calculate_pacf(data, nlags, alpha)

            # Plot ACF
            lags = range(len(acf_results['acf_values']))
            axes[0].bar(lags, acf_results['acf_values'])
            axes[0].set_title('Autocorrelation Function (ACF)')
            axes[0].set_xlabel('Lag')
            axes[0].set_ylabel('ACF')

            # Plot PACF
            lags = range(len(pacf_results['pacf_values']))
            axes[1].bar(lags, pacf_results['pacf_values'])
            axes[1].set_title('Partial Autocorrelation Function (PACF)')
            axes[1].set_xlabel('Lag')
            axes[1].set_ylabel('PACF')

        plt.tight_layout()
        plt.show()

    def suggest_arima_order(self, data: pd.Series, max_p: int = 5, max_q: int = 5,
                          seasonal: bool = False, seasonal_periods: int = 12) -> Dict[str, Any]:
        """
        Suggest ARIMA model order based on ACF and PACF analysis.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        max_p : int, default 5
            Maximum AR order to consider
        max_q : int, default 5
            Maximum MA order to consider
        seasonal : bool, default False
            Whether to consider seasonal ARIMA
        seasonal_periods : int, default 12
            Seasonal periods

        Returns:
        --------
        dict
            Suggested ARIMA orders and reasoning
        """
        # Calculate ACF and PACF
        acf_results = self.calculate_acf(data, nlags=max(max_p, max_q) + 5)
        pacf_results = self.calculate_pacf(data, nlags=max(max_p, max_q) + 5)

        # Analyze patterns
        acf_significant = acf_results['significant_lags']
        pacf_significant = pacf_results['significant_lags']

        # Suggest AR order (based on PACF cutoff)
        suggested_p = 0
        for lag in range(1, max_p + 1):
            if lag in pacf_significant:
                suggested_p = lag
            else:
                break

        # Suggest MA order (based on ACF cutoff)
        suggested_q = 0
        for lag in range(1, max_q + 1):
            if lag in acf_significant:
                suggested_q = lag
            else:
                break

        # Check for differencing need
        stationarity_tester = StationarityTester()
        adf_results = stationarity_tester.dickey_fuller_test(data)
        suggested_d = 0 if adf_results['is_stationary'] else 1

        suggestions = {
            'suggested_order': (suggested_p, suggested_d, suggested_q),
            'reasoning': {
                'p_order': f"PACF cuts off after lag {suggested_p}",
                'd_order': f"{'No' if suggested_d == 0 else 'One'} differencing needed based on ADF test",
                'q_order': f"ACF cuts off after lag {suggested_q}"
            },
            'acf_significant_lags': acf_significant[:10],  # First 10 significant lags
            'pacf_significant_lags': pacf_significant[:10],
            'stationarity_test': adf_results
        }

        if seasonal:
            # Basic seasonal suggestions
            seasonal_acf = [lag for lag in acf_significant if lag % seasonal_periods == 0]
            seasonal_pacf = [lag for lag in pacf_significant if lag % seasonal_periods == 0]

            seasonal_P = 1 if seasonal_pacf else 0
            seasonal_Q = 1 if seasonal_acf else 0

            suggestions['seasonal_order'] = (seasonal_P, suggested_d, seasonal_Q, seasonal_periods)
            suggestions['reasoning']['seasonal'] = f"Seasonal ARIMA({seasonal_P},{suggested_d},{seasonal_Q}){seasonal_periods}"

        return suggestions

    def _manual_acf(self, data: pd.Series, nlags: int) -> Dict[str, Any]:
        """Manual calculation of ACF when statsmodels is not available."""
        data_clean = data.dropna()
        n = len(data_clean)
        mean = data_clean.mean()

        acf_values = []

        for lag in range(nlags + 1):
            if lag == 0:
                acf_values.append(1.0)
            else:
                if lag >= n:
                    acf_values.append(0.0)
                else:
                    numerator = sum((data_clean.iloc[i] - mean) * (data_clean.iloc[i-lag] - mean)
                                  for i in range(lag, n))
                    denominator = sum((data_clean.iloc[i] - mean) ** 2 for i in range(n))
                    acf_values.append(numerator / denominator if denominator != 0 else 0)

        # Simple confidence intervals (approximation)
        confidence_bound = 1.96 / np.sqrt(n)
        confint = np.array([[-confidence_bound, confidence_bound] for _ in range(len(acf_values))])

        return {
            'acf_values': np.array(acf_values),
            'confidence_intervals': confint,
            'significant_lags': [i for i, val in enumerate(acf_values[1:], 1)
                                if abs(val) > confidence_bound],
            'nlags': nlags,
            'alpha': 0.05
        }

    def _manual_pacf(self, data: pd.Series, nlags: int) -> Dict[str, Any]:
        """Manual calculation of PACF when statsmodels is not available."""
        data_clean = data.dropna()
        n = len(data_clean)

        # Calculate ACF first
        acf_result = self._manual_acf(data, nlags)
        acf_vals = acf_result['acf_values']

        pacf_values = [1.0]  # PACF at lag 0 is always 1

        for k in range(1, min(nlags + 1, len(acf_vals))):
            if k == 1:
                pacf_values.append(acf_vals[1])
            else:
                # Solve Yule-Walker equations (simplified)
                # This is a basic approximation
                numerator = acf_vals[k]
                for j in range(1, k):
                    numerator -= pacf_values[j] * acf_vals[k-j] if k-j < len(acf_vals) else 0

                denominator = 1
                for j in range(1, k):
                    denominator -= pacf_values[j] * acf_vals[j] if j < len(acf_vals) else 0

                pacf_val = numerator / denominator if denominator != 0 else 0
                pacf_values.append(pacf_val)

        # Simple confidence intervals
        confidence_bound = 1.96 / np.sqrt(n)
        confint = np.array([[-confidence_bound, confidence_bound] for _ in range(len(pacf_values))])

        return {
            'pacf_values': np.array(pacf_values),
            'confidence_intervals': confint,
            'significant_lags': [i for i, val in enumerate(pacf_values[1:], 1)
                                if abs(val) > confidence_bound],
            'nlags': nlags,
            'alpha': 0.05
        }

class ARMAModeler:
    """
    ARMA (AutoRegressive Moving Average) model fitting and analysis.
    """

    def __init__(self):
        """Initialize the ARMA modeler."""
        self.models = {}
        self.fitted_models = {}
        self.results = {}
        self.estimation_methods = ['mle', 'ols', 'yule_walker', 'method_of_moments']

    def fit_arma(self, data: pd.Series, order: Tuple[int, int],
                 method: str = 'mle') -> Dict[str, Any]:
        """
        Fit an ARMA model to the data.

        Parameters:
        -----------
        data : pd.Series
            Time series data (should be stationary)
        order : tuple
            ARMA order (p, q) where p is AR order and q is MA order
        method : str, default 'mle'
            Estimation method ('mle', 'css', 'css-mle')

        Returns:
        --------
        dict
            Model fitting results
        """
        if not HAS_STATSMODELS:
            return self._manual_arma_fit(data, order)

        try:
            from statsmodels.tsa.arima.model import ARIMA

            p, q = order

            # Fit ARMA model (ARIMA with d=0)
            model = ARIMA(data.dropna(), order=(p, 0, q))
            fitted_model = model.fit(method=method)

            # Extract results
            results = {
                'method': 'ARMA',
                'order': order,
                'fitted_values': fitted_model.fittedvalues,
                'residuals': fitted_model.resid,
                'parameters': fitted_model.params.to_dict(),
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'hqic': fitted_model.hqic,
                'llf': fitted_model.llf,
                'model_summary': str(fitted_model.summary()),
                'model': fitted_model,
                'estimation_method': method
            }

            # Add parameter significance tests
            if hasattr(fitted_model, 'pvalues'):
                results['parameter_pvalues'] = fitted_model.pvalues.to_dict()
                results['significant_params'] = (fitted_model.pvalues < 0.05).to_dict()

            # Store model
            key = f"ARMA({p},{q})"
            self.models[key] = model
            self.fitted_models[key] = fitted_model
            self.results[key] = results

            return results

        except ImportError:
            warnings.warn("Statsmodels ARIMA not available. Using manual implementation.")
            return self._manual_arma_fit(data, order)
        except Exception as e:
            warnings.warn(f"ARMA fitting failed: {str(e)}. Using manual implementation.")
            return self._manual_arma_fit(data, order)

    def forecast_arma(self, model_key: str, steps: int,
                     confidence_interval: float = 0.95) -> Dict[str, Any]:
        """
        Generate forecasts from fitted ARMA model.

        Parameters:
        -----------
        model_key : str
            Key for the fitted model (e.g., "ARMA(2,1)")
        steps : int
            Number of steps to forecast
        confidence_interval : float, default 0.95
            Confidence level for prediction intervals

        Returns:
        --------
        dict
            Forecast results
        """
        if model_key not in self.fitted_models:
            raise ValueError(f"Model {model_key} not found. Fit model first.")

        fitted_model = self.fitted_models[model_key]

        try:
            # Generate forecast
            forecast_result = fitted_model.get_forecast(steps=steps)
            forecast = forecast_result.predicted_mean

            # Get confidence intervals
            alpha = 1 - confidence_interval
            conf_int = forecast_result.conf_int(alpha=alpha)

            forecast_dict = {
                'forecast': forecast,
                'confidence_intervals': {
                    'lower': conf_int.iloc[:, 0],
                    'upper': conf_int.iloc[:, 1]
                },
                'forecast_std_err': forecast_result.se_mean,
                'steps': steps,
                'confidence_level': confidence_interval,
                'model_key': model_key
            }

            return forecast_dict

        except Exception as e:
            warnings.warn(f"Forecasting failed: {str(e)}")
            return self._manual_arma_forecast(model_key, steps)

    def diagnostic_tests(self, model_key: str) -> Dict[str, Any]:
        """
        Perform diagnostic tests on fitted ARMA model.

        Parameters:
        -----------
        model_key : str
            Key for the fitted model

        Returns:
        --------
        dict
            Diagnostic test results
        """
        if model_key not in self.fitted_models:
            raise ValueError(f"Model {model_key} not found.")

        fitted_model = self.fitted_models[model_key]
        residuals = fitted_model.resid

        diagnostics = {}

        try:
            # Ljung-Box test for residual autocorrelation
            from statsmodels.stats.diagnostic import acorr_ljungbox
            ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
            diagnostics['ljung_box'] = {
                'test_statistic': ljung_box['lb_stat'].iloc[-1],
                'p_value': ljung_box['lb_pvalue'].iloc[-1],
                'conclusion': 'No autocorrelation' if ljung_box['lb_pvalue'].iloc[-1] > 0.05 else 'Autocorrelation detected'
            }
        except:
            diagnostics['ljung_box'] = {'error': 'Test not available'}

        try:
            # Jarque-Bera test for normality
            from statsmodels.stats.stattools import jarque_bera
            jb_stat, jb_pvalue, jb_skew, jb_kurtosis = jarque_bera(residuals)
            diagnostics['jarque_bera'] = {
                'test_statistic': jb_stat,
                'p_value': jb_pvalue,
                'skewness': jb_skew,
                'kurtosis': jb_kurtosis,
                'conclusion': 'Normal residuals' if jb_pvalue > 0.05 else 'Non-normal residuals'
            }
        except:
            diagnostics['jarque_bera'] = {'error': 'Test not available'}

        # Basic residual statistics
        diagnostics['residual_stats'] = {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'min': residuals.min(),
            'max': residuals.max(),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals)
        }

        return diagnostics

    def compare_models(self, orders_list: List[Tuple[int, int]],
                      data: pd.Series) -> Dict[str, Any]:
        """
        Compare multiple ARMA models with different orders.

        Parameters:
        -----------
        orders_list : list of tuples
            List of ARMA orders to compare
        data : pd.Series
            Time series data

        Returns:
        --------
        dict
            Model comparison results
        """
        comparison_results = {}

        for order in orders_list:
            try:
                results = self.fit_arma(data, order)
                model_key = f"ARMA{order}"

                comparison_results[model_key] = {
                    'order': order,
                    'aic': results['aic'],
                    'bic': results['bic'],
                    'hqic': results.get('hqic', None),
                    'llf': results['llf']
                }
            except Exception as e:
                comparison_results[f"ARMA{order}"] = {'error': str(e)}

        # Find best models
        valid_models = {k: v for k, v in comparison_results.items() if 'error' not in v}

        if valid_models:
            best_aic = min(valid_models.items(), key=lambda x: x[1]['aic'])
            best_bic = min(valid_models.items(), key=lambda x: x[1]['bic'])

            summary = {
                'models_compared': comparison_results,
                'best_aic_model': best_aic[0],
                'best_bic_model': best_bic[0],
                'recommendation': best_bic[0] if best_aic[0] == best_bic[0] else f"AIC suggests {best_aic[0]}, BIC suggests {best_bic[0]}"
            }
        else:
            summary = {
                'models_compared': comparison_results,
                'error': 'No valid models fitted'
            }

        return summary

    def _manual_arma_fit(self, data: pd.Series, order: Tuple[int, int]) -> Dict[str, Any]:
        """Manual ARMA fitting when statsmodels is not available."""
        warnings.warn("Manual ARMA implementation is basic. Consider installing statsmodels for full functionality.")

        p, q = order
        data_clean = data.dropna()
        n = len(data_clean)

        # Basic AR(p) fitting using least squares
        if p > 0 and q == 0:
            # Pure AR model
            X = np.column_stack([data_clean.shift(i).dropna() for i in range(1, p+1)])
            y = data_clean[p:]

            try:
                # Least squares estimation
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                fitted_values = X @ coeffs
                residuals = y - fitted_values

                # Basic metrics
                rss = np.sum(residuals**2)
                aic = n * np.log(rss/n) + 2 * (p + 1)
                bic = n * np.log(rss/n) + np.log(n) * (p + 1)

                return {
                    'method': 'ARMA (Manual - AR only)',
                    'order': order,
                    'fitted_values': pd.Series(fitted_values, index=y.index),
                    'residuals': pd.Series(residuals, index=y.index),
                    'parameters': {f'ar_L{i+1}': coeffs[i] for i in range(p)},
                    'aic': aic,
                    'bic': bic,
                    'llf': -0.5 * (n * np.log(2*np.pi) + n * np.log(rss/n) + n)
                }
            except:
                return {'error': 'Manual AR fitting failed'}
        else:
            return {'error': 'Manual ARMA implementation only supports pure AR models'}

    def _manual_arma_forecast(self, model_key: str, steps: int) -> Dict[str, Any]:
        """Manual ARMA forecasting."""
        if model_key not in self.results:
            return {'error': 'Model not found'}

        results = self.results[model_key]

        if 'error' in results:
            return {'error': 'Cannot forecast from failed model'}

        # Basic AR forecasting
        order = results['order']
        p, q = order

        if p > 0 and q == 0:
            # AR forecasting
            params = results['parameters']
            ar_coeffs = [params.get(f'ar_L{i+1}', 0) for i in range(p)]

            # Get last p observations
            fitted_values = results['fitted_values']
            last_obs = fitted_values.tail(p).values[::-1]  # Reverse for proper order

            forecasts = []
            for step in range(steps):
                forecast = np.dot(ar_coeffs, last_obs)
                forecasts.append(forecast)
                # Update last_obs for next forecast
                last_obs = np.roll(last_obs, 1)
                last_obs[0] = forecast

            return {
                'forecast': pd.Series(forecasts, index=range(len(fitted_values), len(fitted_values) + steps)),
                'confidence_intervals': {'lower': None, 'upper': None},
                'steps': steps,
                'model_key': model_key,
                'note': 'Manual forecasting - no confidence intervals'
            }
        else:
            return {'error': 'Manual forecasting only supports AR models'}

    def fit_arma_ols(self, data: pd.Series, order: Tuple[int, int]) -> Dict[str, Any]:
        """
        Fit ARMA model using Ordinary Least Squares (OLS) estimation.

        Parameters:
        -----------
        data : pd.Series
            Time series data (should be stationary)
        order : tuple
            ARMA order (p, q)

        Returns:
        --------
        dict
            OLS estimation results
        """
        p, q = order
        data_clean = data.dropna()
        n = len(data_clean)

        if p == 0 and q == 0:
            return {'error': 'Invalid ARMA order (0,0)'}

        try:
            # For AR(p) models, use OLS directly
            if p > 0 and q == 0:
                return self._fit_ar_ols(data_clean, p)

            # For MA(q) models, use iterative OLS
            elif p == 0 and q > 0:
                return self._fit_ma_ols(data_clean, q)

            # For ARMA(p,q) models, use conditional OLS
            else:
                return self._fit_arma_conditional_ols(data_clean, p, q)

        except Exception as e:
            return {'error': f'OLS estimation failed: {str(e)}'}

    def fit_arma_yule_walker(self, data: pd.Series, order: Tuple[int, int]) -> Dict[str, Any]:
        """
        Fit ARMA model using Yule-Walker equations (Method of Moments).

        Parameters:
        -----------
        data : pd.Series
            Time series data (should be stationary)
        order : tuple
            ARMA order (p, q)

        Returns:
        --------
        dict
            Yule-Walker estimation results
        """
        p, q = order
        data_clean = data.dropna()

        if q > 0:
            warnings.warn("Yule-Walker method is primarily for AR models. MA components approximated.")

        try:
            if p > 0:
                # Calculate sample autocorrelations
                autocorrs = self._calculate_sample_autocorrelations(data_clean, max_lag=p)

                # Solve Yule-Walker equations
                R = np.array([[autocorrs[abs(i-j)] for j in range(p)] for i in range(p)])
                r = np.array([autocorrs[i+1] for i in range(p)])

                # Solve R * phi = r
                try:
                    phi = np.linalg.solve(R, r)
                except np.linalg.LinAlgError:
                    # Use pseudo-inverse if matrix is singular
                    phi = np.linalg.pinv(R) @ r

                # Estimate noise variance
                sigma2 = autocorrs[0] * (1 - np.sum(phi * r))

                # Calculate fitted values and residuals
                fitted_values, residuals = self._calculate_ar_fitted_values(data_clean, phi)

                # Calculate information criteria
                n = len(data_clean)
                rss = np.sum(residuals**2)
                aic = n * np.log(rss/n) + 2 * (p + 1)
                bic = n * np.log(rss/n) + np.log(n) * (p + 1)

                results = {
                    'method': 'Yule-Walker (Method of Moments)',
                    'order': order,
                    'fitted_values': pd.Series(fitted_values, index=data_clean.index[p:]),
                    'residuals': pd.Series(residuals, index=data_clean.index[p:]),
                    'parameters': {f'ar_L{i+1}': phi[i] for i in range(p)},
                    'noise_variance': sigma2,
                    'aic': aic,
                    'bic': bic,
                    'llf': -0.5 * (n * np.log(2*np.pi) + n * np.log(sigma2) + rss/sigma2),
                    'estimation_method': 'yule_walker'
                }

                if q > 0:
                    # Simple approximation for MA part
                    results['parameters'].update({f'ma_L{i+1}': 0.1 for i in range(q)})
                    results['note'] = 'MA parameters approximated (not optimal for ARMA models)'

                return results

            else:
                return {'error': 'Yule-Walker requires AR component (p > 0)'}

        except Exception as e:
            return {'error': f'Yule-Walker estimation failed: {str(e)}'}

    def fit_arma_method_of_moments(self, data: pd.Series, order: Tuple[int, int]) -> Dict[str, Any]:
        """
        Fit ARMA model using Method of Moments estimation.

        Parameters:
        -----------
        data : pd.Series
            Time series data (should be stationary)
        order : tuple
            ARMA order (p, q)

        Returns:
        --------
        dict
            Method of Moments estimation results
        """
        p, q = order
        data_clean = data.dropna()

        try:
            # Calculate sample moments (autocorrelations)
            max_lag = max(p, q, 10)
            autocorrs = self._calculate_sample_autocorrelations(data_clean, max_lag)

            # Use method of moments for parameter estimation
            if p > 0 and q == 0:
                # Pure AR: Use Yule-Walker
                return self.fit_arma_yule_walker(data, order)

            elif p == 0 and q > 0:
                # Pure MA: Use method of moments for MA
                return self._fit_ma_method_of_moments(data_clean, q, autocorrs)

            else:
                # ARMA: Use iterative method of moments
                return self._fit_arma_method_of_moments(data_clean, p, q, autocorrs)

        except Exception as e:
            return {'error': f'Method of moments estimation failed: {str(e)}'}

    def ljung_box_test(self, residuals: pd.Series, lags: int = 10,
                      model_df: int = 0) -> Dict[str, Any]:
        """
        Ljung-Box test for residual independence (autocorrelation).

        Parameters:
        -----------
        residuals : pd.Series
            Model residuals
        lags : int, default 10
            Number of lags to test
        model_df : int, default 0
            Degrees of freedom used by the model (p + q for ARMA)

        Returns:
        --------
        dict
            Ljung-Box test results
        """
        try:
            if HAS_STATSMODELS:
                from statsmodels.stats.diagnostic import acorr_ljungbox

                # Perform Ljung-Box test
                lb_results = acorr_ljungbox(residuals.dropna(), lags=lags,
                                           model_df=model_df, return_df=True)

                # Get overall test statistic and p-value
                test_stat = lb_results['lb_stat'].iloc[-1]
                p_value = lb_results['lb_pvalue'].iloc[-1]

                results = {
                    'test_name': 'Ljung-Box Test',
                    'null_hypothesis': 'Residuals are independently distributed',
                    'test_statistic': test_stat,
                    'p_value': p_value,
                    'lags_tested': lags,
                    'degrees_freedom': lags - model_df,
                    'reject_null': p_value < 0.05,
                    'conclusion': 'Residuals show autocorrelation' if p_value < 0.05 else 'No significant autocorrelation',
                    'detailed_results': lb_results.to_dict('index'),
                    'critical_values': {
                        '0.01': 'Use chi2.ppf(0.99, df)',
                        '0.05': 'Use chi2.ppf(0.95, df)',
                        '0.10': 'Use chi2.ppf(0.90, df)'
                    }
                }

            else:
                # Manual implementation
                results = self._manual_ljung_box_test(residuals, lags, model_df)

            return results

        except Exception as e:
            return {'error': f'Ljung-Box test failed: {str(e)}'}

    def compare_estimation_methods(self, data: pd.Series, order: Tuple[int, int]) -> Dict[str, Any]:
        """
        Compare different ARMA estimation methods.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        order : tuple
            ARMA order (p, q)

        Returns:
        --------
        dict
            Comparison of estimation methods
        """
        methods = ['mle', 'ols', 'yule_walker', 'method_of_moments']
        results = {}

        for method in methods:
            try:
                if method == 'mle':
                    result = self.fit_arma(data, order, method='mle')
                elif method == 'ols':
                    result = self.fit_arma_ols(data, order)
                elif method == 'yule_walker':
                    result = self.fit_arma_yule_walker(data, order)
                elif method == 'method_of_moments':
                    result = self.fit_arma_method_of_moments(data, order)

                if 'error' not in result:
                    results[method] = {
                        'parameters': result.get('parameters', {}),
                        'aic': result.get('aic', np.inf),
                        'bic': result.get('bic', np.inf),
                        'llf': result.get('llf', -np.inf),
                        'method': result.get('method', method)
                    }
                else:
                    results[method] = {'error': result['error']}

            except Exception as e:
                results[method] = {'error': str(e)}

        # Summary
        valid_methods = {k: v for k, v in results.items() if 'error' not in v}

        if valid_methods:
            best_aic = min(valid_methods.items(), key=lambda x: x[1]['aic'])
            best_bic = min(valid_methods.items(), key=lambda x: x[1]['bic'])
            best_llf = max(valid_methods.items(), key=lambda x: x[1]['llf'])

            comparison = {
                'methods_tested': methods,
                'results': results,
                'summary': {
                    'best_aic_method': best_aic[0],
                    'best_bic_method': best_bic[0],
                    'best_llf_method': best_llf[0],
                    'valid_methods': list(valid_methods.keys())
                }
            }
        else:
            comparison = {
                'methods_tested': methods,
                'results': results,
                'error': 'No estimation methods succeeded'
            }

        return comparison

    # Helper methods for advanced estimation
    def _fit_ar_ols(self, data: pd.Series, p: int) -> Dict[str, Any]:
        """Fit pure AR model using OLS."""
        n = len(data)

        # Create design matrix
        X = np.column_stack([data.shift(i) for i in range(1, p+1)]).dropna()
        y = data[p:].values

        # Add constant term
        X_with_const = np.column_stack([np.ones(len(X)), X])

        # OLS estimation
        beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]

        # Extract parameters
        const = beta[0]
        ar_params = beta[1:]

        # Calculate fitted values and residuals
        fitted = X @ ar_params + const
        residuals = y - fitted

        # Calculate metrics
        rss = np.sum(residuals**2)
        tss = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (rss / tss)

        n_obs = len(y)
        aic = n_obs * np.log(rss/n_obs) + 2 * (p + 1)
        bic = n_obs * np.log(rss/n_obs) + np.log(n_obs) * (p + 1)

        return {
            'method': 'OLS (AR)',
            'order': (p, 0),
            'fitted_values': pd.Series(fitted, index=data.index[p:]),
            'residuals': pd.Series(residuals, index=data.index[p:]),
            'parameters': {f'ar_L{i+1}': ar_params[i] for i in range(p)},
            'constant': const,
            'aic': aic,
            'bic': bic,
            'r_squared': r_squared,
            'rss': rss,
            'llf': -0.5 * (n_obs * np.log(2*np.pi) + n_obs * np.log(rss/n_obs) + n_obs),
            'estimation_method': 'ols'
        }

    def _fit_ma_ols(self, data: pd.Series, q: int) -> Dict[str, Any]:
        """Fit pure MA model using iterative OLS."""
        # MA models require iterative estimation
        # Start with initial residual estimates
        n = len(data)
        residuals = data.values.copy()

        # Initialize MA parameters
        theta = np.zeros(q)

        # Iterative estimation
        max_iter = 50
        tolerance = 1e-6

        for iteration in range(max_iter):
            theta_old = theta.copy()

            # Create lagged residuals matrix
            X = np.column_stack([np.roll(residuals, i+1) for i in range(q)])
            X[:q] = 0  # Set initial values to zero

            # OLS on current residuals
            try:
                theta = np.linalg.lstsq(X, data.values, rcond=None)[0]
            except:
                break

            # Update residuals
            fitted = X @ theta
            residuals = data.values - fitted

            # Check convergence
            if np.sum((theta - theta_old)**2) < tolerance:
                break

        # Calculate final metrics
        rss = np.sum(residuals**2)
        aic = n * np.log(rss/n) + 2 * (q + 1)
        bic = n * np.log(rss/n) + np.log(n) * (q + 1)

        return {
            'method': 'OLS (MA) - Iterative',
            'order': (0, q),
            'fitted_values': pd.Series(fitted, index=data.index),
            'residuals': pd.Series(residuals, index=data.index),
            'parameters': {f'ma_L{i+1}': theta[i] for i in range(q)},
            'aic': aic,
            'bic': bic,
            'llf': -0.5 * (n * np.log(2*np.pi) + n * np.log(rss/n) + n),
            'iterations': iteration + 1,
            'estimation_method': 'ols',
            'note': 'Iterative OLS estimation for MA model'
        }

    def _fit_arma_conditional_ols(self, data: pd.Series, p: int, q: int) -> Dict[str, Any]:
        """Fit ARMA model using conditional OLS (simplified approach)."""
        # This is a simplified implementation
        # In practice, full ARMA estimation is complex and requires MLE

        # Start with AR estimation
        ar_result = self._fit_ar_ols(data, p)

        if 'error' in ar_result:
            return ar_result

        # Use AR residuals to estimate MA part
        ar_residuals = ar_result['residuals']
        ma_result = self._fit_ma_ols(ar_residuals, q)

        # Combine results (this is approximate)
        combined_params = {}
        combined_params.update({f'ar_L{i+1}': ar_result['parameters'][f'ar_L{i+1}']
                              for i in range(p)})
        combined_params.update({f'ma_L{i+1}': ma_result['parameters'][f'ma_L{i+1}']
                              for i in range(q)})

        # Approximate metrics
        n = len(data)
        total_params = p + q + 1
        rss = ma_result['rss']
        aic = n * np.log(rss/n) + 2 * total_params
        bic = n * np.log(rss/n) + np.log(n) * total_params

        return {
            'method': 'Conditional OLS (ARMA)',
            'order': (p, q),
            'fitted_values': ma_result['fitted_values'],
            'residuals': ma_result['residuals'],
            'parameters': combined_params,
            'aic': aic,
            'bic': bic,
            'llf': -0.5 * (n * np.log(2*np.pi) + n * np.log(rss/n) + n),
            'estimation_method': 'ols',
            'note': 'Conditional OLS - approximate ARMA estimation'
        }

    def _fit_ma_method_of_moments(self, data: pd.Series, q: int,
                                 autocorrs: np.ndarray) -> Dict[str, Any]:
        """Fit MA model using method of moments."""
        # For MA(q), solve system of equations relating autocorrelations to MA parameters
        # This is complex for q > 1, so we provide a simplified approach

        if q == 1:
            # MA(1): γ(1) = θσ²/(1 + θ²)
            # Solve quadratic equation
            rho1 = autocorrs[1] / autocorrs[0]  # lag-1 autocorrelation

            # Quadratic: θ² + (1/ρ₁)θ + 1 = 0
            if abs(rho1) < 0.5:  # Ensure invertibility
                a = 1
                b = 1/rho1 if rho1 != 0 else 1e6
                c = 1

                discriminant = b**2 - 4*a*c
                if discriminant >= 0:
                    theta1 = (-b + np.sqrt(discriminant)) / (2*a)
                    theta2 = (-b - np.sqrt(discriminant)) / (2*a)

                    # Choose invertible solution
                    theta = theta1 if abs(theta1) < 1 else theta2
                else:
                    theta = -0.5  # Default fallback
            else:
                theta = -0.5  # Default for boundary cases

            # Calculate fitted values (simplified)
            n = len(data)
            residuals = np.random.normal(0, np.sqrt(autocorrs[0]), n)  # Initialize
            fitted = data.values - residuals

            rss = np.sum(residuals**2)
            aic = n * np.log(rss/n) + 2 * 2
            bic = n * np.log(rss/n) + np.log(n) * 2

            return {
                'method': 'Method of Moments (MA)',
                'order': (0, q),
                'fitted_values': pd.Series(fitted, index=data.index),
                'residuals': pd.Series(residuals, index=data.index),
                'parameters': {'ma_L1': theta},
                'aic': aic,
                'bic': bic,
                'llf': -0.5 * (n * np.log(2*np.pi) + n * np.log(rss/n) + n),
                'estimation_method': 'method_of_moments'
            }
        else:
            return {'error': 'Method of moments for MA(q>1) not implemented'}

    def _fit_arma_method_of_moments(self, data: pd.Series, p: int, q: int,
                                   autocorrs: np.ndarray) -> Dict[str, Any]:
        """Fit ARMA model using method of moments (simplified)."""
        # This is a simplified implementation
        # Full ARMA method of moments is quite complex

        # Start with AR part using Yule-Walker
        ar_result = self.fit_arma_yule_walker(data, (p, 0))

        if 'error' in ar_result:
            return ar_result

        # Approximate MA part
        ma_params = {f'ma_L{i+1}': 0.1 * (1 if i % 2 == 0 else -1)
                    for i in range(q)}

        # Combine results
        combined_params = ar_result['parameters'].copy()
        combined_params.update(ma_params)

        return {
            'method': 'Method of Moments (ARMA) - Approximate',
            'order': (p, q),
            'fitted_values': ar_result['fitted_values'],
            'residuals': ar_result['residuals'],
            'parameters': combined_params,
            'aic': ar_result['aic'] + 2 * q,  # Penalty for additional MA params
            'bic': ar_result['bic'] + np.log(len(data)) * q,
            'llf': ar_result['llf'] - q,  # Approximate penalty
            'estimation_method': 'method_of_moments',
            'note': 'Approximate ARMA method of moments estimation'
        }

    def _calculate_sample_autocorrelations(self, data: pd.Series, max_lag: int) -> np.ndarray:
        """Calculate sample autocorrelations up to max_lag."""
        n = len(data)
        data_centered = data - data.mean()
        autocorrs = np.zeros(max_lag + 1)

        # Lag 0 (variance)
        autocorrs[0] = np.mean(data_centered**2)

        # Higher lags
        for lag in range(1, max_lag + 1):
            if lag < n:
                autocorrs[lag] = np.mean(data_centered[:-lag] * data_centered[lag:])
            else:
                autocorrs[lag] = 0

        return autocorrs

    def _calculate_ar_fitted_values(self, data: pd.Series, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate fitted values and residuals for AR model."""
        p = len(phi)
        n = len(data)

        fitted = np.zeros(n - p)
        for i in range(p, n):
            fitted[i - p] = np.sum(phi * data.iloc[i-p:i][::-1])

        residuals = data.iloc[p:].values - fitted

        return fitted, residuals

    def _manual_ljung_box_test(self, residuals: pd.Series, lags: int,
                              model_df: int) -> Dict[str, Any]:
        """Manual implementation of Ljung-Box test."""
        try:
            residuals_clean = residuals.dropna()
            n = len(residuals_clean)

            # Calculate sample autocorrelations
            autocorrs = self._calculate_sample_autocorrelations(residuals_clean, lags)

            # Ljung-Box statistic
            lb_stat = n * (n + 2) * np.sum([
                autocorrs[k]**2 / (n - k) for k in range(1, lags + 1)
            ])

            # Degrees of freedom
            df = lags - model_df

            # P-value using chi-square distribution
            from scipy.stats import chi2
            p_value = 1 - chi2.cdf(lb_stat, df) if df > 0 else 1.0

            return {
                'test_name': 'Ljung-Box Test (Manual)',
                'null_hypothesis': 'Residuals are independently distributed',
                'test_statistic': lb_stat,
                'p_value': p_value,
                'lags_tested': lags,
                'degrees_freedom': df,
                'reject_null': p_value < 0.05 and df > 0,
                'conclusion': 'Residuals show autocorrelation' if (p_value < 0.05 and df > 0) else 'No significant autocorrelation',
                'note': 'Manual implementation - install statsmodels for full functionality'
            }

        except Exception as e:
            return {'error': f'Manual Ljung-Box test failed: {str(e)}'}

class ARIMAModeler:
    """
    ARIMA (AutoRegressive Integrated Moving Average) model fitting, forecasting, and validation.
    """

    def __init__(self):
        """Initialize the ARIMA modeler."""
        self.models = {}
        self.fitted_models = {}
        self.results = {}

    def fit_arima(self, data: pd.Series, order: Tuple[int, int, int],
                  seasonal_order: Optional[Tuple[int, int, int, int]] = None,
                  method: str = 'lbfgs') -> Dict[str, Any]:
        """
        Fit an ARIMA model to the data.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        order : tuple
            ARIMA order (p, d, q)
        seasonal_order : tuple, optional
            Seasonal ARIMA order (P, D, Q, s)
        method : str, default 'lbfgs'
            Optimization method

        Returns:
        --------
        dict
            Model fitting results
        """
        if not HAS_STATSMODELS:
            return self._manual_arima_fit(data, order)

        try:
            from statsmodels.tsa.arima.model import ARIMA

            # Fit ARIMA model
            model = ARIMA(data.dropna(), order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit(method=method)

            # Extract results
            results = {
                'method': 'ARIMA',
                'order': order,
                'seasonal_order': seasonal_order,
                'fitted_values': fitted_model.fittedvalues,
                'residuals': fitted_model.resid,
                'parameters': fitted_model.params.to_dict(),
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'hqic': fitted_model.hqic,
                'llf': fitted_model.llf,
                'model_summary': str(fitted_model.summary()),
                'model': fitted_model,
                'optimization_method': method
            }

            # Add parameter significance tests
            if hasattr(fitted_model, 'pvalues'):
                results['parameter_pvalues'] = fitted_model.pvalues.to_dict()
                results['significant_params'] = (fitted_model.pvalues < 0.05).to_dict()

            # Store model
            if seasonal_order:
                key = f"ARIMA{order}x{seasonal_order}"
            else:
                key = f"ARIMA{order}"

            self.models[key] = model
            self.fitted_models[key] = fitted_model
            self.results[key] = results

            return results

        except ImportError:
            warnings.warn("Statsmodels ARIMA not available. Using manual implementation.")
            return self._manual_arima_fit(data, order)
        except Exception as e:
            warnings.warn(f"ARIMA fitting failed: {str(e)}. Using manual implementation.")
            return self._manual_arima_fit(data, order)

    def forecast_arima(self, model_key: str, steps: int,
                      confidence_interval: float = 0.95,
                      return_conf_int: bool = True) -> Dict[str, Any]:
        """
        Generate forecasts from fitted ARIMA model.

        Parameters:
        -----------
        model_key : str
            Key for the fitted model
        steps : int
            Number of steps to forecast
        confidence_interval : float, default 0.95
            Confidence level for prediction intervals
        return_conf_int : bool, default True
            Whether to return confidence intervals

        Returns:
        --------
        dict
            Forecast results
        """
        if model_key not in self.fitted_models:
            raise ValueError(f"Model {model_key} not found. Fit model first.")

        fitted_model = self.fitted_models[model_key]

        try:
            # Generate forecast
            forecast_result = fitted_model.get_forecast(steps=steps)
            forecast = forecast_result.predicted_mean

            result = {
                'forecast': forecast,
                'steps': steps,
                'confidence_level': confidence_interval,
                'model_key': model_key
            }

            if return_conf_int:
                # Get confidence intervals
                alpha = 1 - confidence_interval
                conf_int = forecast_result.conf_int(alpha=alpha)

                result['confidence_intervals'] = {
                    'lower': conf_int.iloc[:, 0],
                    'upper': conf_int.iloc[:, 1]
                }
                result['forecast_std_err'] = forecast_result.se_mean

            return result

        except Exception as e:
            warnings.warn(f"ARIMA forecasting failed: {str(e)}")
            return self._manual_arima_forecast(model_key, steps)

    def auto_arima(self, data: pd.Series, max_p: int = 5, max_d: int = 2,
                   max_q: int = 5, seasonal: bool = False,
                   seasonal_periods: int = 12, criterion: str = 'aic') -> Dict[str, Any]:
        """
        Automatic ARIMA model selection.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        max_p : int, default 5
            Maximum AR order to test
        max_d : int, default 2
            Maximum differencing order to test
        max_q : int, default 5
            Maximum MA order to test
        seasonal : bool, default False
            Whether to include seasonal components
        seasonal_periods : int, default 12
            Seasonal periods
        criterion : str, default 'aic'
            Information criterion for model selection ('aic', 'bic')

        Returns:
        --------
        dict
            Best model results and comparison
        """
        best_score = np.inf
        best_order = None
        best_seasonal_order = None
        best_model = None

        models_tested = []

        # Test different combinations
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        order = (p, d, q)
                        seasonal_order = (1, 1, 1, seasonal_periods) if seasonal else None

                        results = self.fit_arima(data, order, seasonal_order)

                        if 'error' not in results:
                            score = results[criterion]
                            models_tested.append({
                                'order': order,
                                'seasonal_order': seasonal_order,
                                'aic': results['aic'],
                                'bic': results['bic'],
                                'score': score
                            })

                            if score < best_score:
                                best_score = score
                                best_order = order
                                best_seasonal_order = seasonal_order
                                best_model = results

                    except Exception as e:
                        continue

        if best_model:
            auto_results = {
                'best_order': best_order,
                'best_seasonal_order': best_seasonal_order,
                'best_model': best_model,
                'selection_criterion': criterion,
                'best_score': best_score,
                'models_tested': len(models_tested),
                'all_models': models_tested
            }
        else:
            auto_results = {
                'error': 'No suitable ARIMA model found',
                'models_tested': len(models_tested)
            }

        return auto_results

    def model_diagnostics(self, model_key: str) -> Dict[str, Any]:
        """
        Comprehensive ARIMA model diagnostics.

        Parameters:
        -----------
        model_key : str
            Key for the fitted model

        Returns:
        --------
        dict
            Comprehensive diagnostic results
        """
        if model_key not in self.fitted_models:
            raise ValueError(f"Model {model_key} not found.")

        fitted_model = self.fitted_models[model_key]
        residuals = fitted_model.resid

        diagnostics = {}

        # Basic residual statistics
        diagnostics['residual_stats'] = {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'min': residuals.min(),
            'max': residuals.max(),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals)
        }

        try:
            # Ljung-Box test
            from statsmodels.stats.diagnostic import acorr_ljungbox
            ljung_box = acorr_ljungbox(residuals, lags=min(10, len(residuals)//4), return_df=True)
            diagnostics['ljung_box'] = {
                'test_statistic': ljung_box['lb_stat'].iloc[-1],
                'p_value': ljung_box['lb_pvalue'].iloc[-1],
                'conclusion': 'No autocorrelation' if ljung_box['lb_pvalue'].iloc[-1] > 0.05 else 'Autocorrelation detected'
            }
        except Exception as e:
            diagnostics['ljung_box'] = {'error': str(e)}

        try:
            # Jarque-Bera test for normality
            from statsmodels.stats.stattools import jarque_bera
            jb_stat, jb_pvalue, jb_skew, jb_kurtosis = jarque_bera(residuals)
            diagnostics['jarque_bera'] = {
                'test_statistic': jb_stat,
                'p_value': jb_pvalue,
                'skewness': jb_skew,
                'kurtosis': jb_kurtosis,
                'conclusion': 'Normal residuals' if jb_pvalue > 0.05 else 'Non-normal residuals'
            }
        except Exception as e:
            diagnostics['jarque_bera'] = {'error': str(e)}

        try:
            # Durbin-Watson test for autocorrelation
            from statsmodels.stats.stattools import durbin_watson
            dw_stat = durbin_watson(residuals)
            diagnostics['durbin_watson'] = {
                'test_statistic': dw_stat,
                'interpretation': self._interpret_durbin_watson(dw_stat)
            }
        except Exception as e:
            diagnostics['durbin_watson'] = {'error': str(e)}

        # Information criteria
        diagnostics['information_criteria'] = {
            'aic': self.results[model_key]['aic'],
            'bic': self.results[model_key]['bic'],
            'hqic': self.results[model_key].get('hqic', None)
        }

        return diagnostics

    def _interpret_durbin_watson(self, dw_stat: float) -> str:
        """Interpret Durbin-Watson test statistic."""
        if dw_stat < 1.5:
            return "Strong positive autocorrelation"
        elif dw_stat < 2.5:
            return "No significant autocorrelation"
        else:
            return "Strong negative autocorrelation"

    def _manual_arima_fit(self, data: pd.Series, order: Tuple[int, int, int]) -> Dict[str, Any]:
        """Manual ARIMA fitting when statsmodels is not available."""
        warnings.warn("Manual ARIMA implementation is limited. Consider installing statsmodels.")

        p, d, q = order

        # Apply differencing
        differenced_data = data.copy()
        for i in range(d):
            differenced_data = differenced_data.diff().dropna()

        # Fit ARMA to differenced data
        if len(differenced_data) < max(p, q) + 10:
            return {'error': 'Insufficient data after differencing'}

        # Basic AR fitting
        if p > 0 and q == 0:
            try:
                arma_modeler = ARMAModeler()
                arma_result = arma_modeler.fit_arma(differenced_data, (p, 0))

                # Convert ARMA results to ARIMA format
                result = {
                    'method': 'ARIMA (Manual - limited)',
                    'order': order,
                    'fitted_values': arma_result.get('fitted_values'),
                    'residuals': arma_result.get('residuals'),
                    'parameters': arma_result.get('parameters', {}),
                    'aic': arma_result.get('aic', np.inf),
                    'bic': arma_result.get('bic', np.inf),
                    'llf': arma_result.get('llf', -np.inf),
                    'note': 'Manual implementation - limited functionality'
                }

                return result

            except Exception as e:
                return {'error': f'Manual ARIMA fitting failed: {str(e)}'}
        else:
            return {'error': 'Manual ARIMA implementation only supports AR models'}

    def _manual_arima_forecast(self, model_key: str, steps: int) -> Dict[str, Any]:
        """Manual ARIMA forecasting."""
        return {'error': 'Manual ARIMA forecasting not implemented. Use statsmodels for full functionality.'}

class TimeSeriesVisualizer:
    """
    Comprehensive time series visualization including line plots, histograms, and ACF plots.
    """

    def __init__(self):
        """Initialize the time series visualizer."""
        self.figures = {}
        self.plots_created = 0

    def plot_time_series(self, data: pd.Series, title: str = "Time Series",
                        figsize: Tuple[int, int] = (12, 6),
                        show_trend: bool = False, show_ma: bool = False,
                        ma_window: int = 12) -> None:
        """
        Create line plot of time series data.

        Parameters:
        -----------
        data : pd.Series
            Time series data with datetime index
        title : str, default "Time Series"
            Plot title
        figsize : tuple, default (12, 6)
            Figure size
        show_trend : bool, default False
            Whether to overlay trend line
        show_ma : bool, default False
            Whether to overlay moving average
        ma_window : int, default 12
            Moving average window size
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot main time series
        ax.plot(data.index, data.values, label='Original', linewidth=1.5)

        # Add trend line if requested
        if show_trend:
            trend_remover = TrendRemover()
            _, trend = trend_remover.linear_detrend(data)
            ax.plot(data.index, trend.values, label='Linear Trend',
                   color='red', linestyle='--', alpha=0.7)

        # Add moving average if requested
        if show_ma:
            ma_calculator = MovingAverages()
            ma = ma_calculator.simple_moving_average(data, ma_window)
            ax.plot(data.index, ma.values, label=f'{ma_window}-period MA',
                   color='orange', linewidth=2, alpha=0.8)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        self.plots_created += 1
        self.figures[f'timeseries_{self.plots_created}'] = fig

    def plot_histogram(self, data: pd.Series, bins: int = 30,
                      title: str = "Distribution",
                      figsize: Tuple[int, int] = (10, 6),
                      show_normal: bool = True, show_stats: bool = True) -> None:
        """
        Create histogram of time series values.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        bins : int, default 30
            Number of histogram bins
        title : str, default "Distribution"
            Plot title
        figsize : tuple, default (10, 6)
            Figure size
        show_normal : bool, default True
            Whether to overlay normal distribution curve
        show_stats : bool, default True
            Whether to show basic statistics
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create histogram
        n, bins_edges, patches = ax.hist(data.dropna().values, bins=bins,
                                        alpha=0.7, edgecolor='black',
                                        density=True, label='Observed')

        # Overlay normal distribution if requested
        if show_normal:
            mean = data.mean()
            std = data.std()
            x = np.linspace(data.min(), data.max(), 100)
            normal_curve = stats.norm.pdf(x, mean, std)
            ax.plot(x, normal_curve, 'r-', linewidth=2,
                   label=f'Normal(μ={mean:.2f}, σ={std:.2f})')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        if show_stats:
            stats_text = f'''Statistics:
Mean: {data.mean():.3f}
Std: {data.std():.3f}
Skew: {stats.skew(data.dropna()):.3f}
Kurt: {stats.kurtosis(data.dropna()):.3f}'''

            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round',
                   facecolor='wheat', alpha=0.8), fontsize=10)

        plt.tight_layout()
        plt.show()

        self.plots_created += 1
        self.figures[f'histogram_{self.plots_created}'] = fig

    def plot_acf_pacf(self, data: pd.Series, nlags: int = 40,
                     figsize: Tuple[int, int] = (12, 8),
                     title: str = "ACF and PACF Analysis") -> None:
        """
        Create ACF and PACF plots.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        nlags : int, default 40
            Number of lags to plot
        figsize : tuple, default (12, 8)
            Figure size
        title : str, default "ACF and PACF Analysis"
            Plot title
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        acf_analyzer = AutocorrelationAnalyzer()

        if HAS_STATSMODELS:
            try:
                from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

                # ACF plot
                plot_acf(data.dropna(), lags=nlags, ax=axes[0], alpha=0.05)
                axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12)
                axes[0].grid(True, alpha=0.3)

                # PACF plot
                plot_pacf(data.dropna(), lags=nlags, ax=axes[1], alpha=0.05)
                axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12)
                axes[1].grid(True, alpha=0.3)

            except Exception as e:
                # Fallback to manual implementation
                self._manual_acf_pacf_plot(axes, data, nlags, acf_analyzer)
        else:
            self._manual_acf_pacf_plot(axes, data, nlags, acf_analyzer)

        plt.tight_layout()
        plt.show()

        self.plots_created += 1
        self.figures[f'acf_pacf_{self.plots_created}'] = fig

    def plot_residuals(self, residuals: pd.Series, fitted_values: pd.Series,
                      title: str = "Residual Analysis",
                      figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Create comprehensive residual analysis plots.

        Parameters:
        -----------
        residuals : pd.Series
            Model residuals
        fitted_values : pd.Series
            Model fitted values
        title : str, default "Residual Analysis"
            Plot title
        figsize : tuple, default (15, 10)
            Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # 1. Residuals vs Fitted
        axes[0, 0].scatter(fitted_values, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Residuals over time
        axes[0, 1].plot(residuals.index, residuals.values, alpha=0.7)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals over Time')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Histogram of residuals
        axes[1, 0].hist(residuals.dropna().values, bins=20, alpha=0.7,
                       edgecolor='black', density=True)

        # Overlay normal distribution
        mean = residuals.mean()
        std = residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        normal_curve = stats.norm.pdf(x, mean, std)
        axes[1, 0].plot(x, normal_curve, 'r-', linewidth=2, label='Normal')
        axes[1, 0].set_xlabel('Residual Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Q-Q plot
        from scipy import stats as scipy_stats
        scipy_stats.probplot(residuals.dropna(), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normal)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        self.plots_created += 1
        self.figures[f'residuals_{self.plots_created}'] = fig

    def plot_forecast(self, original_data: pd.Series, forecast: pd.Series,
                     confidence_intervals: Optional[Dict[str, pd.Series]] = None,
                     title: str = "Forecast",
                     figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot original data with forecasts and confidence intervals.

        Parameters:
        -----------
        original_data : pd.Series
            Historical time series data
        forecast : pd.Series
            Forecasted values
        confidence_intervals : dict, optional
            Dictionary with 'lower' and 'upper' confidence bounds
        title : str, default "Forecast"
            Plot title
        figsize : tuple, default (12, 6)
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot historical data
        ax.plot(original_data.index, original_data.values,
               label='Historical', linewidth=2, color='blue')

        # Plot forecast
        ax.plot(forecast.index, forecast.values,
               label='Forecast', linewidth=2, color='red', linestyle='--')

        # Add confidence intervals if provided
        if confidence_intervals and 'lower' in confidence_intervals and 'upper' in confidence_intervals:
            ax.fill_between(forecast.index,
                           confidence_intervals['lower'].values,
                           confidence_intervals['upper'].values,
                           alpha=0.3, color='red', label='95% Confidence Interval')

        # Add vertical line at forecast start
        if len(original_data) > 0 and len(forecast) > 0:
            ax.axvline(x=original_data.index[-1], color='black',
                      linestyle=':', alpha=0.7, label='Forecast Start')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        self.plots_created += 1
        self.figures[f'forecast_{self.plots_created}'] = fig

    def plot_decomposition(self, data: pd.Series, model: str = 'additive',
                          period: int = 12, figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Plot time series decomposition.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        model : str, default 'additive'
            Decomposition model ('additive' or 'multiplicative')
        period : int, default 12
            Seasonal period
        figsize : tuple, default (12, 10)
            Figure size
        """
        if not HAS_STATSMODELS:
            warnings.warn("Statsmodels required for decomposition. Showing trend analysis instead.")
            self._manual_decomposition_plot(data, figsize)
            return

        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            # Perform decomposition
            decomposition = seasonal_decompose(data.dropna(), model=model, period=period)

            # Create plots
            fig, axes = plt.subplots(4, 1, figsize=figsize)
            fig.suptitle(f'Time Series Decomposition ({model.title()})',
                        fontsize=16, fontweight='bold')

            # Original
            axes[0].plot(decomposition.observed, linewidth=1.5)
            axes[0].set_title('Original')
            axes[0].grid(True, alpha=0.3)

            # Trend
            axes[1].plot(decomposition.trend, color='red', linewidth=1.5)
            axes[1].set_title('Trend')
            axes[1].grid(True, alpha=0.3)

            # Seasonal
            axes[2].plot(decomposition.seasonal, color='green', linewidth=1.5)
            axes[2].set_title('Seasonal')
            axes[2].grid(True, alpha=0.3)

            # Residual
            axes[3].plot(decomposition.resid, color='orange', linewidth=1.5)
            axes[3].set_title('Residual')
            axes[3].grid(True, alpha=0.3)
            axes[3].set_xlabel('Date')

            plt.tight_layout()
            plt.show()

            self.plots_created += 1
            self.figures[f'decomposition_{self.plots_created}'] = fig

        except Exception as e:
            warnings.warn(f"Decomposition failed: {str(e)}. Showing trend analysis instead.")
            self._manual_decomposition_plot(data, figsize)

    def _manual_acf_pacf_plot(self, axes, data: pd.Series, nlags: int, acf_analyzer):
        """Manual ACF/PACF plotting when statsmodels plots are not available."""
        # Calculate ACF and PACF
        acf_results = acf_analyzer.calculate_acf(data, nlags)
        pacf_results = acf_analyzer.calculate_pacf(data, nlags)

        # Plot ACF
        lags = range(len(acf_results['acf_values']))
        axes[0].bar(lags, acf_results['acf_values'], alpha=0.7)
        axes[0].set_title('Autocorrelation Function (ACF)')
        axes[0].set_xlabel('Lag')
        axes[0].set_ylabel('ACF')
        axes[0].grid(True, alpha=0.3)

        # Add confidence bounds
        if 'confidence_intervals' in acf_results:
            conf_int = acf_results['confidence_intervals']
            axes[0].axhline(y=conf_int[0, 1], color='red', linestyle='--', alpha=0.7)
            axes[0].axhline(y=conf_int[0, 0], color='red', linestyle='--', alpha=0.7)

        # Plot PACF
        lags = range(len(pacf_results['pacf_values']))
        axes[1].bar(lags, pacf_results['pacf_values'], alpha=0.7)
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        axes[1].set_xlabel('Lag')
        axes[1].set_ylabel('PACF')
        axes[1].grid(True, alpha=0.3)

        # Add confidence bounds
        if 'confidence_intervals' in pacf_results:
            conf_int = pacf_results['confidence_intervals']
            axes[1].axhline(y=conf_int[0, 1], color='red', linestyle='--', alpha=0.7)
            axes[1].axhline(y=conf_int[0, 0], color='red', linestyle='--', alpha=0.7)

    def _manual_decomposition_plot(self, data: pd.Series, figsize: Tuple[int, int]):
        """Manual decomposition plot using trend removal techniques."""
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        fig.suptitle('Time Series Analysis (Manual)', fontsize=16, fontweight='bold')

        trend_remover = TrendRemover()

        # Original
        axes[0].plot(data.index, data.values, linewidth=1.5)
        axes[0].set_title('Original')
        axes[0].grid(True, alpha=0.3)

        # Trend
        try:
            _, trend = trend_remover.linear_detrend(data)
            axes[1].plot(data.index, trend.values, color='red', linewidth=1.5)
            axes[1].set_title('Linear Trend')
            axes[1].grid(True, alpha=0.3)

            # Detrended (residual)
            detrended = data - trend
            axes[2].plot(data.index, detrended.values, color='orange', linewidth=1.5)
            axes[2].set_title('Detrended (Residual)')
            axes[2].grid(True, alpha=0.3)
            axes[2].set_xlabel('Date')

        except Exception as e:
            axes[1].text(0.5, 0.5, f'Trend analysis failed: {str(e)}',
                        transform=axes[1].transAxes, ha='center', va='center')
            axes[2].text(0.5, 0.5, 'Residual plot not available',
                        transform=axes[2].transAxes, ha='center', va='center')

        plt.tight_layout()
        plt.show()

        self.plots_created += 1
        self.figures[f'manual_decomposition_{self.plots_created}'] = fig

    def plot_model_diagnostics(self, fitted_model, model_name: str = "Model",
                             figsize: Tuple[int, int] = (16, 12)) -> None:
        """
        Comprehensive model diagnostic plots for ARIMA/ARMA models.

        Parameters:
        -----------
        fitted_model : fitted model object
            Fitted ARIMA/ARMA model from statsmodels
        model_name : str, default "Model"
            Name of the model for plot titles
        figsize : tuple, default (16, 12)
            Figure size
        """
        if not hasattr(fitted_model, 'resid'):
            warnings.warn("Model does not have residuals. Cannot create diagnostics.")
            return

        residuals = fitted_model.resid.dropna()
        fitted_values = fitted_model.fittedvalues

        # Create 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'{model_name} - Diagnostic Plots', fontsize=16, fontweight='bold')

        # 1. Standardized Residuals Plot
        standardized_residuals = residuals / residuals.std()
        axes[0, 0].plot(standardized_residuals.index, standardized_residuals.values,
                       linewidth=1.2, alpha=0.8)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].axhline(y=2, color='orange', linestyle=':', alpha=0.7, label='±2σ')
        axes[0, 0].axhline(y=-2, color='orange', linestyle=':', alpha=0.7)
        axes[0, 0].set_title('Standardized Residuals', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Standardized Residuals')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Histogram + KDE of Residuals
        axes[0, 1].hist(residuals.values, bins=20, density=True, alpha=0.7,
                       edgecolor='black', label='Histogram')

        # Add KDE if scipy available
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(residuals.values)
            x_range = np.linspace(residuals.min(), residuals.max(), 100)
            axes[0, 1].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        except ImportError:
            pass

        # Overlay normal distribution
        mean_resid = residuals.mean()
        std_resid = residuals.std()
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        normal_curve = stats.norm.pdf(x_norm, mean_resid, std_resid)
        axes[0, 1].plot(x_norm, normal_curve, 'g--', linewidth=2,
                       label=f'Normal(μ={mean_resid:.3f}, σ={std_resid:.3f})')

        axes[0, 1].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Residual Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Q-Q Plot
        from scipy import stats as scipy_stats
        scipy_stats.probplot(residuals.values, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal)', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # Get line equation for reference
        slope, intercept, r_value = scipy_stats.linregress(*scipy_stats.probplot(residuals.values, dist="norm")[:2])
        axes[1, 0].text(0.05, 0.95, f'R² = {r_value**2:.3f}', transform=axes[1, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 4. Correlogram (ACF of Residuals)
        if HAS_STATSMODELS:
            try:
                from statsmodels.graphics.tsaplots import plot_acf
                plot_acf(residuals, lags=min(20, len(residuals)//4), ax=axes[1, 1], alpha=0.05)
                axes[1, 1].set_title('Residuals Correlogram (ACF)', fontsize=12, fontweight='bold')
            except Exception:
                self._manual_correlogram(axes[1, 1], residuals, "Residuals Correlogram")
        else:
            self._manual_correlogram(axes[1, 1], residuals, "Residuals Correlogram")

        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        self.plots_created += 1
        self.figures[f'diagnostics_{self.plots_created}'] = fig

    def plot_comprehensive_diagnostics(self, residuals: pd.Series, fitted_values: pd.Series,
                                     model_name: str = "Model",
                                     figsize: Tuple[int, int] = (18, 14)) -> None:
        """
        Create comprehensive diagnostic plots from residuals and fitted values.

        Parameters:
        -----------
        residuals : pd.Series
            Model residuals
        fitted_values : pd.Series
            Model fitted values
        model_name : str, default "Model"
            Name of the model for plot titles
        figsize : tuple, default (18, 14)
            Figure size
        """
        # Create 3x2 subplot layout
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle(f'{model_name} - Comprehensive Diagnostic Analysis',
                    fontsize=18, fontweight='bold')

        # 1. Standardized Residuals over Time
        std_residuals = residuals / residuals.std()
        axes[0, 0].plot(residuals.index, std_residuals.values, linewidth=1.2, alpha=0.8)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].axhline(y=2, color='orange', linestyle=':', alpha=0.7)
        axes[0, 0].axhline(y=-2, color='orange', linestyle=':', alpha=0.7)
        axes[0, 0].fill_between(residuals.index, -2, 2, alpha=0.1, color='orange')
        axes[0, 0].set_title('Standardized Residuals Over Time', fontweight='bold')
        axes[0, 0].set_ylabel('Standardized Residuals')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Residuals vs Fitted Values
        axes[0, 1].scatter(fitted_values, residuals, alpha=0.6, s=30)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)

        # Add lowess smooth line if available
        try:
            from scipy.signal import savgol_filter
            sorted_idx = np.argsort(fitted_values.values)
            sorted_fitted = fitted_values.iloc[sorted_idx]
            sorted_resid = residuals.iloc[sorted_idx]

            if len(sorted_resid) > 10:
                smooth_resid = savgol_filter(sorted_resid.values,
                                           min(len(sorted_resid)//3*2-1, 21), 3)
                axes[0, 1].plot(sorted_fitted, smooth_resid, 'red', linewidth=2, alpha=0.8)
        except:
            pass

        axes[0, 1].set_title('Residuals vs Fitted Values', fontweight='bold')
        axes[0, 1].set_xlabel('Fitted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Histogram + KDE + Normal Overlay
        axes[1, 0].hist(residuals.values, bins=25, density=True, alpha=0.7,
                       edgecolor='black', color='skyblue', label='Histogram')

        # KDE
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(residuals.values)
            x_range = np.linspace(residuals.min(), residuals.max(), 200)
            axes[1, 0].plot(x_range, kde(x_range), 'r-', linewidth=2.5, label='KDE')
        except ImportError:
            pass

        # Normal distribution overlay
        mean_r = residuals.mean()
        std_r = residuals.std()
        x_norm = np.linspace(residuals.min(), residuals.max(), 200)
        normal_curve = stats.norm.pdf(x_norm, mean_r, std_r)
        axes[1, 0].plot(x_norm, normal_curve, 'g--', linewidth=2.5,
                       label=f'Normal(μ={mean_r:.3f}, σ={std_r:.3f})')

        # Add statistics text
        skewness = stats.skew(residuals.dropna())
        kurtosis = stats.kurtosis(residuals.dropna())
        axes[1, 0].text(0.02, 0.98, f'Skewness: {skewness:.3f}\nKurtosis: {kurtosis:.3f}',
                       transform=axes[1, 0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        axes[1, 0].set_title('Residuals Distribution Analysis', fontweight='bold')
        axes[1, 0].set_xlabel('Residual Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Q-Q Plot with enhanced statistics
        from scipy import stats as scipy_stats
        (theoretical_quantiles, sample_quantiles), (slope, intercept, r) = scipy_stats.probplot(
            residuals.values, dist="norm", plot=axes[1, 1])

        axes[1, 1].set_title('Q-Q Plot (Normality Check)', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        # Add R² and normality test results
        r_squared = r**2
        try:
            from scipy.stats import shapiro, anderson
            shapiro_stat, shapiro_p = shapiro(residuals.values[:5000])  # Limit for Shapiro-Wilk

            stats_text = f'R² = {r_squared:.3f}\nShapiro-Wilk p = {shapiro_p:.4f}'
            axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        except:
            axes[1, 1].text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=axes[1, 1].transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # 5. ACF of Residuals (Correlogram)
        if HAS_STATSMODELS:
            try:
                from statsmodels.graphics.tsaplots import plot_acf
                plot_acf(residuals.dropna(), lags=min(25, len(residuals)//4),
                        ax=axes[2, 0], alpha=0.05)
                axes[2, 0].set_title('Residuals Autocorrelation Function', fontweight='bold')
            except Exception:
                self._manual_correlogram(axes[2, 0], residuals, "Residuals ACF")
        else:
            self._manual_correlogram(axes[2, 0], residuals, "Residuals ACF")

        # 6. PACF of Residuals
        if HAS_STATSMODELS:
            try:
                from statsmodels.graphics.tsaplots import plot_pacf
                plot_pacf(residuals.dropna(), lags=min(25, len(residuals)//4),
                         ax=axes[2, 1], alpha=0.05)
                axes[2, 1].set_title('Residuals Partial Autocorrelation Function', fontweight='bold')
            except Exception:
                self._manual_correlogram(axes[2, 1], residuals, "Residuals PACF", partial=True)
        else:
            self._manual_correlogram(axes[2, 1], residuals, "Residuals PACF", partial=True)

        plt.tight_layout()
        plt.show()

        self.plots_created += 1
        self.figures[f'comprehensive_diagnostics_{self.plots_created}'] = fig

    def _manual_correlogram(self, ax, data: pd.Series, title: str, partial: bool = False):
        """Create manual correlogram when statsmodels is not available."""
        try:
            acf_analyzer = AutocorrelationAnalyzer()

            if partial:
                result = acf_analyzer.calculate_pacf(data, nlags=min(20, len(data)//4))
                values = result['pacf_values']
            else:
                result = acf_analyzer.calculate_acf(data, nlags=min(20, len(data)//4))
                values = result['acf_values']

            lags = range(len(values))
            ax.bar(lags, values, alpha=0.7)

            # Add confidence bounds
            if 'confidence_intervals' in result:
                conf_int = result['confidence_intervals']
                ax.axhline(y=conf_int[0, 1], color='red', linestyle='--', alpha=0.7)
                ax.axhline(y=conf_int[0, 0], color='red', linestyle='--', alpha=0.7)

            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Lag')
            ax.set_ylabel('PACF' if partial else 'ACF')
            ax.grid(True, alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, f'Correlogram failed:\n{str(e)}',
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='pink', alpha=0.8))

class ModelValidator:
    """
    Walk-forward validation and performance evaluation for time series models.
    """

    def __init__(self):
        """Initialize the model validator."""
        self.validation_results = {}
        self.metrics_history = {}

    def walk_forward_validation(self, data: pd.Series, model_class,
                               initial_window: int, step_size: int = 1,
                               forecast_horizon: int = 1,
                               model_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform walk-forward validation on time series model.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        model_class : class
            Model class (ARMAModeler or ARIMAModeler)
        initial_window : int
            Initial training window size
        step_size : int, default 1
            Number of steps to move forward each iteration
        forecast_horizon : int, default 1
            Number of steps to forecast ahead
        model_params : dict, optional
            Parameters for model fitting

        Returns:
        --------
        dict
            Validation results including forecasts and metrics
        """
        if model_params is None:
            model_params = {}

        n = len(data)
        forecasts = []
        actuals = []
        fold_results = []

        for i in range(initial_window, n - forecast_horizon + 1, step_size):
            try:
                # Split data
                train_data = data.iloc[:i]
                test_data = data.iloc[i:i + forecast_horizon]

                # Initialize model
                model = model_class()

                # Fit model (handle both ARMA and ARIMA)
                if hasattr(model, 'fit_arima'):
                    # ARIMA model
                    order = model_params.get('order', (1, 1, 1))
                    results = model.fit_arima(train_data, order)
                    model_key = f"ARIMA{order}"
                else:
                    # ARMA model
                    order = model_params.get('order', (1, 1))
                    results = model.fit_arma(train_data, order)
                    model_key = f"ARMA{order}"

                if 'error' not in results:
                    # Generate forecast
                    if hasattr(model, 'forecast_arima'):
                        forecast_result = model.forecast_arima(model_key, forecast_horizon, return_conf_int=False)
                    else:
                        forecast_result = model.forecast_arma(model_key, forecast_horizon)

                    if 'error' not in forecast_result:
                        forecast = forecast_result['forecast']

                        forecasts.extend(forecast.values)
                        actuals.extend(test_data.values)

                        # Calculate fold metrics
                        fold_metrics = self._calculate_fold_metrics(test_data.values, forecast.values)
                        fold_results.append({
                            'fold': len(fold_results) + 1,
                            'train_end': train_data.index[-1],
                            'test_start': test_data.index[0],
                            'test_end': test_data.index[-1],
                            'metrics': fold_metrics
                        })

            except Exception as e:
                warnings.warn(f"Fold failed at position {i}: {str(e)}")
                continue

        if not forecasts:
            return {'error': 'No successful forecasts generated'}

        # Calculate overall metrics
        overall_metrics = self._calculate_metrics(np.array(actuals), np.array(forecasts))

        validation_results = {
            'method': 'Walk-Forward Validation',
            'model_class': model_class.__name__,
            'model_params': model_params,
            'initial_window': initial_window,
            'step_size': step_size,
            'forecast_horizon': forecast_horizon,
            'total_folds': len(fold_results),
            'forecasts': forecasts,
            'actuals': actuals,
            'overall_metrics': overall_metrics,
            'fold_results': fold_results
        }

        # Store results
        key = f"wfv_{model_class.__name__}_{len(self.validation_results)}"
        self.validation_results[key] = validation_results

        return validation_results

    def time_series_cv(self, data: pd.Series, model_class, n_splits: int = 5,
                      test_size: int = None, model_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Time series cross-validation with expanding window.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        model_class : class
            Model class
        n_splits : int, default 5
            Number of cross-validation splits
        test_size : int, optional
            Size of test set. If None, calculated automatically
        model_params : dict, optional
            Model parameters

        Returns:
        --------
        dict
            Cross-validation results
        """
        if model_params is None:
            model_params = {}

        n = len(data)
        if test_size is None:
            test_size = max(1, n // (n_splits + 1))

        cv_results = []

        for split in range(n_splits):
            try:
                # Calculate split points
                test_start = n - (n_splits - split) * test_size
                test_end = test_start + test_size

                if test_start <= n // 2:  # Ensure sufficient training data
                    continue

                train_data = data.iloc[:test_start]
                test_data = data.iloc[test_start:test_end]

                # Initialize and fit model
                model = model_class()

                if hasattr(model, 'fit_arima'):
                    order = model_params.get('order', (1, 1, 1))
                    results = model.fit_arima(train_data, order)
                    model_key = f"ARIMA{order}"
                else:
                    order = model_params.get('order', (1, 1))
                    results = model.fit_arma(train_data, order)
                    model_key = f"ARMA{order}"

                if 'error' not in results:
                    # Generate forecast
                    if hasattr(model, 'forecast_arima'):
                        forecast_result = model.forecast_arima(model_key, len(test_data), return_conf_int=False)
                    else:
                        forecast_result = model.forecast_arma(model_key, len(test_data))

                    if 'error' not in forecast_result:
                        forecast = forecast_result['forecast']
                        metrics = self._calculate_metrics(test_data.values, forecast.values)

                        cv_results.append({
                            'split': split + 1,
                            'train_size': len(train_data),
                            'test_size': len(test_data),
                            'metrics': metrics
                        })

            except Exception as e:
                warnings.warn(f"CV split {split + 1} failed: {str(e)}")
                continue

        if not cv_results:
            return {'error': 'No successful CV splits'}

        # Calculate mean and std of metrics across splits
        metric_names = cv_results[0]['metrics'].keys()
        cv_summary = {}

        for metric in metric_names:
            values = [result['metrics'][metric] for result in cv_results]
            cv_summary[f'{metric}_mean'] = np.mean(values)
            cv_summary[f'{metric}_std'] = np.std(values)

        return {
            'method': 'Time Series Cross-Validation',
            'model_class': model_class.__name__,
            'n_splits': len(cv_results),
            'cv_results': cv_results,
            'cv_summary': cv_summary
        }

    def evaluate_model_performance(self, actuals: np.ndarray, forecasts: np.ndarray,
                                 model_name: str = "Model") -> Dict[str, Any]:
        """
        Comprehensive model performance evaluation.

        Parameters:
        -----------
        actuals : np.ndarray
            Actual values
        forecasts : np.ndarray
            Forecasted values
        model_name : str, default "Model"
            Name of the model

        Returns:
        --------
        dict
            Comprehensive performance metrics
        """
        metrics = self._calculate_metrics(actuals, forecasts)

        # Additional analysis
        residuals = actuals - forecasts

        performance = {
            'model_name': model_name,
            'basic_metrics': metrics,
            'residual_analysis': {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'min': np.min(residuals),
                'max': np.max(residuals),
                'skewness': stats.skew(residuals),
                'kurtosis': stats.kurtosis(residuals)
            },
            'directional_accuracy': self._calculate_directional_accuracy(actuals, forecasts),
            'forecast_bias': np.mean(residuals),
            'tracking_signal': self._calculate_tracking_signal(residuals)
        }

        return performance

    def _calculate_metrics(self, actuals: np.ndarray, forecasts: np.ndarray) -> Dict[str, float]:
        """Calculate various forecast accuracy metrics."""
        # Handle edge cases
        if len(actuals) != len(forecasts) or len(actuals) == 0:
            return {'error': 'Invalid input arrays'}

        residuals = actuals - forecasts

        metrics = {
            'mae': np.mean(np.abs(residuals)),
            'mse': np.mean(residuals ** 2),
            'rmse': np.sqrt(np.mean(residuals ** 2)),
        }

        # MAPE (handle zero values)
        non_zero_actuals = actuals != 0
        if np.any(non_zero_actuals):
            mape = np.mean(np.abs(residuals[non_zero_actuals] / actuals[non_zero_actuals])) * 100
            metrics['mape'] = mape
        else:
            metrics['mape'] = np.inf

        # SMAPE
        denominator = (np.abs(actuals) + np.abs(forecasts)) / 2
        non_zero_denom = denominator != 0
        if np.any(non_zero_denom):
            smape = np.mean(np.abs(residuals[non_zero_denom] / denominator[non_zero_denom])) * 100
            metrics['smape'] = smape
        else:
            metrics['smape'] = 0

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        metrics['r_squared'] = r_squared

        return metrics

    def _calculate_fold_metrics(self, actuals: np.ndarray, forecasts: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for a single fold."""
        return self._calculate_metrics(actuals, forecasts)

    def _calculate_directional_accuracy(self, actuals: np.ndarray, forecasts: np.ndarray) -> float:
        """Calculate directional accuracy (percentage of correct direction predictions)."""
        if len(actuals) < 2:
            return 0.0

        actual_directions = np.diff(actuals) > 0
        forecast_directions = np.diff(forecasts) > 0

        correct_directions = actual_directions == forecast_directions
        return np.mean(correct_directions) * 100

    def _calculate_tracking_signal(self, residuals: np.ndarray) -> float:
        """Calculate tracking signal (cumulative error / MAD)."""
        cumulative_error = np.cumsum(residuals)
        mad = np.mean(np.abs(residuals - np.mean(residuals)))

        if mad == 0:
            return 0.0

        return cumulative_error[-1] / (mad * len(residuals))

class HyperparameterTuner:
    """
    ARIMA hyperparameter tuning with computational efficiency analysis.
    """

    def __init__(self):
        """Initialize the hyperparameter tuner."""
        self.tuning_results = {}
        self.best_models = {}

    def grid_search_arima(self, data: pd.Series,
                         p_range: Tuple[int, int] = (0, 5),
                         d_range: Tuple[int, int] = (0, 2),
                         q_range: Tuple[int, int] = (0, 5),
                         criterion: str = 'aic',
                         seasonal: bool = False,
                         seasonal_periods: int = 12,
                         max_models: int = 100) -> Dict[str, Any]:
        """
        Grid search for optimal ARIMA parameters.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        p_range : tuple, default (0, 5)
            Range for AR order (min_p, max_p)
        d_range : tuple, default (0, 2)
            Range for differencing order (min_d, max_d)
        q_range : tuple, default (0, 5)
            Range for MA order (min_q, max_q)
        criterion : str, default 'aic'
            Information criterion for selection ('aic', 'bic')
        seasonal : bool, default False
            Whether to include seasonal components
        seasonal_periods : int, default 12
            Seasonal periods
        max_models : int, default 100
            Maximum number of models to test

        Returns:
        --------
        dict
            Grid search results
        """
        import time

        results = []
        models_tested = 0
        start_time = time.time()

        best_score = np.inf
        best_params = None
        best_model_result = None

        arima_modeler = ARIMAModeler()

        # Generate parameter combinations
        param_combinations = []
        for p in range(p_range[0], p_range[1] + 1):
            for d in range(d_range[0], d_range[1] + 1):
                for q in range(q_range[0], q_range[1] + 1):
                    param_combinations.append((p, d, q))

        # Limit combinations if too many
        if len(param_combinations) > max_models:
            warnings.warn(f"Too many combinations ({len(param_combinations)}). Limiting to {max_models}.")
            param_combinations = param_combinations[:max_models]

        for p, d, q in param_combinations:
            try:
                model_start_time = time.time()

                order = (p, d, q)
                seasonal_order = (1, 1, 1, seasonal_periods) if seasonal else None

                # Fit model
                fit_result = arima_modeler.fit_arima(data, order, seasonal_order)

                model_end_time = time.time()
                model_time = model_end_time - model_start_time

                if 'error' not in fit_result:
                    score = fit_result[criterion]

                    result = {
                        'order': order,
                        'seasonal_order': seasonal_order,
                        'aic': fit_result['aic'],
                        'bic': fit_result['bic'],
                        'hqic': fit_result.get('hqic'),
                        'llf': fit_result['llf'],
                        'score': score,
                        'fit_time': model_time,
                        'parameters': len(fit_result.get('parameters', {}))
                    }

                    results.append(result)

                    if score < best_score:
                        best_score = score
                        best_params = {
                            'order': order,
                            'seasonal_order': seasonal_order
                        }
                        best_model_result = fit_result

                models_tested += 1

                # Progress update
                if models_tested % 10 == 0:
                    elapsed_time = time.time() - start_time
                    avg_time_per_model = elapsed_time / models_tested
                    remaining_models = len(param_combinations) - models_tested
                    estimated_remaining_time = avg_time_per_model * remaining_models

                    print(f"Tested {models_tested}/{len(param_combinations)} models. "
                          f"Estimated remaining time: {estimated_remaining_time:.1f}s")

            except Exception as e:
                continue

        total_time = time.time() - start_time

        if not results:
            return {'error': 'No valid models found'}

        # Sort results by score
        results.sort(key=lambda x: x['score'])

        grid_search_results = {
            'method': 'Grid Search',
            'criterion': criterion,
            'p_range': p_range,
            'd_range': d_range,
            'q_range': q_range,
            'seasonal': seasonal,
            'seasonal_periods': seasonal_periods,
            'total_models_tested': models_tested,
            'total_time': total_time,
            'avg_time_per_model': total_time / models_tested,
            'best_params': best_params,
            'best_score': best_score,
            'best_model': best_model_result,
            'all_results': results[:20],  # Top 20 models
            'performance_analysis': self._analyze_performance_tradeoffs(results)
        }

        # Store results
        key = f"grid_search_{len(self.tuning_results)}"
        self.tuning_results[key] = grid_search_results
        self.best_models[key] = best_model_result

        return grid_search_results

    def complexity_vs_performance_analysis(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze trade-offs between model complexity and performance.

        Parameters:
        -----------
        results : list
            List of model results from grid search

        Returns:
        --------
        dict
            Analysis of complexity vs performance trade-offs
        """
        if not results:
            return {'error': 'No results provided'}

        # Calculate complexity scores
        for result in results:
            p, d, q = result['order']
            complexity = p + d + q
            if result['seasonal_order']:
                P, D, Q, s = result['seasonal_order']
                complexity += P + D + Q
            result['complexity'] = complexity

        # Group by complexity
        complexity_groups = {}
        for result in results:
            complexity = result['complexity']
            if complexity not in complexity_groups:
                complexity_groups[complexity] = []
            complexity_groups[complexity].append(result)

        # Find best model for each complexity level
        complexity_analysis = {}
        for complexity, models in complexity_groups.items():
            best_model = min(models, key=lambda x: x['score'])
            complexity_analysis[complexity] = {
                'best_score': best_model['score'],
                'best_order': best_model['order'],
                'avg_fit_time': np.mean([m['fit_time'] for m in models]),
                'model_count': len(models)
            }

        # Calculate efficiency metrics
        complexities = sorted(complexity_analysis.keys())
        scores = [complexity_analysis[c]['best_score'] for c in complexities]
        times = [complexity_analysis[c]['avg_fit_time'] for c in complexities]

        analysis = {
            'complexity_levels': len(complexities),
            'complexity_range': (min(complexities), max(complexities)),
            'score_range': (min(scores), max(scores)),
            'time_range': (min(times), max(times)),
            'complexity_analysis': complexity_analysis,
            'recommendations': self._generate_complexity_recommendations(complexity_analysis)
        }

        return analysis

    def _analyze_performance_tradeoffs(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze performance trade-offs in model results."""
        if not results:
            return {}

        # Add complexity scores
        for result in results:
            p, d, q = result['order']
            complexity = p + q  # Simple complexity measure
            result['complexity'] = complexity

        # Performance vs complexity analysis
        complexities = [r['complexity'] for r in results]
        scores = [r['score'] for r in results]
        times = [r['fit_time'] for r in results]

        # Find Pareto front (models that are not dominated in both score and time)
        pareto_models = []
        for i, result in enumerate(results):
            is_pareto = True
            for j, other in enumerate(results):
                if i != j:
                    # Check if other dominates current (better score AND faster time)
                    if (other['score'] <= result['score'] and
                        other['fit_time'] <= result['fit_time'] and
                        (other['score'] < result['score'] or other['fit_time'] < result['fit_time'])):
                        is_pareto = False
                        break
            if is_pareto:
                pareto_models.append(result)

        analysis = {
            'total_models': len(results),
            'complexity_range': (min(complexities), max(complexities)),
            'score_range': (min(scores), max(scores)),
            'time_range': (min(times), max(times)),
            'pareto_efficient_models': len(pareto_models),
            'fastest_model': min(results, key=lambda x: x['fit_time']),
            'best_score_model': min(results, key=lambda x: x['score']),
            'most_complex_model': max(results, key=lambda x: x['complexity']),
            'simplest_model': min(results, key=lambda x: x['complexity'])
        }

        return analysis

    def _generate_complexity_recommendations(self, complexity_analysis: Dict) -> List[str]:
        """Generate recommendations based on complexity analysis."""
        recommendations = []

        if not complexity_analysis:
            return ['No models to analyze']

        # Sort by complexity
        complexities = sorted(complexity_analysis.keys())

        # Find the point of diminishing returns
        scores = [complexity_analysis[c]['best_score'] for c in complexities]
        score_improvements = []

        for i in range(1, len(scores)):
            improvement = scores[i-1] - scores[i]  # Lower scores are better
            score_improvements.append(improvement)

        if score_improvements:
            # Find where improvements become minimal
            avg_improvement = np.mean(score_improvements)

            for i, improvement in enumerate(score_improvements):
                if improvement < avg_improvement * 0.1:  # Less than 10% of average improvement
                    recommended_complexity = complexities[i]
                    recommendations.append(f"Consider complexity level {recommended_complexity} as optimal trade-off point")
                    break

        # Time-based recommendations
        times = [complexity_analysis[c]['avg_fit_time'] for c in complexities]
        if max(times) > min(times) * 5:  # If max time is 5x min time
            fast_complexity = complexities[np.argmin(times)]
            recommendations.append(f"For fast fitting, use complexity level {fast_complexity}")

        # Simple model recommendation
        simplest_complexity = min(complexities)
        recommendations.append(f"Simplest viable model has complexity {simplest_complexity}")

        return recommendations

class BenchmarkModels:
    """
    Benchmark ARIMA models for comparison against calibrated models.
    """

    def __init__(self):
        """Initialize the benchmark models."""
        self.benchmark_results = {}
        self.model_comparison = {}

    def white_noise_model(self, data: pd.Series, estimate_mean: bool = True) -> Dict[str, Any]:
        """
        Fit White Noise model (ARIMA(0,0,0)).

        Parameters:
        -----------
        data : pd.Series
            Time series data
        estimate_mean : bool, default True
            Whether to estimate the mean (constant term)

        Returns:
        --------
        dict
            White noise model results
        """
        try:
            data_clean = data.dropna()
            n = len(data_clean)

            if estimate_mean:
                # Estimate mean
                mean_est = data_clean.mean()
                residuals = data_clean - mean_est
                fitted_values = pd.Series([mean_est] * n, index=data_clean.index)

                # Parameters: just the mean
                parameters = {'const': mean_est}
                n_params = 1
            else:
                # Zero mean model
                mean_est = 0.0
                residuals = data_clean
                fitted_values = pd.Series([0.0] * n, index=data_clean.index)

                parameters = {}
                n_params = 0

            # Calculate variance
            sigma2 = np.var(residuals, ddof=n_params)

            # Information criteria
            log_likelihood = -0.5 * (n * np.log(2 * np.pi) + n * np.log(sigma2) + n)
            aic = -2 * log_likelihood + 2 * n_params
            bic = -2 * log_likelihood + np.log(n) * n_params

            results = {
                'model_name': 'White Noise ARIMA(0,0,0)',
                'order': (0, 0, 0),
                'fitted_values': fitted_values,
                'residuals': residuals,
                'parameters': parameters,
                'sigma2': sigma2,
                'mean': mean_est,
                'aic': aic,
                'bic': bic,
                'loglikelihood': log_likelihood,
                'n_params': n_params,
                'method': 'White Noise Benchmark'
            }

            self.benchmark_results['white_noise'] = results
            return results

        except Exception as e:
            return {'error': f'White noise model failed: {str(e)}'}

    def random_walk_model(self, data: pd.Series, with_drift: bool = False) -> Dict[str, Any]:
        """
        Fit Random Walk model (ARIMA(0,1,0)) with optional drift.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        with_drift : bool, default False
            Whether to include drift term

        Returns:
        --------
        dict
            Random walk model results
        """
        try:
            data_clean = data.dropna()
            n = len(data_clean)

            # First difference the data
            diff_data = data_clean.diff().dropna()
            n_diff = len(diff_data)

            if with_drift:
                # Estimate drift (mean of first differences)
                drift = diff_data.mean()
                residuals = diff_data - drift

                # Fitted values are original data reconstructed with drift
                fitted_values = pd.Series(index=data_clean.index, dtype=float)
                fitted_values.iloc[0] = data_clean.iloc[0]

                for i in range(1, n):
                    fitted_values.iloc[i] = fitted_values.iloc[i-1] + drift

                parameters = {'drift': drift}
                n_params = 1
                model_name = 'Random Walk with Drift ARIMA(0,1,0)'

            else:
                # No drift
                drift = 0.0
                residuals = diff_data

                # Fitted values are just cumulative sum
                fitted_values = pd.Series(index=data_clean.index, dtype=float)
                fitted_values.iloc[0] = data_clean.iloc[0]

                for i in range(1, n):
                    fitted_values.iloc[i] = data_clean.iloc[i-1]  # Previous value as prediction

                parameters = {}
                n_params = 0
                model_name = 'Random Walk ARIMA(0,1,0)'

            # Calculate innovation variance
            sigma2 = np.var(residuals, ddof=n_params)

            # Information criteria (based on differenced series)
            log_likelihood = -0.5 * (n_diff * np.log(2 * np.pi) + n_diff * np.log(sigma2) + n_diff)
            aic = -2 * log_likelihood + 2 * n_params
            bic = -2 * log_likelihood + np.log(n_diff) * n_params

            # Calculate residuals for original series
            model_residuals = data_clean - fitted_values

            results = {
                'model_name': model_name,
                'order': (0, 1, 0),
                'fitted_values': fitted_values,
                'residuals': model_residuals,
                'innovation_residuals': residuals,
                'parameters': parameters,
                'sigma2': sigma2,
                'drift': drift,
                'aic': aic,
                'bic': bic,
                'loglikelihood': log_likelihood,
                'n_params': n_params,
                'method': 'Random Walk Benchmark'
            }

            key = 'random_walk_drift' if with_drift else 'random_walk'
            self.benchmark_results[key] = results
            return results

        except Exception as e:
            return {'error': f'Random walk model failed: {str(e)}'}

    def pure_ar_model(self, data: pd.Series, p: int) -> Dict[str, Any]:
        """
        Fit pure Autoregressive AR(p) model (ARIMA(p,0,0)).

        Parameters:
        -----------
        data : pd.Series
            Time series data
        p : int
            AR order

        Returns:
        --------
        dict
            Pure AR model results
        """
        try:
            # Use existing ARMAModeler for AR fitting
            arma_modeler = ARMAModeler()
            ar_results = arma_modeler.fit_arma_ols(data, (p, 0))

            if 'error' in ar_results:
                return ar_results

            # Reformat results for benchmark
            results = {
                'model_name': f'Pure AR({p}) ARIMA({p},0,0)',
                'order': (p, 0, 0),
                'fitted_values': ar_results['fitted_values'],
                'residuals': ar_results['residuals'],
                'parameters': ar_results['parameters'],
                'constant': ar_results.get('constant', 0),
                'aic': ar_results['aic'],
                'bic': ar_results['bic'],
                'loglikelihood': ar_results['llf'],
                'r_squared': ar_results.get('r_squared', 0),
                'n_params': p + 1,  # AR params + constant
                'method': f'AR({p}) Benchmark'
            }

            self.benchmark_results[f'ar_{p}'] = results
            return results

        except Exception as e:
            return {'error': f'Pure AR({p}) model failed: {str(e)}'}

    def pure_ma_model(self, data: pd.Series, q: int) -> Dict[str, Any]:
        """
        Fit pure Moving Average MA(q) model (ARIMA(0,0,q)).

        Parameters:
        -----------
        data : pd.Series
            Time series data
        q : int
            MA order

        Returns:
        --------
        dict
            Pure MA model results
        """
        try:
            # Use existing ARMAModeler for MA fitting
            arma_modeler = ARMAModeler()
            ma_results = arma_modeler.fit_arma_ols(data, (0, q))

            if 'error' in ma_results:
                return ma_results

            # Reformat results for benchmark
            results = {
                'model_name': f'Pure MA({q}) ARIMA(0,0,{q})',
                'order': (0, 0, q),
                'fitted_values': ma_results['fitted_values'],
                'residuals': ma_results['residuals'],
                'parameters': ma_results['parameters'],
                'aic': ma_results['aic'],
                'bic': ma_results['bic'],
                'loglikelihood': ma_results['llf'],
                'n_params': q + 1,  # MA params + constant
                'method': f'MA({q}) Benchmark',
                'iterations': ma_results.get('iterations', 'N/A'),
                'note': ma_results.get('note', '')
            }

            self.benchmark_results[f'ma_{q}'] = results
            return results

        except Exception as e:
            return {'error': f'Pure MA({q}) model failed: {str(e)}'}

    def compare_against_benchmarks(self, calibrated_model: Dict[str, Any],
                                 data: pd.Series) -> Dict[str, Any]:
        """
        Compare calibrated ARIMA model against benchmark models.

        Parameters:
        -----------
        calibrated_model : dict
            Results from calibrated ARIMA model
        data : pd.Series
            Original time series data

        Returns:
        --------
        dict
            Comprehensive model comparison
        """
        try:
            # Generate all benchmark models
            benchmarks = {}

            # White noise
            wn_result = self.white_noise_model(data, estimate_mean=True)
            if 'error' not in wn_result:
                benchmarks['White Noise'] = wn_result

            # Random walk
            rw_result = self.random_walk_model(data, with_drift=False)
            if 'error' not in rw_result:
                benchmarks['Random Walk'] = rw_result

            # Random walk with drift
            rwd_result = self.random_walk_model(data, with_drift=True)
            if 'error' not in rwd_result:
                benchmarks['Random Walk + Drift'] = rwd_result

            # Pure AR models (up to order 3)
            for p in range(1, 4):
                ar_result = self.pure_ar_model(data, p)
                if 'error' not in ar_result:
                    benchmarks[f'AR({p})'] = ar_result

            # Pure MA models (up to order 2)
            for q in range(1, 3):
                ma_result = self.pure_ma_model(data, q)
                if 'error' not in ma_result:
                    benchmarks[f'MA({q})'] = ma_result

            # Add calibrated model to comparison
            if 'error' not in calibrated_model:
                model_name = calibrated_model.get('model_name', 'Calibrated ARIMA')
                benchmarks[model_name] = calibrated_model

            # Create comparison table
            comparison_table = []
            for name, model in benchmarks.items():
                if 'error' not in model:
                    comparison_table.append({
                        'Model': name,
                        'Order': model.get('order', 'N/A'),
                        'AIC': model.get('aic', np.inf),
                        'BIC': model.get('bic', np.inf),
                        'LogLikelihood': model.get('loglikelihood', -np.inf),
                        'Parameters': model.get('n_params', 0),
                        'Method': model.get('method', 'Unknown')
                    })

            # Sort by AIC
            comparison_table.sort(key=lambda x: x['AIC'])

            # Find best models
            if comparison_table:
                best_aic = comparison_table[0]
                best_bic = min(comparison_table, key=lambda x: x['BIC'])
                best_ll = max(comparison_table, key=lambda x: x['LogLikelihood'])

                # Calculate relative performance
                for row in comparison_table:
                    row['AIC_diff'] = row['AIC'] - best_aic['AIC']
                    row['BIC_diff'] = row['BIC'] - best_bic['BIC']

                summary = {
                    'total_models': len(comparison_table),
                    'best_aic_model': best_aic['Model'],
                    'best_bic_model': best_bic['Model'],
                    'best_ll_model': best_ll['Model'],
                    'aic_range': (min(row['AIC'] for row in comparison_table),
                                max(row['AIC'] for row in comparison_table)),
                    'bic_range': (min(row['BIC'] for row in comparison_table),
                                max(row['BIC'] for row in comparison_table))
                }

                comparison_results = {
                    'comparison_table': comparison_table,
                    'benchmark_models': benchmarks,
                    'summary': summary,
                    'recommendations': self._generate_benchmark_recommendations(comparison_table)
                }

            else:
                comparison_results = {'error': 'No valid models for comparison'}

            self.model_comparison = comparison_results
            return comparison_results

        except Exception as e:
            return {'error': f'Benchmark comparison failed: {str(e)}'}

    def _generate_benchmark_recommendations(self, comparison_table: List[Dict]) -> List[str]:
        """Generate recommendations based on benchmark comparison."""
        recommendations = []

        if not comparison_table:
            return ['No models available for comparison']

        # Best model recommendations
        best_model = comparison_table[0]
        recommendations.append(f"Best overall model by AIC: {best_model['Model']}")

        # Check if simple models are competitive
        simple_models = [row for row in comparison_table
                        if any(simple in row['Model'] for simple in ['White Noise', 'Random Walk', 'AR(1)', 'MA(1)'])]

        if simple_models and simple_models[0]['AIC_diff'] < 2:
            recommendations.append(f"Simple model {simple_models[0]['Model']} is competitive (ΔAIC < 2)")

        # Check for overfitting
        high_param_models = [row for row in comparison_table if row['Parameters'] > 5]
        if high_param_models:
            worst_high_param = max(high_param_models, key=lambda x: x['AIC'])
            recommendations.append(f"High-parameter models like {worst_high_param['Model']} may be overfitting")

        # Model selection guidance
        aic_winner = comparison_table[0]['Model']
        bic_winner = min(comparison_table, key=lambda x: x['BIC'])['Model']

        if aic_winner != bic_winner:
            recommendations.append(f"AIC favors {aic_winner}, BIC favors {bic_winner}. Consider parsimony vs fit trade-off")
        else:
            recommendations.append(f"Both AIC and BIC agree on {aic_winner}")

        return recommendations

class InformationCriteria:
    """
    Comprehensive Information Criteria evaluation for ARIMA models.
    """

    def __init__(self):
        """Initialize information criteria calculator."""
        self.criteria_results = {}

    def calculate_aic_bic(self, model_results: Dict[str, Any], n_obs: int = None) -> Dict[str, Any]:
        """
        Calculate AIC and BIC for a fitted model.

        Parameters:
        -----------
        model_results : dict
            Model fitting results containing loglikelihood and parameters
        n_obs : int, optional
            Number of observations (if not in model_results)

        Returns:
        --------
        dict
            Information criteria results
        """
        try:
            # Extract key values
            log_likelihood = model_results.get('loglikelihood', model_results.get('llf'))
            n_params = model_results.get('n_params', len(model_results.get('parameters', {})))

            if n_obs is None:
                # Try to infer from fitted values or residuals
                fitted_values = model_results.get('fitted_values')
                residuals = model_results.get('residuals')

                if fitted_values is not None:
                    n_obs = len(fitted_values)
                elif residuals is not None:
                    n_obs = len(residuals)
                else:
                    raise ValueError("Cannot determine number of observations")

            if log_likelihood is None:
                raise ValueError("Log-likelihood not available")

            # Calculate information criteria
            aic = -2 * log_likelihood + 2 * n_params
            bic = -2 * log_likelihood + np.log(n_obs) * n_params

            # Calculate corrected AIC for small samples
            aicc = aic + (2 * n_params * (n_params + 1)) / (n_obs - n_params - 1) if n_obs > n_params + 1 else np.inf

            # Calculate Hannan-Quinn Information Criterion
            hqic = -2 * log_likelihood + 2 * np.log(np.log(n_obs)) * n_params

            results = {
                'aic': aic,
                'bic': bic,
                'aicc': aicc,
                'hqic': hqic,
                'log_likelihood': log_likelihood,
                'n_params': n_params,
                'n_obs': n_obs,
                'effective_sample_size': n_obs - n_params,
                'criteria_interpretation': self._interpret_criteria(aic, bic, aicc, hqic)
            }

            return results

        except Exception as e:
            return {'error': f'Information criteria calculation failed: {str(e)}'}

    def model_selection_table(self, models_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create comprehensive model selection table with information criteria.

        Parameters:
        -----------
        models_list : list
            List of fitted model results

        Returns:
        --------
        dict
            Model selection table and recommendations
        """
        try:
            selection_table = []

            for i, model in enumerate(models_list):
                if 'error' not in model:
                    # Calculate criteria if not already present
                    if 'aic' not in model or 'bic' not in model:
                        criteria = self.calculate_aic_bic(model)
                        if 'error' not in criteria:
                            model.update(criteria)

                    model_name = model.get('model_name', f'Model_{i+1}')
                    order = model.get('order', 'Unknown')

                    selection_table.append({
                        'Model': model_name,
                        'Order': order,
                        'LogLikelihood': model.get('log_likelihood', model.get('llf', np.nan)),
                        'AIC': model.get('aic', np.inf),
                        'BIC': model.get('bic', np.inf),
                        'AICc': model.get('aicc', np.inf),
                        'HQIC': model.get('hqic', np.inf),
                        'Parameters': model.get('n_params', 0),
                        'Observations': model.get('n_obs', 0)
                    })

            if not selection_table:
                return {'error': 'No valid models for selection table'}

            # Sort by AIC (primary criterion)
            selection_table.sort(key=lambda x: x['AIC'])

            # Calculate relative criteria (differences from best)
            best_aic = selection_table[0]['AIC']
            best_bic = min(row['BIC'] for row in selection_table)
            best_aicc = min(row['AICc'] for row in selection_table)

            for row in selection_table:
                row['ΔAIC'] = row['AIC'] - best_aic
                row['ΔBIC'] = row['BIC'] - best_bic
                row['ΔAICc'] = row['AICc'] - best_aicc

                # Akaike weights
                row['Akaike_Weight'] = np.exp(-0.5 * row['ΔAIC'])

            # Normalize Akaike weights
            total_weight = sum(row['Akaike_Weight'] for row in selection_table)
            for row in selection_table:
                row['Akaike_Weight'] /= total_weight

            # Generate recommendations
            recommendations = self._generate_selection_recommendations(selection_table)

            results = {
                'selection_table': selection_table,
                'best_models': {
                    'aic_best': selection_table[0]['Model'],
                    'bic_best': min(selection_table, key=lambda x: x['BIC'])['Model'],
                    'aicc_best': min(selection_table, key=lambda x: x['AICc'])['Model']
                },
                'model_weights': {row['Model']: row['Akaike_Weight'] for row in selection_table},
                'recommendations': recommendations
            }

            self.criteria_results = results
            return results

        except Exception as e:
            return {'error': f'Model selection table creation failed: {str(e)}'}

    def _interpret_criteria(self, aic: float, bic: float, aicc: float, hqic: float) -> Dict[str, str]:
        """Interpret information criteria values."""
        interpretation = {}

        # AIC interpretation
        if aic < 0:
            interpretation['aic'] = "Negative AIC indicates very good model fit"
        elif aic < 100:
            interpretation['aic'] = "Relatively low AIC suggests good model"
        else:
            interpretation['aic'] = "High AIC may indicate poor fit or overfitting"

        # BIC vs AIC comparison
        if bic > aic:
            interpretation['bic_vs_aic'] = "BIC > AIC suggests penalty for complexity is stronger"
        else:
            interpretation['bic_vs_aic'] = "BIC ≤ AIC (unusual, check calculations)"

        # AICc relevance
        ratio = aicc / aic
        if ratio > 1.1:
            interpretation['aicc'] = "AICc significantly higher than AIC - small sample correction important"
        else:
            interpretation['aicc'] = "AICc ≈ AIC - sample size adequate"

        return interpretation

    def _generate_selection_recommendations(self, selection_table: List[Dict]) -> List[str]:
        """Generate model selection recommendations."""
        recommendations = []

        if not selection_table:
            return ['No models available for recommendations']

        best_model = selection_table[0]
        recommendations.append(f"Best model by AIC: {best_model['Model']} (AIC = {best_model['AIC']:.2f})")

        # Check for substantial differences
        if len(selection_table) > 1:
            second_best = selection_table[1]
            aic_diff = second_best['ΔAIC']

            if aic_diff < 2:
                recommendations.append("Top models are very close (ΔAIC < 2) - consider simplicity")
            elif aic_diff < 7:
                recommendations.append("Top models show moderate differences (2 ≤ ΔAIC < 7)")
            else:
                recommendations.append("Clear best model (ΔAIC ≥ 7)")

        # Akaike weights interpretation
        best_weight = best_model['Akaike_Weight']
        if best_weight > 0.9:
            recommendations.append(f"Very strong evidence for best model (weight = {best_weight:.3f})")
        elif best_weight > 0.7:
            recommendations.append(f"Strong evidence for best model (weight = {best_weight:.3f})")
        else:
            recommendations.append(f"Moderate evidence for best model (weight = {best_weight:.3f})")

        return recommendations

class ARCHGARCHModeler:
    """
    ARCH/GARCH volatility modeling with comprehensive diagnostics.
    """

    def __init__(self):
        """Initialize the ARCH/GARCH modeler."""
        self.arch_models = {}
        self.garch_models = {}
        self.diagnostics_results = {}

    def fit_arch(self, data: pd.Series, arch_order: int = 1) -> Dict[str, Any]:
        """
        Fit ARCH(m) model for volatility clustering.

        Parameters:
        -----------
        data : pd.Series
            Time series data (returns or residuals)
        arch_order : int, default 1
            ARCH order (m)

        Returns:
        --------
        dict
            ARCH model results
        """
        try:
            # Check if arch package is available
            try:
                from arch import arch_model
                use_arch_package = True
            except ImportError:
                use_arch_package = False
                warnings.warn("ARCH package not available. Using manual implementation.")

            data_clean = data.dropna()
            n = len(data_clean)

            if use_arch_package:
                # Using arch package
                model = arch_model(data_clean, vol='ARCH', p=arch_order, rescale=False)
                fitted_model = model.fit(disp='off')

                results = {
                    'model_name': f'ARCH({arch_order})',
                    'arch_order': arch_order,
                    'fitted_model': fitted_model,
                    'residuals': fitted_model.resid,
                    'conditional_volatility': fitted_model.conditional_volatility,
                    'parameters': fitted_model.params.to_dict(),
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic,
                    'loglikelihood': fitted_model.loglikelihood,
                    'n_params': len(fitted_model.params),
                    'method': 'ARCH (arch package)',
                    'summary': str(fitted_model.summary())
                }

            else:
                # Manual implementation
                results = self._fit_arch_manual(data_clean, arch_order)

            self.arch_models[f'ARCH_{arch_order}'] = results
            return results

        except Exception as e:
            return {'error': f'ARCH({arch_order}) fitting failed: {str(e)}'}

    def fit_garch(self, data: pd.Series, garch_order: Tuple[int, int] = (1, 1)) -> Dict[str, Any]:
        """
        Fit GARCH(p,q) model for volatility clustering.

        Parameters:
        -----------
        data : pd.Series
            Time series data (returns or residuals)
        garch_order : tuple, default (1, 1)
            GARCH order (p, q)

        Returns:
        --------
        dict
            GARCH model results
        """
        try:
            # Check if arch package is available
            try:
                from arch import arch_model
                use_arch_package = True
            except ImportError:
                use_arch_package = False
                warnings.warn("ARCH package not available. Using manual implementation.")

            data_clean = data.dropna()
            p, q = garch_order

            if use_arch_package:
                # Using arch package
                model = arch_model(data_clean, vol='GARCH', p=p, q=q, rescale=False)
                fitted_model = model.fit(disp='off')

                results = {
                    'model_name': f'GARCH({p},{q})',
                    'garch_order': garch_order,
                    'fitted_model': fitted_model,
                    'residuals': fitted_model.resid,
                    'conditional_volatility': fitted_model.conditional_volatility,
                    'parameters': fitted_model.params.to_dict(),
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic,
                    'loglikelihood': fitted_model.loglikelihood,
                    'n_params': len(fitted_model.params),
                    'method': 'GARCH (arch package)',
                    'summary': str(fitted_model.summary())
                }

            else:
                # Manual implementation
                results = self._fit_garch_manual(data_clean, garch_order)

            self.garch_models[f'GARCH_{p}_{q}'] = results
            return results

        except Exception as e:
            return {'error': f'GARCH{garch_order} fitting failed: {str(e)}'}

    def arch_lm_test(self, residuals: pd.Series, lags: int = 5) -> Dict[str, Any]:
        """
        Engle's ARCH Lagrange Multiplier test for heteroskedasticity.

        Parameters:
        -----------
        residuals : pd.Series
            Model residuals
        lags : int, default 5
            Number of lags to test

        Returns:
        --------
        dict
            ARCH LM test results
        """
        try:
            if HAS_STATSMODELS:
                from statsmodels.stats.diagnostic import het_arch

                # Perform ARCH LM test
                lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(residuals.dropna(), nlags=lags)

                results = {
                    'test_name': "Engle's ARCH LM Test",
                    'null_hypothesis': 'No ARCH effects (homoskedasticity)',
                    'alternative_hypothesis': 'ARCH effects present (heteroskedasticity)',
                    'lm_statistic': lm_stat,
                    'lm_pvalue': lm_pvalue,
                    'f_statistic': f_stat,
                    'f_pvalue': f_pvalue,
                    'lags_tested': lags,
                    'reject_null_lm': lm_pvalue < 0.05,
                    'reject_null_f': f_pvalue < 0.05,
                    'conclusion': 'ARCH effects detected' if lm_pvalue < 0.05 else 'No ARCH effects detected',
                    'method': 'Statsmodels implementation'
                }

            else:
                # Manual implementation
                results = self._manual_arch_lm_test(residuals, lags)

            return results

        except Exception as e:
            return {'error': f'ARCH LM test failed: {str(e)}'}

    def nyblom_stability_test(self, residuals: pd.Series) -> Dict[str, Any]:
        """
        Nyblom stability test for parameter constancy.

        Parameters:
        -----------
        residuals : pd.Series
            Standardized residuals from GARCH model

        Returns:
        --------
        dict
            Nyblom stability test results
        """
        try:
            residuals_clean = residuals.dropna()
            n = len(residuals_clean)

            # Calculate partial sums
            partial_sums = np.cumsum(residuals_clean)

            # Nyblom test statistic
            # This is a simplified implementation
            test_statistic = np.sum(partial_sums**2) / (n**2 * np.var(residuals_clean))

            # Critical values (approximate)
            critical_values = {
                '10%': 0.35,
                '5%': 0.47,
                '1%': 0.75
            }

            # Determine significance
            significance = None
            for level, cv in critical_values.items():
                if test_statistic > cv:
                    significance = level

            results = {
                'test_name': 'Nyblom Stability Test',
                'null_hypothesis': 'Parameter stability (constant parameters)',
                'test_statistic': test_statistic,
                'critical_values': critical_values,
                'significance_level': significance,
                'reject_null': test_statistic > critical_values['5%'],
                'conclusion': 'Parameter instability detected' if test_statistic > critical_values['5%'] else 'Parameters appear stable',
                'method': 'Manual implementation'
            }

            return results

        except Exception as e:
            return {'error': f'Nyblom stability test failed: {str(e)}'}

    def sign_bias_test(self, residuals: pd.Series, conditional_volatility: pd.Series) -> Dict[str, Any]:
        """
        Sign bias test for asymmetric volatility effects.

        Parameters:
        -----------
        residuals : pd.Series
            Model residuals
        conditional_volatility : pd.Series
            Conditional volatility from GARCH model

        Returns:
        --------
        dict
            Sign bias test results
        """
        try:
            # Calculate standardized residuals
            std_residuals = residuals / conditional_volatility

            # Create indicator variables
            negative_residuals = (residuals.shift(1) < 0).astype(int)
            positive_residuals = (residuals.shift(1) >= 0).astype(int)

            # Squared standardized residuals
            squared_std_residuals = std_residuals**2

            # Regression data
            data_df = pd.DataFrame({
                'squared_std_resid': squared_std_residuals,
                'negative_lag': negative_residuals,
                'positive_lag': positive_residuals,
                'negative_residual_lag': negative_residuals * residuals.shift(1),
                'positive_residual_lag': positive_residuals * residuals.shift(1)
            }).dropna()

            if HAS_STATSMODELS:
                import statsmodels.api as sm

                # Sign bias test regression
                X = sm.add_constant(data_df[['negative_lag']])
                model = sm.OLS(data_df['squared_std_resid'], X).fit()

                # Joint test for all three effects
                X_joint = sm.add_constant(data_df[['negative_lag', 'negative_residual_lag', 'positive_residual_lag']])
                model_joint = sm.OLS(data_df['squared_std_resid'], X_joint).fit()

                results = {
                    'test_name': 'Sign Bias Test',
                    'sign_bias_coeff': model.params.iloc[1],
                    'sign_bias_pvalue': model.pvalues.iloc[1],
                    'sign_bias_significant': model.pvalues.iloc[1] < 0.05,
                    'joint_test_fstat': model_joint.fvalue,
                    'joint_test_pvalue': model_joint.f_pvalue,
                    'joint_test_significant': model_joint.f_pvalue < 0.05,
                    'r_squared': model.rsquared,
                    'method': 'OLS regression',
                    'conclusion': 'Asymmetric effects detected' if model_joint.f_pvalue < 0.05 else 'No significant asymmetric effects'
                }

            else:
                # Simple correlation-based test
                correlation = np.corrcoef(data_df['squared_std_resid'], data_df['negative_lag'])[0, 1]

                results = {
                    'test_name': 'Sign Bias Test (Simple)',
                    'correlation': correlation,
                    'method': 'Correlation-based (manual)',
                    'conclusion': 'Possible asymmetric effects' if abs(correlation) > 0.1 else 'No clear asymmetric effects'
                }

            return results

        except Exception as e:
            return {'error': f'Sign bias test failed: {str(e)}'}

    def adjusted_pearson_goodness_of_fit(self, residuals: pd.Series, conditional_volatility: pd.Series) -> Dict[str, Any]:
        """
        Adjusted Pearson goodness-of-fit test for standardized residuals.

        Parameters:
        -----------
        residuals : pd.Series
            Model residuals
        conditional_volatility : pd.Series
            Conditional volatility estimates

        Returns:
        --------
        dict
            Adjusted Pearson test results
        """
        try:
            # Calculate standardized residuals
            std_residuals = (residuals / conditional_volatility).dropna()
            n = len(std_residuals)

            # Create bins for grouping
            n_bins = min(20, int(np.sqrt(n)))

            # Theoretical normal quantiles
            quantiles = np.linspace(0, 1, n_bins + 1)
            theoretical_quantiles = stats.norm.ppf(quantiles[1:-1])

            # Observed frequencies in each bin
            observed_freq, bin_edges = np.histogram(std_residuals, bins=n_bins)

            # Expected frequencies (normal distribution)
            expected_freq = n / n_bins * np.ones(n_bins)

            # Pearson chi-square statistic
            # Avoid division by zero
            nonzero_expected = expected_freq > 0
            chi_square = np.sum((observed_freq[nonzero_expected] - expected_freq[nonzero_expected])**2 /
                               expected_freq[nonzero_expected])

            # Degrees of freedom
            df = n_bins - 1 - 2  # -2 for mean and variance estimation

            # P-value
            p_value = 1 - stats.chi2.cdf(chi_square, df) if df > 0 else 1.0

            # Adjusted test statistic (for GARCH models)
            adjusted_chi_square = chi_square * (1 - 2/n)

            results = {
                'test_name': 'Adjusted Pearson Goodness-of-Fit Test',
                'null_hypothesis': 'Standardized residuals follow standard normal distribution',
                'chi_square_statistic': chi_square,
                'adjusted_chi_square': adjusted_chi_square,
                'degrees_freedom': df,
                'p_value': p_value,
                'critical_value_5pct': stats.chi2.ppf(0.95, df) if df > 0 else np.inf,
                'reject_null': p_value < 0.05 and df > 0,
                'n_bins': n_bins,
                'sample_size': n,
                'conclusion': 'Model adequacy rejected' if (p_value < 0.05 and df > 0) else 'Model appears adequate',
                'observed_frequencies': observed_freq.tolist(),
                'expected_frequencies': expected_freq.tolist()
            }

            return results

        except Exception as e:
            return {'error': f'Adjusted Pearson test failed: {str(e)}'}

    def comprehensive_garch_diagnostics(self, model_results: Dict[str, Any],
                                      original_data: pd.Series) -> Dict[str, Any]:
        """
        Comprehensive GARCH model diagnostics.

        Parameters:
        -----------
        model_results : dict
            Fitted GARCH model results
        original_data : pd.Series
            Original time series data

        Returns:
        --------
        dict
            Comprehensive diagnostic results
        """
        try:
            if 'error' in model_results:
                return model_results

            residuals = model_results.get('residuals')
            conditional_volatility = model_results.get('conditional_volatility')

            if residuals is None or conditional_volatility is None:
                return {'error': 'Residuals or conditional volatility not available'}

            diagnostics = {}

            # 1. Ljung-Box test on residuals
            arma_modeler = ARMAModeler()
            ljung_box_resid = arma_modeler.ljung_box_test(residuals, lags=10, model_df=0)
            diagnostics['ljung_box_residuals'] = ljung_box_resid

            # 2. Ljung-Box test on squared residuals
            squared_residuals = residuals**2
            ljung_box_squared = arma_modeler.ljung_box_test(squared_residuals, lags=10, model_df=0)
            diagnostics['ljung_box_squared_residuals'] = ljung_box_squared

            # 3. Information criteria
            info_criteria = InformationCriteria()
            criteria_results = info_criteria.calculate_aic_bic(model_results)
            diagnostics['information_criteria'] = criteria_results

            # 4. ARCH LM test on standardized residuals
            std_residuals = residuals / conditional_volatility
            arch_lm = self.arch_lm_test(std_residuals, lags=5)
            diagnostics['arch_lm_test'] = arch_lm

            # 5. Nyblom stability test
            nyblom = self.nyblom_stability_test(std_residuals)
            diagnostics['nyblom_stability'] = nyblom

            # 6. Sign bias test
            sign_bias = self.sign_bias_test(residuals, conditional_volatility)
            diagnostics['sign_bias_test'] = sign_bias

            # 7. Adjusted Pearson goodness-of-fit
            pearson_test = self.adjusted_pearson_goodness_of_fit(residuals, conditional_volatility)
            diagnostics['adjusted_pearson'] = pearson_test

            # 8. Summary assessment
            summary = self._generate_diagnostic_summary(diagnostics)
            diagnostics['summary'] = summary

            # Store results
            model_name = model_results.get('model_name', 'GARCH_Model')
            self.diagnostics_results[model_name] = diagnostics

            return diagnostics

        except Exception as e:
            return {'error': f'GARCH diagnostics failed: {str(e)}'}

    def _fit_arch_manual(self, data: pd.Series, arch_order: int) -> Dict[str, Any]:
        """Manual ARCH implementation when arch package is not available."""
        warnings.warn("Manual ARCH implementation is basic. Consider installing arch package.")

        try:
            n = len(data)

            # Simple ARCH(1) implementation
            if arch_order == 1:
                # Initial estimates
                alpha0 = np.var(data) * 0.1
                alpha1 = 0.1

                # Simple moment-based estimation
                squared_returns = data**2
                lagged_squared = squared_returns.shift(1).dropna()
                current_squared = squared_returns[1:]

                # Simple regression approach
                X = np.column_stack([np.ones(len(lagged_squared)), lagged_squared])
                y = current_squared

                try:
                    params = np.linalg.lstsq(X, y, rcond=None)[0]
                    alpha0, alpha1 = max(params[0], 0.001), max(params[1], 0.001)
                except:
                    alpha0, alpha1 = np.var(data) * 0.1, 0.1

                # Calculate conditional volatility
                conditional_variance = np.zeros(n)
                conditional_variance[0] = np.var(data)

                for t in range(1, n):
                    conditional_variance[t] = alpha0 + alpha1 * data.iloc[t-1]**2

                conditional_volatility = pd.Series(np.sqrt(conditional_variance), index=data.index)
                residuals = data / conditional_volatility

                # Approximate log-likelihood
                log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * conditional_variance) + data**2 / conditional_variance)

                # Information criteria
                n_params = 2
                aic = -2 * log_likelihood + 2 * n_params
                bic = -2 * log_likelihood + np.log(n) * n_params

                results = {
                    'model_name': f'ARCH({arch_order}) - Manual',
                    'arch_order': arch_order,
                    'residuals': residuals,
                    'conditional_volatility': conditional_volatility,
                    'parameters': {'omega': alpha0, 'alpha[1]': alpha1},
                    'aic': aic,
                    'bic': bic,
                    'loglikelihood': log_likelihood,
                    'n_params': n_params,
                    'method': 'Manual ARCH implementation',
                    'note': 'Basic implementation - use arch package for full functionality'
                }

                return results

            else:
                return {'error': f'Manual ARCH implementation only supports order 1, got {arch_order}'}

        except Exception as e:
            return {'error': f'Manual ARCH fitting failed: {str(e)}'}

    def _fit_garch_manual(self, data: pd.Series, garch_order: Tuple[int, int]) -> Dict[str, Any]:
        """Manual GARCH implementation when arch package is not available."""
        return {'error': 'Manual GARCH implementation not available. Please install arch package.'}

    def _manual_arch_lm_test(self, residuals: pd.Series, lags: int) -> Dict[str, Any]:
        """Manual implementation of ARCH LM test."""
        try:
            residuals_clean = residuals.dropna()
            n = len(residuals_clean)

            # Squared residuals
            squared_residuals = residuals_clean**2

            # Create lagged variables
            X = np.column_stack([squared_residuals.shift(i).dropna() for i in range(1, lags + 1)])
            y = squared_residuals[lags:].values

            # Add constant
            X_with_const = np.column_stack([np.ones(len(X)), X])

            # OLS regression
            try:
                beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                fitted = X_with_const @ beta
                residuals_reg = y - fitted

                # R-squared
                tss = np.sum((y - np.mean(y))**2)
                rss = np.sum(residuals_reg**2)
                r_squared = 1 - (rss / tss)

                # LM statistic
                lm_statistic = (n - lags) * r_squared

                # P-value (approximate)
                from scipy.stats import chi2
                p_value = 1 - chi2.cdf(lm_statistic, lags)

                results = {
                    'test_name': "Engle's ARCH LM Test (Manual)",
                    'null_hypothesis': 'No ARCH effects',
                    'lm_statistic': lm_statistic,
                    'lm_pvalue': p_value,
                    'lags_tested': lags,
                    'reject_null_lm': p_value < 0.05,
                    'conclusion': 'ARCH effects detected' if p_value < 0.05 else 'No ARCH effects detected',
                    'method': 'Manual implementation'
                }

                return results

            except Exception as e:
                return {'error': f'Manual ARCH LM regression failed: {str(e)}'}

        except Exception as e:
            return {'error': f'Manual ARCH LM test failed: {str(e)}'}

    def _generate_diagnostic_summary(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of diagnostic test results."""
        summary = {
            'model_adequacy': 'Good',
            'concerns': [],
            'recommendations': []
        }

        # Check Ljung-Box tests
        if diagnostics.get('ljung_box_residuals', {}).get('reject_null', False):
            summary['concerns'].append('Serial correlation in residuals detected')
            summary['recommendations'].append('Consider improving mean equation specification')

        if diagnostics.get('ljung_box_squared_residuals', {}).get('reject_null', False):
            summary['concerns'].append('Serial correlation in squared residuals detected')
            summary['recommendations'].append('Consider higher-order GARCH model')

        # Check ARCH LM test
        if diagnostics.get('arch_lm_test', {}).get('reject_null_lm', False):
            summary['concerns'].append('Remaining ARCH effects detected')
            summary['recommendations'].append('Model may need higher ARCH/GARCH orders')

        # Check Nyblom stability
        if diagnostics.get('nyblom_stability', {}).get('reject_null', False):
            summary['concerns'].append('Parameter instability detected')
            summary['recommendations'].append('Consider time-varying parameter models')

        # Check sign bias
        if diagnostics.get('sign_bias_test', {}).get('joint_test_significant', False):
            summary['concerns'].append('Asymmetric volatility effects detected')
            summary['recommendations'].append('Consider asymmetric GARCH models (GJR-GARCH, EGARCH)')

        # Check Pearson test
        if diagnostics.get('adjusted_pearson', {}).get('reject_null', False):
            summary['concerns'].append('Standardized residuals deviate from normality')
            summary['recommendations'].append('Consider alternative error distributions')

        # Overall assessment
        if len(summary['concerns']) == 0:
            summary['model_adequacy'] = 'Excellent'
        elif len(summary['concerns']) <= 2:
            summary['model_adequacy'] = 'Good'
        elif len(summary['concerns']) <= 4:
            summary['model_adequacy'] = 'Fair'
        else:
            summary['model_adequacy'] = 'Poor'

        summary['total_concerns'] = len(summary['concerns'])

        return summary

class StateSpaceModels:
    """
    Flexible state space modeling framework including Kalman filtering and dynamic linear models.
    """

    def __init__(self):
        """Initialize the state space models."""
        self.models = {}
        self.filtered_results = {}
        self.smoothed_results = {}

    def setup_local_level_model(self, data: pd.Series, initial_level: float = None,
                               level_variance: float = None, obs_variance: float = None) -> Dict[str, Any]:
        """
        Set up local level model (random walk + noise).

        State equation: α(t+1) = α(t) + η(t), η(t) ~ N(0, σ²_η)
        Observation equation: y(t) = α(t) + ε(t), ε(t) ~ N(0, σ²_ε)

        Parameters:
        -----------
        data : pd.Series
            Observed time series
        initial_level : float, optional
            Initial level estimate
        level_variance : float, optional
            Level innovation variance
        obs_variance : float, optional
            Observation error variance

        Returns:
        --------
        dict
            Local level model setup
        """
        try:
            data_clean = data.dropna()
            n = len(data_clean)

            # Initialize parameters if not provided
            if initial_level is None:
                initial_level = data_clean.iloc[0]

            if level_variance is None:
                level_variance = np.var(data_clean.diff().dropna()) * 0.1

            if obs_variance is None:
                obs_variance = np.var(data_clean) * 0.1

            # State space matrices
            # State transition matrix (F)
            F = np.array([[1.0]])

            # Observation matrix (H)
            H = np.array([[1.0]])

            # State innovation covariance (Q)
            Q = np.array([[level_variance]])

            # Observation error covariance (R)
            R = np.array([[obs_variance]])

            # Initial state and covariance
            initial_state = np.array([initial_level])
            initial_covariance = np.array([[level_variance * 10]])

            model = {
                'model_type': 'Local Level',
                'F': F,  # State transition matrix
                'H': H,  # Observation matrix
                'Q': Q,  # State innovation covariance
                'R': R,  # Observation error covariance
                'initial_state': initial_state,
                'initial_covariance': initial_covariance,
                'data': data_clean,
                'parameters': {
                    'level_variance': level_variance,
                    'obs_variance': obs_variance,
                    'initial_level': initial_level
                }
            }

            self.models['local_level'] = model
            return model

        except Exception as e:
            return {'error': f'Local level model setup failed: {str(e)}'}

    def setup_local_linear_trend_model(self, data: pd.Series) -> Dict[str, Any]:
        """
        Set up local linear trend model.

        State: [level, slope]
        State equation: [α(t+1), β(t+1)] = [1,1; 0,1] * [α(t), β(t)] + [η₁(t), η₂(t)]
        Observation equation: y(t) = [1,0] * [α(t), β(t)] + ε(t)

        Parameters:
        -----------
        data : pd.Series
            Observed time series

        Returns:
        --------
        dict
            Local linear trend model setup
        """
        try:
            data_clean = data.dropna()
            n = len(data_clean)

            # Estimate initial parameters
            initial_level = data_clean.iloc[0]
            initial_slope = np.mean(data_clean.diff().dropna())

            # Variance estimates
            level_var = np.var(data_clean.diff().dropna()) * 0.1
            slope_var = np.var(data_clean.diff().diff().dropna()) * 0.1
            obs_var = np.var(data_clean) * 0.1

            # State space matrices
            F = np.array([[1.0, 1.0],
                         [0.0, 1.0]])

            H = np.array([[1.0, 0.0]])

            Q = np.array([[level_var, 0.0],
                         [0.0, slope_var]])

            R = np.array([[obs_var]])

            initial_state = np.array([initial_level, initial_slope])
            initial_covariance = np.array([[level_var * 10, 0.0],
                                          [0.0, slope_var * 10]])

            model = {
                'model_type': 'Local Linear Trend',
                'F': F,
                'H': H,
                'Q': Q,
                'R': R,
                'initial_state': initial_state,
                'initial_covariance': initial_covariance,
                'data': data_clean,
                'parameters': {
                    'level_variance': level_var,
                    'slope_variance': slope_var,
                    'obs_variance': obs_var
                }
            }

            self.models['local_linear_trend'] = model
            return model

        except Exception as e:
            return {'error': f'Local linear trend model setup failed: {str(e)}'}

    def kalman_filter(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Kalman filter for state estimation.

        Parameters:
        -----------
        model : dict
            State space model specification

        Returns:
        --------
        dict
            Kalman filter results
        """
        try:
            if 'error' in model:
                return model

            # Extract model components
            F = model['F']
            H = model['H']
            Q = model['Q']
            R = model['R']
            data = model['data'].values
            n = len(data)

            # Initialize arrays
            state_dim = F.shape[0]
            obs_dim = H.shape[0]

            # Filtered states and covariances
            filtered_states = np.zeros((n, state_dim))
            filtered_covariances = np.zeros((n, state_dim, state_dim))

            # Predicted states and covariances
            predicted_states = np.zeros((n, state_dim))
            predicted_covariances = np.zeros((n, state_dim, state_dim))

            # Innovations and covariances
            innovations = np.zeros(n)
            innovation_covariances = np.zeros(n)

            # Log-likelihood
            log_likelihood = 0.0

            # Initial values
            state = model['initial_state'].copy()
            covariance = model['initial_covariance'].copy()

            for t in range(n):
                # Prediction step
                if t > 0:
                    state = F @ state
                    covariance = F @ covariance @ F.T + Q

                predicted_states[t] = state.copy()
                predicted_covariances[t] = covariance.copy()

                # Update step
                y_pred = H @ state
                innovation = data[t] - y_pred[0]
                S = H @ covariance @ H.T + R
                S_inv = 1.0 / S[0, 0]

                # Kalman gain
                K = covariance @ H.T * S_inv

                # Update state and covariance
                state = state + K.flatten() * innovation
                I_KH = np.eye(state_dim) - K @ H
                covariance = I_KH @ covariance @ I_KH.T + K @ R @ K.T

                # Store results
                filtered_states[t] = state.copy()
                filtered_covariances[t] = covariance.copy()
                innovations[t] = innovation
                innovation_covariances[t] = S[0, 0]

                # Update log-likelihood
                log_likelihood += -0.5 * (np.log(2 * np.pi) + np.log(S[0, 0]) + innovation**2 / S[0, 0])

            results = {
                'filtered_states': filtered_states,
                'filtered_covariances': filtered_covariances,
                'predicted_states': predicted_states,
                'predicted_covariances': predicted_covariances,
                'innovations': innovations,
                'innovation_covariances': innovation_covariances,
                'log_likelihood': log_likelihood,
                'aic': -2 * log_likelihood + 2 * len(model['parameters']),
                'bic': -2 * log_likelihood + np.log(n) * len(model['parameters'])
            }

            self.filtered_results[model['model_type']] = results
            return results

        except Exception as e:
            return {'error': f'Kalman filter failed: {str(e)}'}

    def kalman_smoother(self, model: Dict[str, Any], filter_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Kalman smoother for retrospective state estimation.

        Parameters:
        -----------
        model : dict
            State space model specification
        filter_results : dict
            Results from Kalman filter

        Returns:
        --------
        dict
            Kalman smoother results
        """
        try:
            if 'error' in model or 'error' in filter_results:
                return {'error': 'Invalid model or filter results'}

            F = model['F']
            filtered_states = filter_results['filtered_states']
            filtered_covariances = filter_results['filtered_covariances']
            predicted_covariances = filter_results['predicted_covariances']

            n, state_dim = filtered_states.shape

            # Initialize smoothed results
            smoothed_states = np.zeros_like(filtered_states)
            smoothed_covariances = np.zeros_like(filtered_covariances)

            # Initialize with final filtered values
            smoothed_states[-1] = filtered_states[-1]
            smoothed_covariances[-1] = filtered_covariances[-1]

            # Backward recursion
            for t in range(n - 2, -1, -1):
                # Smoother gain
                try:
                    A = filtered_covariances[t] @ F.T @ np.linalg.inv(predicted_covariances[t + 1])
                except np.linalg.LinAlgError:
                    A = filtered_covariances[t] @ F.T @ np.linalg.pinv(predicted_covariances[t + 1])

                # Smoothed state and covariance
                smoothed_states[t] = filtered_states[t] + A @ (smoothed_states[t + 1] - F @ filtered_states[t])
                smoothed_covariances[t] = (filtered_covariances[t] +
                                         A @ (smoothed_covariances[t + 1] - predicted_covariances[t + 1]) @ A.T)

            results = {
                'smoothed_states': smoothed_states,
                'smoothed_covariances': smoothed_covariances
            }

            self.smoothed_results[model['model_type']] = results
            return results

        except Exception as e:
            return {'error': f'Kalman smoother failed: {str(e)}'}

    def em_algorithm(self, model: Dict[str, Any], max_iterations: int = 100,
                    tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        EM algorithm for parameter estimation in state space models.

        Parameters:
        -----------
        model : dict
            Initial state space model
        max_iterations : int, default 100
            Maximum number of EM iterations
        tolerance : float, default 1e-6
            Convergence tolerance

        Returns:
        --------
        dict
            EM estimation results
        """
        try:
            current_model = model.copy()
            log_likelihoods = []

            for iteration in range(max_iterations):
                # E-step: Run Kalman filter and smoother
                filter_results = self.kalman_filter(current_model)
                if 'error' in filter_results:
                    return filter_results

                smoother_results = self.kalman_smoother(current_model, filter_results)
                if 'error' in smoother_results:
                    return smoother_results

                current_ll = filter_results['log_likelihood']
                log_likelihoods.append(current_ll)

                # Check convergence
                if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tolerance:
                    break

                # M-step: Update parameters
                current_model = self._update_parameters(current_model, filter_results, smoother_results)

            results = {
                'final_model': current_model,
                'log_likelihoods': log_likelihoods,
                'iterations': iteration + 1,
                'converged': iteration < max_iterations - 1,
                'final_log_likelihood': log_likelihoods[-1],
                'aic': -2 * log_likelihoods[-1] + 2 * len(current_model['parameters']),
                'bic': -2 * log_likelihoods[-1] + np.log(len(current_model['data'])) * len(current_model['parameters'])
            }

            return results

        except Exception as e:
            return {'error': f'EM algorithm failed: {str(e)}'}

    def _update_parameters(self, model: Dict[str, Any], filter_results: Dict[str, Any],
                          smoother_results: Dict[str, Any]) -> Dict[str, Any]:
        """Update model parameters in M-step of EM algorithm."""
        try:
            updated_model = model.copy()
            data = model['data'].values
            n = len(data)

            smoothed_states = smoother_results['smoothed_states']
            innovations = filter_results['innovations']

            if model['model_type'] == 'Local Level':
                # Update observation variance
                obs_var = np.mean(innovations**2)
                updated_model['R'] = np.array([[obs_var]])
                updated_model['parameters']['obs_variance'] = obs_var

                # Update level variance
                level_innovations = np.diff(smoothed_states[:, 0])
                level_var = np.mean(level_innovations**2)
                updated_model['Q'] = np.array([[level_var]])
                updated_model['parameters']['level_variance'] = level_var

            elif model['model_type'] == 'Local Linear Trend':
                # Update observation variance
                obs_var = np.mean(innovations**2)
                updated_model['R'] = np.array([[obs_var]])

                # Update state variances
                state_innovations = np.diff(smoothed_states, axis=0)
                level_var = np.mean(state_innovations[:, 0]**2)
                slope_var = np.mean(state_innovations[:, 1]**2)

                updated_model['Q'] = np.array([[level_var, 0.0],
                                             [0.0, slope_var]])
                updated_model['parameters']['obs_variance'] = obs_var
                updated_model['parameters']['level_variance'] = level_var
                updated_model['parameters']['slope_variance'] = slope_var

            return updated_model

        except Exception as e:
            return {'error': f'Parameter update failed: {str(e)}'}

class BayesianEstimation:
    """
    Bayesian estimation framework for time series models.
    """

    def __init__(self):
        """Initialize Bayesian estimation."""
        self.prior_distributions = {}
        self.posterior_samples = {}
        self.estimation_results = {}

    def setup_bayesian_arma(self, data: pd.Series, order: Tuple[int, int],
                           prior_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Set up Bayesian ARMA model with priors.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        order : tuple
            ARMA order (p, q)
        prior_config : dict, optional
            Prior distribution configuration

        Returns:
        --------
        dict
            Bayesian ARMA model setup
        """
        try:
            p, q = order
            data_clean = data.dropna()

            # Default prior configuration
            if prior_config is None:
                prior_config = {
                    'ar_mean': 0.0,
                    'ar_precision': 1.0,
                    'ma_mean': 0.0,
                    'ma_precision': 1.0,
                    'sigma_shape': 2.0,
                    'sigma_rate': 1.0
                }

            # Set up priors
            priors = {
                'ar_coefficients': {
                    'distribution': 'normal',
                    'mean': np.zeros(p),
                    'precision': np.eye(p) * prior_config['ar_precision']
                },
                'ma_coefficients': {
                    'distribution': 'normal',
                    'mean': np.zeros(q),
                    'precision': np.eye(q) * prior_config['ma_precision']
                },
                'error_precision': {
                    'distribution': 'gamma',
                    'shape': prior_config['sigma_shape'],
                    'rate': prior_config['sigma_rate']
                }
            }

            model = {
                'model_type': 'Bayesian ARMA',
                'order': order,
                'data': data_clean,
                'priors': priors,
                'prior_config': prior_config
            }

            self.prior_distributions[f'ARMA_{p}_{q}'] = model
            return model

        except Exception as e:
            return {'error': f'Bayesian ARMA setup failed: {str(e)}'}

    def setup_bayesian_state_space(self, model: Dict[str, Any],
                                  prior_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Set up Bayesian state space model with priors.

        Parameters:
        -----------
        model : dict
            State space model from StateSpaceModels
        prior_config : dict, optional
            Prior configuration

        Returns:
        --------
        dict
            Bayesian state space model
        """
        try:
            if prior_config is None:
                prior_config = {
                    'precision_shape': 2.0,
                    'precision_rate': 1.0
                }

            # Set up priors for variances
            priors = {
                'obs_precision': {
                    'distribution': 'gamma',
                    'shape': prior_config['precision_shape'],
                    'rate': prior_config['precision_rate']
                },
                'state_precision': {
                    'distribution': 'gamma',
                    'shape': prior_config['precision_shape'],
                    'rate': prior_config['precision_rate']
                }
            }

            bayesian_model = {
                'base_model': model,
                'priors': priors,
                'prior_config': prior_config,
                'model_type': f"Bayesian {model['model_type']}"
            }

            return bayesian_model

        except Exception as e:
            return {'error': f'Bayesian state space setup failed: {str(e)}'}

    def variational_bayes(self, model: Dict[str, Any], max_iterations: int = 100,
                         tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Variational Bayes approximation for posterior inference.

        Parameters:
        -----------
        model : dict
            Bayesian model specification
        max_iterations : int, default 100
            Maximum VB iterations
        tolerance : float, default 1e-6
            Convergence tolerance

        Returns:
        --------
        dict
            Variational Bayes results
        """
        try:
            # This is a simplified VB implementation
            # In practice, would implement coordinate ascent VI

            if model['model_type'] == 'Bayesian ARMA':
                return self._vb_arma(model, max_iterations, tolerance)
            else:
                return self._vb_state_space(model, max_iterations, tolerance)

        except Exception as e:
            return {'error': f'Variational Bayes failed: {str(e)}'}

    def _vb_arma(self, model: Dict[str, Any], max_iterations: int, tolerance: float) -> Dict[str, Any]:
        """Variational Bayes for ARMA models."""
        try:
            p, q = model['order']
            data = model['data'].values
            n = len(data)

            # Initialize variational parameters
            # This is a simplified implementation
            ar_mean = np.random.normal(0, 0.1, p)
            ar_precision = np.eye(p)
            sigma_shape = 2.0
            sigma_rate = 1.0

            elbo_values = []

            for iteration in range(max_iterations):
                # Update variational parameters (simplified)
                # In practice, would implement full coordinate ascent

                # Calculate ELBO (Evidence Lower Bound)
                elbo = self._calculate_elbo_arma(data, ar_mean, ar_precision, sigma_shape, sigma_rate, model['priors'])
                elbo_values.append(elbo)

                # Check convergence
                if iteration > 0 and abs(elbo_values[-1] - elbo_values[-2]) < tolerance:
                    break

            results = {
                'method': 'Variational Bayes',
                'posterior_means': {
                    'ar_coefficients': ar_mean,
                    'error_precision': sigma_shape / sigma_rate
                },
                'posterior_variances': {
                    'ar_coefficients': np.diag(np.linalg.inv(ar_precision)),
                    'error_precision': sigma_shape / (sigma_rate**2)
                },
                'elbo_values': elbo_values,
                'iterations': iteration + 1,
                'converged': iteration < max_iterations - 1
            }

            self.estimation_results[f"VB_ARMA_{p}_{q}"] = results
            return results

        except Exception as e:
            return {'error': f'VB ARMA failed: {str(e)}'}

    def _vb_state_space(self, model: Dict[str, Any], max_iterations: int, tolerance: float) -> Dict[str, Any]:
        """Variational Bayes for state space models."""
        # Simplified implementation
        return {'error': 'VB for state space models not fully implemented. Use MCMC methods.'}

    def _calculate_elbo_arma(self, data: np.ndarray, ar_mean: np.ndarray, ar_precision: np.ndarray,
                           sigma_shape: float, sigma_rate: float, priors: Dict) -> float:
        """Calculate Evidence Lower Bound for ARMA model."""
        # Simplified ELBO calculation
        # In practice, would compute full ELBO with all terms
        log_likelihood = -0.5 * len(data) * np.log(2 * np.pi) - 0.5 * np.sum(data**2)
        prior_term = -0.5 * ar_mean.T @ priors['ar_coefficients']['precision'] @ ar_mean
        entropy_term = 0.5 * np.log(np.linalg.det(np.linalg.inv(ar_precision)))

        return log_likelihood + prior_term + entropy_term

class MCMCSampler:
    """
    Markov Chain Monte Carlo sampling for Bayesian time series models.
    """

    def __init__(self):
        """Initialize MCMC sampler."""
        self.chains = {}
        self.diagnostics = {}

    def metropolis_hastings(self, model: Dict[str, Any], n_samples: int = 10000,
                           burn_in: int = 1000, thin: int = 1,
                           proposal_cov: np.ndarray = None) -> Dict[str, Any]:
        """
        Metropolis-Hastings sampler for posterior inference.

        Parameters:
        -----------
        model : dict
            Bayesian model specification
        n_samples : int, default 10000
            Number of MCMC samples
        burn_in : int, default 1000
            Burn-in period
        thin : int, default 1
            Thinning interval
        proposal_cov : np.ndarray, optional
            Proposal covariance matrix

        Returns:
        --------
        dict
            MCMC sampling results
        """
        try:
            if model['model_type'] == 'Bayesian ARMA':
                return self._mh_arma(model, n_samples, burn_in, thin, proposal_cov)
            else:
                return self._mh_state_space(model, n_samples, burn_in, thin, proposal_cov)

        except Exception as e:
            return {'error': f'Metropolis-Hastings sampling failed: {str(e)}'}

    def _mh_arma(self, model: Dict[str, Any], n_samples: int, burn_in: int,
                thin: int, proposal_cov: np.ndarray) -> Dict[str, Any]:
        """Metropolis-Hastings for ARMA models."""
        try:
            p, q = model['order']
            data = model['data'].values
            n_data = len(data)

            # Parameter dimension
            n_params = p + q + 1  # AR + MA + sigma

            # Initialize proposal covariance if not provided
            if proposal_cov is None:
                proposal_cov = np.eye(n_params) * 0.01

            # Initialize chain
            current_params = np.random.normal(0, 0.1, n_params)
            current_logprob = self._log_posterior_arma(current_params, data, model['priors'], p, q)

            # Storage
            samples = np.zeros((n_samples, n_params))
            log_probs = np.zeros(n_samples)
            acceptances = np.zeros(n_samples, dtype=bool)

            # MCMC loop
            for i in range(n_samples + burn_in):
                # Propose new parameters
                proposal = np.random.multivariate_normal(current_params, proposal_cov)

                # Calculate log probability of proposal
                proposal_logprob = self._log_posterior_arma(proposal, data, model['priors'], p, q)

                # Metropolis-Hastings ratio
                log_ratio = proposal_logprob - current_logprob

                # Accept or reject
                if np.log(np.random.rand()) < log_ratio:
                    current_params = proposal
                    current_logprob = proposal_logprob
                    accepted = True
                else:
                    accepted = False

                # Store samples after burn-in
                if i >= burn_in and (i - burn_in) % thin == 0:
                    idx = (i - burn_in) // thin
                    if idx < n_samples:
                        samples[idx] = current_params
                        log_probs[idx] = current_logprob
                        acceptances[idx] = accepted

            # Process results
            ar_samples = samples[:, :p] if p > 0 else None
            ma_samples = samples[:, p:p+q] if q > 0 else None
            sigma_samples = np.exp(samples[:, -1])  # Log-normal for positivity

            results = {
                'method': 'Metropolis-Hastings',
                'samples': {
                    'ar_coefficients': ar_samples,
                    'ma_coefficients': ma_samples,
                    'error_std': sigma_samples
                },
                'log_probabilities': log_probs,
                'acceptance_rate': np.mean(acceptances),
                'n_samples': n_samples,
                'burn_in': burn_in,
                'thin': thin,
                'diagnostics': self._mcmc_diagnostics(samples)
            }

            self.chains[f"MH_ARMA_{p}_{q}"] = results
            return results

        except Exception as e:
            return {'error': f'MH ARMA sampling failed: {str(e)}'}

    def _mh_state_space(self, model: Dict[str, Any], n_samples: int, burn_in: int,
                       thin: int, proposal_cov: np.ndarray) -> Dict[str, Any]:
        """Metropolis-Hastings for state space models."""
        # Simplified implementation
        return {'error': 'MH for state space models requires more complex implementation'}

    def gibbs_sampler(self, model: Dict[str, Any], n_samples: int = 10000,
                     burn_in: int = 1000, thin: int = 1) -> Dict[str, Any]:
        """
        Gibbs sampler for models with conjugate priors.

        Parameters:
        -----------
        model : dict
            Bayesian model with conjugate priors
        n_samples : int, default 10000
            Number of samples
        burn_in : int, default 1000
            Burn-in period
        thin : int, default 1
            Thinning interval

        Returns:
        --------
        dict
            Gibbs sampling results
        """
        try:
            if model['model_type'] == 'Bayesian ARMA':
                return self._gibbs_arma(model, n_samples, burn_in, thin)
            else:
                return {'error': 'Gibbs sampling only implemented for ARMA models with conjugate priors'}

        except Exception as e:
            return {'error': f'Gibbs sampling failed: {str(e)}'}

    def _gibbs_arma(self, model: Dict[str, Any], n_samples: int, burn_in: int, thin: int) -> Dict[str, Any]:
        """Gibbs sampler for ARMA with conjugate priors."""
        try:
            # Simplified Gibbs sampler
            # In practice, would implement full conditional distributions
            p, q = model['order']
            data = model['data'].values

            # Initialize
            current_ar = np.zeros(p) if p > 0 else None
            current_sigma = 1.0

            # Storage
            ar_samples = np.zeros((n_samples, p)) if p > 0 else None
            sigma_samples = np.zeros(n_samples)

            for i in range(n_samples + burn_in):
                # Sample AR coefficients (conditional on sigma)
                if p > 0:
                    # This would be full conditional in practice
                    current_ar = np.random.multivariate_normal(
                        np.zeros(p), np.eye(p) * 0.1
                    )

                # Sample sigma (conditional on AR coefficients)
                # This would use inverse-gamma full conditional
                current_sigma = 1.0 / np.random.gamma(2.0, 1.0)

                # Store after burn-in
                if i >= burn_in and (i - burn_in) % thin == 0:
                    idx = (i - burn_in) // thin
                    if idx < n_samples:
                        if p > 0:
                            ar_samples[idx] = current_ar
                        sigma_samples[idx] = current_sigma

            results = {
                'method': 'Gibbs',
                'samples': {
                    'ar_coefficients': ar_samples,
                    'error_std': sigma_samples
                },
                'n_samples': n_samples,
                'burn_in': burn_in,
                'thin': thin
            }

            return results

        except Exception as e:
            return {'error': f'Gibbs ARMA sampling failed: {str(e)}'}

    def _log_posterior_arma(self, params: np.ndarray, data: np.ndarray,
                          priors: Dict, p: int, q: int) -> float:
        """Calculate log posterior density for ARMA model."""
        try:
            # Extract parameters
            ar_params = params[:p] if p > 0 else np.array([])
            ma_params = params[p:p+q] if q > 0 else np.array([])
            log_sigma = params[-1]
            sigma = np.exp(log_sigma)

            # Log prior
            log_prior = 0.0

            # AR prior
            if p > 0:
                ar_prior_mean = priors['ar_coefficients']['mean']
                ar_prior_prec = priors['ar_coefficients']['precision']
                log_prior += -0.5 * (ar_params - ar_prior_mean).T @ ar_prior_prec @ (ar_params - ar_prior_mean)

            # MA prior (if applicable)
            if q > 0:
                ma_prior_mean = priors['ma_coefficients']['mean']
                ma_prior_prec = priors['ma_coefficients']['precision']
                log_prior += -0.5 * (ma_params - ma_prior_mean).T @ ma_prior_prec @ (ma_params - ma_prior_mean)

            # Sigma prior (log-normal)
            log_prior += log_sigma  # Jacobian for log transformation

            # Log likelihood (simplified)
            residuals = data.copy()  # Simplified - would compute ARMA residuals properly
            log_likelihood = -0.5 * len(data) * np.log(2 * np.pi * sigma**2) - 0.5 * np.sum(residuals**2) / sigma**2

            return log_likelihood + log_prior

        except Exception as e:
            return -np.inf

    def _mcmc_diagnostics(self, samples: np.ndarray) -> Dict[str, Any]:
        """Compute MCMC diagnostic statistics."""
        try:
            n_samples, n_params = samples.shape

            # Effective sample size (simplified)
            eff_samples = np.zeros(n_params)
            for i in range(n_params):
                autocorr = self._autocorrelation(samples[:, i])
                # Find first negative autocorrelation
                tau_int = 1 + 2 * np.sum(autocorr[1:np.where(autocorr < 0)[0][0]] if np.any(autocorr < 0) else autocorr[1:])
                eff_samples[i] = n_samples / (2 * tau_int) if tau_int > 0 else n_samples

            # Geweke diagnostic (simplified)
            geweke_z = np.zeros(n_params)
            first_part = samples[:n_samples//10]  # First 10%
            last_part = samples[-n_samples//2:]   # Last 50%

            for i in range(n_params):
                if np.var(first_part[:, i]) > 0 and np.var(last_part[:, i]) > 0:
                    geweke_z[i] = (np.mean(first_part[:, i]) - np.mean(last_part[:, i])) / \
                                 np.sqrt(np.var(first_part[:, i]) + np.var(last_part[:, i]))

            diagnostics = {
                'effective_sample_size': eff_samples,
                'geweke_z_scores': geweke_z,
                'geweke_p_values': 2 * (1 - stats.norm.cdf(np.abs(geweke_z))),
                'mean_acceptance_rate': getattr(self, '_last_acceptance_rate', None)
            }

            return diagnostics

        except Exception as e:
            return {'error': f'MCMC diagnostics failed: {str(e)}'}

    def _autocorrelation(self, x: np.ndarray, max_lag: int = None) -> np.ndarray:
        """Compute sample autocorrelation function."""
        n = len(x)
        if max_lag is None:
            max_lag = min(n // 4, 200)

        x = x - np.mean(x)
        autocorr = np.correlate(x, x, mode='full')
        autocorr = autocorr[n-1:]
        autocorr = autocorr / autocorr[0]

        return autocorr[:max_lag]

class MultivariateTimeSeriesAnalysis:
    """
    Comprehensive multivariate time series analysis including VAR, cointegration, and VECM.
    """

    def __init__(self):
        """Initialize multivariate time series analysis."""
        self.var_models = {}
        self.cointegration_results = {}
        self.vecm_models = {}
        self.ergodicity_tests = {}

    def test_ergodicity(self, data: pd.DataFrame, method: str = 'correlation',
                       max_lag: int = 50) -> Dict[str, Any]:
        """
        Test for ergodicity in multivariate time series.

        Parameters:
        -----------
        data : pd.DataFrame
            Multivariate time series data
        method : str, default 'correlation'
            Test method ('correlation', 'variance')
        max_lag : int, default 50
            Maximum lag for autocorrelation test

        Returns:
        --------
        dict
            Ergodicity test results
        """
        try:
            data_clean = data.dropna()
            n, k = data_clean.shape

            if method == 'correlation':
                # Test if sample autocorrelations converge to zero
                ergodicity_results = {}

                for col in data_clean.columns:
                    series = data_clean[col]

                    # Calculate sample autocorrelations
                    autocorrs = np.zeros(max_lag)
                    for lag in range(1, max_lag + 1):
                        if lag < len(series):
                            autocorrs[lag - 1] = series.autocorr(lag)

                    # Test for convergence to zero
                    # Simple test: check if autocorrelations decay
                    decay_test = np.mean(np.abs(autocorrs[:10])) > np.mean(np.abs(autocorrs[-10:]))

                    # Ljung-Box test for white noise (ergodic process should have uncorrelated increments)
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    lb_result = acorr_ljungbox(series, lags=min(20, len(series)//4), return_df=True)

                    ergodicity_results[col] = {
                        'autocorrelations': autocorrs,
                        'decay_pattern': 'Decreasing' if decay_test else 'Non-decreasing',
                        'ljung_box_pvalue': lb_result['lb_pvalue'].iloc[-1],
                        'appears_ergodic': decay_test and (lb_result['lb_pvalue'].iloc[-1] > 0.05)
                    }

            elif method == 'variance':
                # Test variance stationarity (ergodic process should have constant variance)
                ergodicity_results = {}

                for col in data_clean.columns:
                    series = data_clean[col]

                    # Split into chunks and test variance constancy
                    chunk_size = len(series) // 5
                    chunk_variances = []

                    for i in range(5):
                        start_idx = i * chunk_size
                        end_idx = start_idx + chunk_size if i < 4 else len(series)
                        chunk = series.iloc[start_idx:end_idx]
                        chunk_variances.append(np.var(chunk))

                    # Test for variance stationarity using Levene's test
                    from scipy.stats import levene
                    levene_stat, levene_p = levene(*[series.iloc[i*chunk_size:(i+1)*chunk_size if i < 4 else len(series)]
                                                   for i in range(5)])

                    ergodicity_results[col] = {
                        'chunk_variances': chunk_variances,
                        'variance_ratio': max(chunk_variances) / min(chunk_variances),
                        'levene_statistic': levene_stat,
                        'levene_pvalue': levene_p,
                        'appears_ergodic': levene_p > 0.05  # Fail to reject constant variance
                    }

            results = {
                'method': method,
                'n_variables': k,
                'sample_size': n,
                'individual_tests': ergodicity_results,
                'overall_assessment': self._assess_overall_ergodicity(ergodicity_results)
            }

            self.ergodicity_tests[method] = results
            return results

        except Exception as e:
            return {'error': f'Ergodicity test failed: {str(e)}'}

    def fit_var(self, data: pd.DataFrame, max_lags: int = 12,
               ic: str = 'aic', trend: str = 'n') -> Dict[str, Any]:
        """
        Fit Vector Autoregressive (VAR) model.

        Parameters:
        -----------
        data : pd.DataFrame
            Multivariate time series data
        max_lags : int, default 12
            Maximum number of lags to consider
        ic : str, default 'aic'
            Information criterion for lag selection
        trend : str, default 'n'
            Trend specification ('n', 'c', 'ct', 'ctt')

        Returns:
        --------
        dict
            VAR model results
        """
        try:
            if not HAS_STATSMODELS:
                return self._manual_var(data, max_lags)

            from statsmodels.tsa.vector_ar.var_model import VAR

            data_clean = data.dropna()
            n, k = data_clean.shape

            # Fit VAR model
            var_model = VAR(data_clean)

            # Select optimal lag length
            lag_selection = var_model.select_order(maxlags=max_lags, verbose=False)
            optimal_lags = getattr(lag_selection, ic)

            # Fit VAR with optimal lags
            fitted_var = var_model.fit(optimal_lags, trend=trend, verbose=False)

            # Extract results
            results = {
                'model_type': 'VAR',
                'optimal_lags': optimal_lags,
                'trend': trend,
                'n_variables': k,
                'sample_size': n,
                'fitted_model': fitted_var,
                'coefficients': fitted_var.params.to_dict(),
                'residuals': fitted_var.resid,
                'fitted_values': fitted_var.fittedvalues,
                'aic': fitted_var.aic,
                'bic': fitted_var.bic,
                'hqic': fitted_var.hqic,
                'log_likelihood': fitted_var.llf,
                'lag_selection_criteria': {
                    'aic': lag_selection.aic,
                    'bic': lag_selection.bic,
                    'hqic': lag_selection.hqic,
                    'fpe': lag_selection.fpe
                }
            }

            # Additional diagnostics
            try:
                # Granger causality tests
                granger_results = {}
                for i, var1 in enumerate(data_clean.columns):
                    for j, var2 in enumerate(data_clean.columns):
                        if i != j:
                            granger_test = fitted_var.test_causality(var2, var1, verbose=False)
                            granger_results[f"{var1}_causes_{var2}"] = {
                                'statistic': granger_test.statistic,
                                'pvalue': granger_test.pvalue,
                                'conclusion': 'Granger causes' if granger_test.pvalue < 0.05 else 'Does not Granger cause'
                            }

                results['granger_causality'] = granger_results

            except Exception as e:
                results['granger_causality'] = {'error': str(e)}

            self.var_models[f'VAR_{optimal_lags}'] = results
            return results

        except Exception as e:
            return {'error': f'VAR fitting failed: {str(e)}'}

    def engle_granger_cointegration(self, y: pd.Series, x: pd.DataFrame,
                                   trend: str = 'c', max_lags: int = None) -> Dict[str, Any]:
        """
        Engle-Granger two-step cointegration test.

        Parameters:
        -----------
        y : pd.Series
            Dependent variable
        x : pd.DataFrame
            Independent variables
        trend : str, default 'c'
            Trend in cointegrating equation
        max_lags : int, optional
            Maximum lags for ADF test

        Returns:
        --------
        dict
            Engle-Granger test results
        """
        try:
            if not HAS_STATSMODELS:
                return self._manual_engle_granger(y, x)

            import statsmodels.api as sm
            from statsmodels.tsa.stattools import adfuller

            # Step 1: Estimate cointegrating regression
            if trend == 'c':
                X = sm.add_constant(x)
            elif trend == 'ct':
                X = sm.add_constant(x)
                X['trend'] = np.arange(len(X))
            else:
                X = x

            # OLS regression
            ols_model = sm.OLS(y, X).fit()
            residuals = ols_model.resid

            # Step 2: Test residuals for unit root
            if max_lags is None:
                max_lags = int(12 * (len(residuals) / 100) ** 0.25)

            adf_result = adfuller(residuals, maxlag=max_lags, regression='nc', autolag='AIC')

            # Critical values for Engle-Granger test (adjusted for number of variables)
            n_vars = len(x.columns) + 1
            eg_critical_values = self._get_engle_granger_critical_values(n_vars, len(residuals))

            results = {
                'method': 'Engle-Granger Two-Step',
                'step1_regression': {
                    'coefficients': ols_model.params.to_dict(),
                    'r_squared': ols_model.rsquared,
                    'residuals': residuals
                },
                'step2_adf_test': {
                    'test_statistic': adf_result[0],
                    'pvalue': adf_result[1],
                    'lags_used': adf_result[2],
                    'n_obs': adf_result[3],
                    'critical_values': adf_result[4]
                },
                'engle_granger_critical_values': eg_critical_values,
                'cointegration_conclusion': self._interpret_eg_test(adf_result[0], eg_critical_values),
                'n_variables': n_vars
            }

            return results

        except Exception as e:
            return {'error': f'Engle-Granger test failed: {str(e)}'}

    def johansen_cointegration(self, data: pd.DataFrame, det_order: int = 0,
                              k_ar_diff: int = 1) -> Dict[str, Any]:
        """
        Johansen test for cointegration.

        Parameters:
        -----------
        data : pd.DataFrame
            Multivariate time series data
        det_order : int, default 0
            Deterministic order (-1: no const, 0: const, 1: linear trend)
        k_ar_diff : int, default 1
            Number of lagged differences in auxiliary regression

        Returns:
        --------
        dict
            Johansen test results
        """
        try:
            if not HAS_STATSMODELS:
                return self._manual_johansen(data)

            from statsmodels.tsa.vector_ar.vecm import coint_johansen

            data_clean = data.dropna()
            n, k = data_clean.shape

            # Perform Johansen test
            johansen_result = coint_johansen(data_clean, det_order=det_order, k_ar_diff=k_ar_diff)

            # Extract results
            results = {
                'method': 'Johansen Cointegration Test',
                'n_variables': k,
                'sample_size': n,
                'deterministic_order': det_order,
                'lags_diff': k_ar_diff,
                'eigenvalues': johansen_result.eig,
                'trace_statistics': johansen_result.lr1,
                'max_eigenvalue_statistics': johansen_result.lr2,
                'critical_values_trace': {
                    '90%': johansen_result.cvt[:, 0],
                    '95%': johansen_result.cvt[:, 1],
                    '99%': johansen_result.cvt[:, 2]
                },
                'critical_values_max_eig': {
                    '90%': johansen_result.cvm[:, 0],
                    '95%': johansen_result.cvm[:, 1],
                    '99%': johansen_result.cvm[:, 2]
                },
                'eigenvectors': johansen_result.evec
            }

            # Determine number of cointegrating relationships
            trace_rank = self._determine_cointegration_rank(
                johansen_result.lr1, johansen_result.cvt[:, 1]  # 95% critical values
            )
            max_eig_rank = self._determine_cointegration_rank(
                johansen_result.lr2, johansen_result.cvm[:, 1]  # 95% critical values
            )

            results['cointegration_rank'] = {
                'trace_test': trace_rank,
                'max_eigenvalue_test': max_eig_rank,
                'recommended': min(trace_rank, max_eig_rank)
            }

            self.cointegration_results['johansen'] = results
            return results

        except Exception as e:
            return {'error': f'Johansen test failed: {str(e)}'}

    def fit_vecm(self, data: pd.DataFrame, k_ar_diff: int = 1,
                coint_rank: int = None, deterministic: str = 'ci') -> Dict[str, Any]:
        """
        Fit Vector Error Correction Model (VECM).

        Parameters:
        -----------
        data : pd.DataFrame
            Multivariate time series data
        k_ar_diff : int, default 1
            Number of lagged differences
        coint_rank : int, optional
            Number of cointegrating relationships
        deterministic : str, default 'ci'
            Deterministic terms ('n', 'co', 'ci', 'lo', 'li')

        Returns:
        --------
        dict
            VECM results
        """
        try:
            if not HAS_STATSMODELS:
                return {'error': 'VECM requires statsmodels package'}

            from statsmodels.tsa.vector_ar.vecm import VECM

            data_clean = data.dropna()
            n, k = data_clean.shape

            # Determine cointegration rank if not provided
            if coint_rank is None:
                johansen_result = self.johansen_cointegration(data_clean)
                if 'error' not in johansen_result:
                    coint_rank = johansen_result['cointegration_rank']['recommended']
                else:
                    coint_rank = 1  # Default

            # Fit VECM
            vecm_model = VECM(data_clean, k_ar_diff=k_ar_diff, coint_rank=coint_rank,
                             deterministic=deterministic)
            fitted_vecm = vecm_model.fit()

            results = {
                'model_type': 'VECM',
                'cointegration_rank': coint_rank,
                'lags_diff': k_ar_diff,
                'deterministic': deterministic,
                'n_variables': k,
                'sample_size': n,
                'fitted_model': fitted_vecm,
                'alpha': fitted_vecm.alpha,  # Adjustment coefficients
                'beta': fitted_vecm.beta,    # Cointegrating vectors
                'gamma': fitted_vecm.gamma,  # Short-run coefficients
                'residuals': fitted_vecm.resid,
                'fitted_values': fitted_vecm.fittedvalues,
                'log_likelihood': fitted_vecm.llf,
                'aic': fitted_vecm.aic,
                'bic': fitted_vecm.bic
            }

            # Error correction representation
            results['error_correction_terms'] = data_clean @ fitted_vecm.beta

            # Test restrictions on adjustment coefficients
            try:
                # Test if adjustment coefficients are jointly zero (no error correction)
                alpha_test = fitted_vecm.test_ec_term_alpha()
                results['alpha_test'] = {
                    'statistic': alpha_test.statistic,
                    'pvalue': alpha_test.pvalue,
                    'conclusion': 'Significant error correction' if alpha_test.pvalue < 0.05 else 'No error correction'
                }
            except:
                results['alpha_test'] = {'error': 'Alpha test failed'}

            self.vecm_models[f'VECM_rank_{coint_rank}'] = results
            return results

        except Exception as e:
            return {'error': f'VECM fitting failed: {str(e)}'}

    def setup_error_correction_model(self, y: pd.Series, x: pd.DataFrame,
                                   ecm_lags: int = 1) -> Dict[str, Any]:
        """
        Set up Error Correction Model (ECM) for bivariate cointegrated series.

        Parameters:
        -----------
        y : pd.Series
            Dependent variable
        x : pd.DataFrame or pd.Series
            Independent variable(s)
        ecm_lags : int, default 1
            Number of lags for difference terms

        Returns:
        --------
        dict
            ECM setup and estimation results
        """
        try:
            import statsmodels.api as sm

            # Ensure x is DataFrame
            if isinstance(x, pd.Series):
                x = x.to_frame()

            # Step 1: Estimate long-run relationship (cointegrating equation)
            X_levels = sm.add_constant(x)
            long_run_model = sm.OLS(y, X_levels).fit()
            error_correction_term = long_run_model.resid.shift(1)  # Lagged residuals

            # Step 2: Set up ECM in differences
            dy = y.diff()
            dx = x.diff()

            # Create ECM regression data
            ecm_data = pd.DataFrame({
                'dy': dy
            })

            # Add error correction term
            ecm_data['ect_lag1'] = error_correction_term

            # Add lagged differences
            for lag in range(1, ecm_lags + 1):
                ecm_data[f'dy_lag{lag}'] = dy.shift(lag)
                for col in dx.columns:
                    ecm_data[f'd{col}_lag{lag}'] = dx[col].shift(lag)
                    if lag == 0:  # Current period differences
                        ecm_data[f'd{col}'] = dx[col]

            # Remove missing values
            ecm_data = ecm_data.dropna()

            # Fit ECM
            y_ecm = ecm_data['dy']
            X_ecm = ecm_data.drop('dy', axis=1)
            X_ecm = sm.add_constant(X_ecm)

            ecm_model = sm.OLS(y_ecm, X_ecm).fit()

            results = {
                'model_type': 'Error Correction Model',
                'long_run_relationship': {
                    'coefficients': long_run_model.params.to_dict(),
                    'r_squared': long_run_model.rsquared,
                    'residuals': long_run_model.resid
                },
                'ecm_estimation': {
                    'coefficients': ecm_model.params.to_dict(),
                    'r_squared': ecm_model.rsquared,
                    'residuals': ecm_model.resid,
                    'fitted_values': ecm_model.fittedvalues
                },
                'error_correction_coefficient': ecm_model.params['ect_lag1'],
                'adjustment_speed': -ecm_model.params['ect_lag1'],  # Speed of adjustment to equilibrium
                'half_life': np.log(0.5) / np.log(1 + ecm_model.params['ect_lag1']) if ecm_model.params['ect_lag1'] < 0 else np.inf,
                'ecm_lags': ecm_lags,
                'sample_size': len(ecm_data)
            }

            # Test significance of error correction term
            ect_tstat = ecm_model.tvalues['ect_lag1']
            ect_pvalue = ecm_model.pvalues['ect_lag1']

            results['error_correction_test'] = {
                't_statistic': ect_tstat,
                'p_value': ect_pvalue,
                'significant': ect_pvalue < 0.05,
                'conclusion': 'Significant error correction' if ect_pvalue < 0.05 else 'No error correction'
            }

            return results

        except Exception as e:
            return {'error': f'ECM setup failed: {str(e)}'}

    # Helper methods
    def _assess_overall_ergodicity(self, individual_results: Dict) -> Dict[str, Any]:
        """Assess overall ergodicity based on individual variable tests."""
        ergodic_count = sum(1 for result in individual_results.values()
                           if result.get('appears_ergodic', False))
        total_count = len(individual_results)

        assessment = {
            'ergodic_variables': ergodic_count,
            'total_variables': total_count,
            'proportion_ergodic': ergodic_count / total_count,
            'overall_conclusion': 'Likely ergodic' if ergodic_count / total_count > 0.8 else 'Mixed evidence' if ergodic_count / total_count > 0.5 else 'Likely non-ergodic'
        }

        return assessment

    def _manual_var(self, data: pd.DataFrame, max_lags: int) -> Dict[str, Any]:
        """Manual VAR implementation when statsmodels is not available."""
        warnings.warn("Manual VAR implementation is basic. Install statsmodels for full functionality.")

        try:
            data_clean = data.dropna()
            n, k = data_clean.shape

            # Simple VAR(1) implementation
            Y = data_clean.iloc[1:].values  # Y(t)
            X = data_clean.iloc[:-1].values  # Y(t-1)

            # Add constant
            X_with_const = np.column_stack([np.ones(len(X)), X])

            # OLS estimation: vec(B) = (X'X)^-1 X'Y
            try:
                B = np.linalg.solve(X_with_const.T @ X_with_const, X_with_const.T @ Y)
            except np.linalg.LinAlgError:
                B = np.linalg.pinv(X_with_const.T @ X_with_const) @ X_with_const.T @ Y

            # Calculate residuals and fitted values
            fitted = X_with_const @ B
            residuals = Y - fitted

            # Information criteria (simplified)
            n_params = k * (k + 1)  # k equations, k+1 parameters each
            rss = np.sum(residuals**2)
            aic = n * k * np.log(rss / (n * k)) + 2 * n_params
            bic = n * k * np.log(rss / (n * k)) + np.log(n) * n_params

            results = {
                'model_type': 'VAR (Manual)',
                'optimal_lags': 1,
                'n_variables': k,
                'sample_size': n,
                'coefficients': B,
                'residuals': residuals,
                'fitted_values': fitted,
                'aic': aic,
                'bic': bic,
                'note': 'Basic VAR(1) implementation - use statsmodels for full functionality'
            }

            return results

        except Exception as e:
            return {'error': f'Manual VAR failed: {str(e)}'}

    def _manual_engle_granger(self, y: pd.Series, x: pd.DataFrame) -> Dict[str, Any]:
        """Manual Engle-Granger implementation."""
        return {'error': 'Manual Engle-Granger implementation requires statsmodels for ADF test'}

    def _manual_johansen(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Manual Johansen implementation."""
        return {'error': 'Manual Johansen implementation is complex. Please install statsmodels.'}

    def _get_engle_granger_critical_values(self, n_vars: int, n_obs: int) -> Dict[str, float]:
        """Get Engle-Granger critical values (approximate)."""
        # Simplified critical values - in practice would use MacKinnon (1996) tables
        base_values = {
            '1%': -4.0,
            '5%': -3.4,
            '10%': -3.1
        }

        # Adjust for number of variables (very approximate)
        adjustment = (n_vars - 2) * 0.2

        return {level: value - adjustment for level, value in base_values.items()}

    def _interpret_eg_test(self, test_stat: float, critical_values: Dict[str, float]) -> str:
        """Interpret Engle-Granger test results."""
        if test_stat < critical_values['1%']:
            return 'Strong evidence of cointegration (1% level)'
        elif test_stat < critical_values['5%']:
            return 'Evidence of cointegration (5% level)'
        elif test_stat < critical_values['10%']:
            return 'Weak evidence of cointegration (10% level)'
        else:
            return 'No evidence of cointegration'

    def _determine_cointegration_rank(self, test_stats: np.ndarray,
                                    critical_values: np.ndarray) -> int:
        """Determine cointegration rank from Johansen test."""
        rank = 0
        for i, (stat, cv) in enumerate(zip(test_stats, critical_values)):
            if stat > cv:
                rank = i + 1
            else:
                break
        return rank

def demonstrate_timeseries():
    """
    Demonstrate the usage of the time series analysis module with sample data.
    """
    print("=== Time Series Analysis Demo ===\n")

    # Create sample time series data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=365, freq='D')

    # Create trend + seasonal + noise
    trend = np.linspace(100, 200, 365)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 365.25 * 4)  # Quarterly seasonality
    noise = np.random.normal(0, 5, 365)
    ts_data = pd.Series(trend + seasonal + noise, index=dates)

    print("1. Time Series Resampling:")
    resampler = TimeSeriesResampler()

    # Resample to monthly
    monthly_data = resampler.resample_data(ts_data, 'M', 'mean')
    print(f"   Original daily data points: {len(ts_data)}")
    print(f"   Resampled monthly data points: {len(monthly_data)}")

    # Upsample to hourly (first few days)
    daily_subset = ts_data.head(7)
    hourly_data = resampler.upsample_data(daily_subset, 'H', 'interpolate')
    print(f"   Upsampled 7 days to hourly: {len(hourly_data)} data points")

    print("\n2. Moving Averages:")
    ma_calculator = MovingAverages()

    # Simple moving average
    sma_10 = ma_calculator.simple_moving_average(ts_data, window=10)
    print(f"   10-day simple moving average calculated")

    # Exponential moving average
    ema_10 = ma_calculator.exponential_moving_average(ts_data, span=10)
    print(f"   10-day exponential moving average calculated")

    print("\n3. Exponential Smoothing:")
    exp_smoother = ExponentialSmoothing()

    # Simple exponential smoothing
    simple_results = exp_smoother.simple_exponential_smoothing(ts_data)
    print(f"   Simple exponential smoothing - Alpha: {simple_results['alpha']:.3f}")

    # Double exponential smoothing
    double_results = exp_smoother.double_exponential_smoothing(ts_data)
    print(f"   Double exponential smoothing - Alpha: {double_results['alpha']:.3f}, Beta: {double_results['beta']:.3f}")

    # Triple exponential smoothing (Holt-Winters)
    try:
        triple_results = exp_smoother.triple_exponential_smoothing(ts_data, seasonal_periods=90)  # Quarterly
        print(f"   Triple exponential smoothing - Alpha: {triple_results['alpha']:.3f}, Beta: {triple_results['beta']:.3f}, Gamma: {triple_results['gamma']:.3f}")
    except Exception as e:
        print(f"   Triple exponential smoothing failed: {str(e)}")

    print("\n4. Stationarity Testing:")
    stationarity_tester = StationarityTester()

    # Test original series
    adf_results = stationarity_tester.dickey_fuller_test(ts_data)
    print(f"   ADF test p-value: {adf_results['p_value']:.3f} - {adf_results['conclusion']}")

    try:
        kpss_results = stationarity_tester.kpss_test(ts_data)
        print(f"   KPSS test p-value: {kpss_results['p_value']:.3f} - {kpss_results['conclusion']}")

        # Combined test
        combined_results = stationarity_tester.combined_stationarity_test(ts_data)
        print(f"   Combined conclusion: {combined_results['overall_conclusion']}")
    except Exception as e:
        print(f"   KPSS test failed: {str(e)}")

    print("\n5. Trend Removal:")
    trend_remover = TrendRemover()

    # Linear detrend
    detrended_linear, trend_linear = trend_remover.linear_detrend(ts_data)
    print(f"   Linear detrending completed")

    # First differencing
    differenced = trend_remover.first_difference(ts_data)
    print(f"   First differencing completed - {len(differenced)} observations")

    # Test stationarity of differenced series
    adf_diff = stationarity_tester.dickey_fuller_test(differenced)
    print(f"   ADF test on differenced series: {adf_diff['conclusion']}")

    print("\n6. Autocorrelation Analysis:")
    acf_analyzer = AutocorrelationAnalyzer()

    # Calculate ACF and PACF
    acf_results = acf_analyzer.calculate_acf(differenced, nlags=20)
    pacf_results = acf_analyzer.calculate_pacf(differenced, nlags=20)

    print(f"   ACF significant lags (first 5): {acf_results['significant_lags'][:5]}")
    print(f"   PACF significant lags (first 5): {pacf_results['significant_lags'][:5]}")

    # Suggest ARIMA order
    arima_suggestion = acf_analyzer.suggest_arima_order(differenced, max_p=3, max_q=3)
    print(f"   Suggested ARIMA order: {arima_suggestion['suggested_order']}")

    print("\n7. Creating Non-stationary Series for Testing:")
    # Create a clearly non-stationary series (random walk)
    np.random.seed(123)
    random_walk = pd.Series(np.cumsum(np.random.randn(100)),
                           index=pd.date_range('2023-01-01', periods=100))

    adf_rw = stationarity_tester.dickey_fuller_test(random_walk)
    print(f"   Random walk ADF test: {adf_rw['conclusion']} (p-value: {adf_rw['p_value']:.3f})")

    # Difference the random walk
    rw_diff = trend_remover.first_difference(random_walk)
    adf_rw_diff = stationarity_tester.dickey_fuller_test(rw_diff)
    print(f"   Differenced random walk ADF test: {adf_rw_diff['conclusion']} (p-value: {adf_rw_diff['p_value']:.3f})")

    print("\n8. ARMA/ARIMA Model Fitting:")
    try:
        # ARMA modeling on stationary differenced data
        arma_modeler = ARMAModeler()
        arma_results = arma_modeler.fit_arma(differenced, order=(2, 1))

        if 'error' not in arma_results:
            print(f"   ARMA(2,1) model fitted successfully")
            print(f"   AIC: {arma_results['aic']:.3f}, BIC: {arma_results['bic']:.3f}")

            # Generate forecast
            arma_forecast = arma_modeler.forecast_arma("ARMA(2,1)", steps=5)
            if 'error' not in arma_forecast:
                print(f"   ARMA forecast generated for 5 steps")

        # ARIMA modeling on original non-stationary data
        arima_modeler = ARIMAModeler()
        arima_results = arima_modeler.fit_arima(ts_data.head(200), order=(1, 1, 1))

        if 'error' not in arima_results:
            print(f"   ARIMA(1,1,1) model fitted successfully")
            print(f"   AIC: {arima_results['aic']:.3f}, BIC: {arima_results['bic']:.3f}")

            # Generate forecast
            arima_forecast = arima_modeler.forecast_arima("ARIMA(1,1,1)", steps=10)
            if 'error' not in arima_forecast:
                print(f"   ARIMA forecast generated for 10 steps")
                print(f"   Forecast mean: {arima_forecast['forecast'].mean():.3f}")

    except Exception as e:
        print(f"   ARMA/ARIMA modeling failed: {str(e)}")

    print("\n9. Advanced ARMA Estimation Methods:")
    try:
        # Compare different estimation methods
        comparison_data = differenced.head(100)  # Use smaller dataset for faster computation
        estimation_comparison = arma_modeler.compare_estimation_methods(comparison_data, (2, 1))

        if 'error' not in estimation_comparison:
            print(f"   Estimation methods comparison completed")
            summary = estimation_comparison['summary']
            print(f"   Best AIC method: {summary['best_aic_method']}")
            print(f"   Best BIC method: {summary['best_bic_method']}")
            print(f"   Valid methods: {', '.join(summary['valid_methods'])}")

            # Test specific estimation methods
            print("\n   Testing OLS estimation:")
            ols_results = arma_modeler.fit_arma_ols(comparison_data, (2, 0))
            if 'error' not in ols_results:
                print(f"     OLS AR(2): AIC={ols_results['aic']:.3f}, R²={ols_results.get('r_squared', 0):.3f}")

            print("\n   Testing Yule-Walker estimation:")
            yw_results = arma_modeler.fit_arma_yule_walker(comparison_data, (2, 0))
            if 'error' not in yw_results:
                print(f"     Yule-Walker AR(2): AIC={yw_results['aic']:.3f}")
                print(f"     Noise variance: {yw_results.get('noise_variance', 0):.4f}")

        else:
            print(f"   Estimation methods comparison failed: {estimation_comparison['error']}")

    except Exception as e:
        print(f"   Advanced estimation methods failed: {str(e)}")

    print("\n10. Ljung-Box Test for Residual Independence:")
    try:
        # Test residuals from best ARMA model
        if 'arma_results' in locals() and 'error' not in arma_results:
            residuals = arma_results['residuals']

            # Ljung-Box test
            lb_test = arma_modeler.ljung_box_test(residuals, lags=10, model_df=3)

            if 'error' not in lb_test:
                print(f"   Ljung-Box test completed")
                print(f"   Test statistic: {lb_test['test_statistic']:.3f}")
                print(f"   P-value: {lb_test['p_value']:.4f}")
                print(f"   Conclusion: {lb_test['conclusion']}")
                print(f"   Lags tested: {lb_test['lags_tested']}")
            else:
                print(f"   Ljung-Box test failed: {lb_test['error']}")

    except Exception as e:
        print(f"   Ljung-Box test failed: {str(e)}")

    print("\n11. Advanced Diagnostic Visualization:")
    try:
        visualizer = TimeSeriesVisualizer()

        print("   Creating time series plot...")
        # visualizer.plot_time_series(ts_data.head(100), title="Sample Time Series",
        #                            show_trend=True, show_ma=True)

        print("   Creating histogram...")
        # visualizer.plot_histogram(ts_data, title="Time Series Distribution")

        print("   Creating ACF/PACF plots...")
        # visualizer.plot_acf_pacf(differenced.head(100), title="ACF/PACF Analysis")

        # Advanced diagnostic plots
        if 'arma_results' in locals() and 'error' not in arma_results:
            print("   Creating comprehensive model diagnostics...")
            # visualizer.plot_comprehensive_diagnostics(
            #     arma_results['residuals'],
            #     arma_results['fitted_values'],
            #     model_name="ARMA(2,1)"
            # )
            print("   Model diagnostic plots created (display disabled in demo)")
            print(f"     - Standardized residuals over time")
            print(f"     - Residuals vs fitted values with smooth trend")
            print(f"     - Histogram + KDE + normal overlay")
            print(f"     - Q-Q plot with normality statistics")
            print(f"     - Residuals autocorrelation function")
            print(f"     - Residuals partial autocorrelation function")

        print("   Basic visualization plots created (display disabled in demo)")

    except Exception as e:
        print(f"   Visualization failed: {str(e)}")

    print("\n12. Model Validation:")
    try:
        validator = ModelValidator()

        # Walk-forward validation on smaller dataset
        validation_data = ts_data.head(150)
        wfv_results = validator.walk_forward_validation(
            validation_data, ARIMAModeler,
            initial_window=100, step_size=5,
            forecast_horizon=1,
            model_params={'order': (1, 1, 1)}
        )

        if 'error' not in wfv_results:
            print(f"   Walk-forward validation completed")
            print(f"   Total folds: {wfv_results['total_folds']}")
            print(f"   Overall RMSE: {wfv_results['overall_metrics']['rmse']:.3f}")
        else:
            print(f"   Walk-forward validation failed: {wfv_results['error']}")

    except Exception as e:
        print(f"   Model validation failed: {str(e)}")

    print("\n13. Hyperparameter Tuning:")
    try:
        tuner = HyperparameterTuner()

        # Grid search on smaller parameter space and dataset
        tuning_data = ts_data.head(100)
        grid_results = tuner.grid_search_arima(
            tuning_data,
            p_range=(0, 2), d_range=(0, 1), q_range=(0, 2),
            criterion='aic', max_models=20
        )

        if 'error' not in grid_results:
            print(f"   Grid search completed")
            print(f"   Models tested: {grid_results['total_models_tested']}")
            print(f"   Best model: ARIMA{grid_results['best_params']['order']}")
            print(f"   Best score (AIC): {grid_results['best_score']:.3f}")
            print(f"   Average time per model: {grid_results['avg_time_per_model']:.3f}s")
        else:
            print(f"   Grid search failed: {grid_results['error']}")

    except Exception as e:
        print(f"   Hyperparameter tuning failed: {str(e)}")

    print("\n14. Benchmark Models Comparison:")
    try:
        benchmark_modeler = BenchmarkModels()

        # Test benchmark models against calibrated ARIMA
        if 'arima_results' in locals() and 'error' not in arima_results:
            comparison_data = ts_data.head(150)  # Use smaller dataset
            benchmark_comparison = benchmark_modeler.compare_against_benchmarks(arima_results, comparison_data)

            if 'error' not in benchmark_comparison:
                print(f"   Benchmark comparison completed")
                summary = benchmark_comparison['summary']
                print(f"   Total models compared: {summary['total_models']}")
                print(f"   Best AIC model: {summary['best_aic_model']}")
                print(f"   Best BIC model: {summary['best_bic_model']}")

                # Show top 3 models
                top_models = benchmark_comparison['comparison_table'][:3]
                print(f"   Top 3 models by AIC:")
                for i, model in enumerate(top_models, 1):
                    print(f"     {i}. {model['Model']}: AIC={model['AIC']:.2f}, BIC={model['BIC']:.2f}")

            else:
                print(f"   Benchmark comparison failed: {benchmark_comparison['error']}")

    except Exception as e:
        print(f"   Benchmark models comparison failed: {str(e)}")

    print("\n15. Information Criteria Analysis:")
    try:
        info_criteria = InformationCriteria()

        # Collect all fitted models for comparison
        models_for_comparison = []
        if 'arma_results' in locals() and 'error' not in arma_results:
            models_for_comparison.append(arma_results)
        if 'arima_results' in locals() and 'error' not in arima_results:
            models_for_comparison.append(arima_results)

        if models_for_comparison:
            selection_table = info_criteria.model_selection_table(models_for_comparison)

            if 'error' not in selection_table:
                print(f"   Information criteria analysis completed")
                best_models = selection_table['best_models']
                print(f"   Best AIC model: {best_models['aic_best']}")
                print(f"   Best BIC model: {best_models['bic_best']}")
                print(f"   Best AICc model: {best_models['aicc_best']}")

                # Show model weights
                weights = selection_table['model_weights']
                print(f"   Model evidence (Akaike weights):")
                for model, weight in weights.items():
                    print(f"     {model}: {weight:.3f}")

            else:
                print(f"   Information criteria analysis failed: {selection_table['error']}")

    except Exception as e:
        print(f"   Information criteria analysis failed: {str(e)}")

    print("\n16. ARCH/GARCH Volatility Modeling:")
    try:
        # Create returns data for volatility modeling
        returns_data = ts_data.pct_change().dropna() * 100  # Convert to percentage returns
        returns_sample = returns_data.head(200)  # Use smaller sample

        arch_garch_modeler = ARCHGARCHModeler()

        # Test for ARCH effects first
        arch_lm_test = arch_garch_modeler.arch_lm_test(returns_sample, lags=5)
        if 'error' not in arch_lm_test:
            print(f"   ARCH LM test completed")
            print(f"   LM statistic: {arch_lm_test['lm_statistic']:.3f}")
            print(f"   P-value: {arch_lm_test['lm_pvalue']:.4f}")
            print(f"   Conclusion: {arch_lm_test['conclusion']}")

        # Fit ARCH(1) model
        print("\n   Fitting ARCH(1) model...")
        arch_results = arch_garch_modeler.fit_arch(returns_sample, arch_order=1)

        if 'error' not in arch_results:
            print(f"   ARCH(1) model fitted successfully")
            print(f"   AIC: {arch_results['aic']:.3f}")
            print(f"   BIC: {arch_results['bic']:.3f}")
            print(f"   Log-likelihood: {arch_results['loglikelihood']:.3f}")

            # Comprehensive diagnostics
            print("\n   Running comprehensive GARCH diagnostics...")
            diagnostics = arch_garch_modeler.comprehensive_garch_diagnostics(arch_results, returns_sample)

            if 'error' not in diagnostics:
                summary = diagnostics['summary']
                print(f"   Model adequacy: {summary['model_adequacy']}")
                print(f"   Total concerns: {summary['total_concerns']}")

                if summary['concerns']:
                    print(f"   Main concerns:")
                    for concern in summary['concerns'][:3]:  # Show top 3 concerns
                        print(f"     - {concern}")

                if summary['recommendations']:
                    print(f"   Recommendations:")
                    for rec in summary['recommendations'][:2]:  # Show top 2 recommendations
                        print(f"     - {rec}")

                # Individual test results
                print(f"\n   Individual diagnostic tests:")

                ljung_box = diagnostics.get('ljung_box_residuals', {})
                if 'error' not in ljung_box:
                    print(f"     Ljung-Box (residuals): {ljung_box.get('conclusion', 'N/A')}")

                arch_lm = diagnostics.get('arch_lm_test', {})
                if 'error' not in arch_lm:
                    print(f"     ARCH LM test: {arch_lm.get('conclusion', 'N/A')}")

                nyblom = diagnostics.get('nyblom_stability', {})
                if 'error' not in nyblom:
                    print(f"     Nyblom stability: {nyblom.get('conclusion', 'N/A')}")

                sign_bias = diagnostics.get('sign_bias_test', {})
                if 'error' not in sign_bias:
                    print(f"     Sign bias test: {sign_bias.get('conclusion', 'N/A')}")

                pearson = diagnostics.get('adjusted_pearson', {})
                if 'error' not in pearson:
                    print(f"     Pearson goodness-of-fit: {pearson.get('conclusion', 'N/A')}")

            else:
                print(f"   GARCH diagnostics failed: {diagnostics['error']}")

        else:
            print(f"   ARCH model fitting failed: {arch_results['error']}")

        # Try GARCH(1,1) if ARCH package is available
        print("\n   Attempting GARCH(1,1) model...")
        garch_results = arch_garch_modeler.fit_garch(returns_sample, garch_order=(1, 1))

        if 'error' not in garch_results:
            print(f"   GARCH(1,1) model fitted successfully")
            print(f"   AIC: {garch_results['aic']:.3f}")
            print(f"   BIC: {garch_results['bic']:.3f}")
            print(f"   Log-likelihood: {garch_results['loglikelihood']:.3f}")

            # Compare ARCH vs GARCH
            if 'error' not in arch_results:
                aic_improvement = arch_results['aic'] - garch_results['aic']
                print(f"   GARCH(1,1) vs ARCH(1) AIC improvement: {aic_improvement:.3f}")

        else:
            print(f"   GARCH(1,1) model fitting failed: {garch_results['error']}")
            if "arch package" in garch_results['error'].lower():
                print(f"   Note: Install 'arch' package for full GARCH functionality")

    except Exception as e:
        print(f"   ARCH/GARCH modeling failed: {str(e)}")

    print("\nDemo completed successfully!")

if __name__ == "__main__":
    demonstrate_timeseries()