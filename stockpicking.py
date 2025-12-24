"""
Stock Picking: Comprehensive Stock Analysis Framework

This module provides extensive stock analysis functionality including:
- Fundamental analysis of balance sheets, income statements, and cash flow trends
- Technical analysis indicators (ADX, MACD, Parabolic SAR, Stochastic, RSI)
- Behavioral and sentiment analysis for cognitive biases and market psychology
- Integrated stock screening and ranking capabilities
- Portfolio optimization and risk management tools

Author: Claude AI
Created: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from enum import Enum
import yfinance as yf
import requests
from textblob import TextBlob
import nltk
from wordcloud import WordCloud
import re
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer


class AnalysisType(Enum):
    """Analysis type enumeration."""
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    SENTIMENT = "sentiment"


class TrendDirection(Enum):
    """Trend direction enumeration."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"


@dataclass
class AnalysisConfig:
    """Configuration class for stock analysis calculations."""

    # General settings
    precision: int = 6
    max_iterations: int = 1000
    tolerance: float = 1e-8

    # Data settings
    data_lookback_days: int = 252  # 1 year
    price_lookback_days: int = 63   # 3 months

    # Technical analysis settings
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    adx_period: int = 14
    sar_acceleration: float = 0.02
    sar_maximum: float = 0.2

    # Fundamental analysis settings
    growth_lookback_years: int = 5
    financial_health_weights: Dict[str, float] = None

    # Behavioral analysis settings
    sentiment_window: int = 30
    herding_threshold: float = 0.7
    overconfidence_threshold: float = 0.6

    # Plotting settings
    plot_style: str = 'seaborn-v0_8'
    figure_size: Tuple[int, int] = (15, 10)

    def __post_init__(self):
        if self.financial_health_weights is None:
            self.financial_health_weights = {
                'liquidity': 0.25,
                'profitability': 0.30,
                'leverage': 0.20,
                'efficiency': 0.15,
                'growth': 0.10
            }


class FundamentalAnalysis:
    """
    Fundamental analysis of financial statements and company metrics.
    """

    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()

    def analyze_balance_sheet(self, ticker: str, years: int = None) -> Dict[str, Any]:
        """
        Analyze balance sheet trends and financial health.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        years : int
            Number of years to analyze

        Returns:
        --------
        Dict with balance sheet analysis
        """
        try:
            if years is None:
                years = self.config.growth_lookback_years

            stock = yf.Ticker(ticker)
            balance_sheet = stock.balance_sheet

            if balance_sheet.empty:
                return {'error': f'No balance sheet data available for {ticker}'}

            # Get the most recent years of data
            recent_data = balance_sheet.iloc[:, :min(years, balance_sheet.shape[1])]

            # Key balance sheet metrics
            metrics = {}

            # Assets Analysis
            if 'Total Assets' in recent_data.index:
                total_assets = recent_data.loc['Total Assets'].dropna()
                if len(total_assets) > 1:
                    asset_growth = self._calculate_growth_rate(total_assets)
                    metrics['asset_growth_rate'] = round(asset_growth, self.config.precision)
                    metrics['total_assets_trend'] = self._analyze_trend(total_assets.values)

            # Liabilities Analysis
            if 'Total Liab' in recent_data.index:
                total_liabilities = recent_data.loc['Total Liab'].dropna()
                if len(total_liabilities) > 1:
                    liability_growth = self._calculate_growth_rate(total_liabilities)
                    metrics['liability_growth_rate'] = round(liability_growth, self.config.precision)
                    metrics['total_liabilities_trend'] = self._analyze_trend(total_liabilities.values)

            # Equity Analysis
            if 'Total Stockholder Equity' in recent_data.index:
                total_equity = recent_data.loc['Total Stockholder Equity'].dropna()
                if len(total_equity) > 1:
                    equity_growth = self._calculate_growth_rate(total_equity)
                    metrics['equity_growth_rate'] = round(equity_growth, self.config.precision)
                    metrics['total_equity_trend'] = self._analyze_trend(total_equity.values)

            # Financial Ratios
            liquidity_ratios = self._calculate_liquidity_ratios(recent_data)
            leverage_ratios = self._calculate_leverage_ratios(recent_data)

            # Asset composition analysis
            asset_composition = self._analyze_asset_composition(recent_data)

            return {
                'ticker': ticker,
                'analysis_date': datetime.now(),
                'years_analyzed': min(years, recent_data.shape[1]),
                'growth_metrics': metrics,
                'liquidity_ratios': liquidity_ratios,
                'leverage_ratios': leverage_ratios,
                'asset_composition': asset_composition,
                'balance_sheet_data': recent_data.to_dict(),
                'analysis_type': 'balance_sheet'
            }

        except Exception as e:
            return {'error': str(e)}

    def analyze_income_statement(self, ticker: str, years: int = None) -> Dict[str, Any]:
        """
        Analyze income statement trends and profitability metrics.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        years : int
            Number of years to analyze

        Returns:
        --------
        Dict with income statement analysis
        """
        try:
            if years is None:
                years = self.config.growth_lookback_years

            stock = yf.Ticker(ticker)
            income_stmt = stock.financials

            if income_stmt.empty:
                return {'error': f'No income statement data available for {ticker}'}

            # Get the most recent years of data
            recent_data = income_stmt.iloc[:, :min(years, income_stmt.shape[1])]

            # Key income statement metrics
            metrics = {}

            # Revenue Analysis
            if 'Total Revenue' in recent_data.index:
                revenue = recent_data.loc['Total Revenue'].dropna()
                if len(revenue) > 1:
                    revenue_growth = self._calculate_growth_rate(revenue)
                    metrics['revenue_growth_rate'] = round(revenue_growth, self.config.precision)
                    metrics['revenue_trend'] = self._analyze_trend(revenue.values)
                    metrics['revenue_volatility'] = round(np.std(revenue.pct_change().dropna()), self.config.precision)

            # Profitability Analysis
            profitability_metrics = self._analyze_profitability(recent_data)

            # Operating Performance
            operating_metrics = self._analyze_operating_performance(recent_data)

            # Earnings Quality
            earnings_quality = self._analyze_earnings_quality(recent_data)

            # Margin Analysis
            margin_analysis = self._calculate_margin_trends(recent_data)

            return {
                'ticker': ticker,
                'analysis_date': datetime.now(),
                'years_analyzed': min(years, recent_data.shape[1]),
                'revenue_metrics': metrics,
                'profitability_metrics': profitability_metrics,
                'operating_metrics': operating_metrics,
                'earnings_quality': earnings_quality,
                'margin_analysis': margin_analysis,
                'income_statement_data': recent_data.to_dict(),
                'analysis_type': 'income_statement'
            }

        except Exception as e:
            return {'error': str(e)}

    def analyze_cash_flow(self, ticker: str, years: int = None) -> Dict[str, Any]:
        """
        Analyze cash flow statement trends and cash generation capability.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        years : int
            Number of years to analyze

        Returns:
        --------
        Dict with cash flow analysis
        """
        try:
            if years is None:
                years = self.config.growth_lookback_years

            stock = yf.Ticker(ticker)
            cash_flow = stock.cashflow

            if cash_flow.empty:
                return {'error': f'No cash flow data available for {ticker}'}

            # Get the most recent years of data
            recent_data = cash_flow.iloc[:, :min(years, cash_flow.shape[1])]

            # Operating Cash Flow Analysis
            operating_cf_analysis = self._analyze_operating_cash_flow(recent_data)

            # Investment Cash Flow Analysis
            investing_cf_analysis = self._analyze_investing_cash_flow(recent_data)

            # Financing Cash Flow Analysis
            financing_cf_analysis = self._analyze_financing_cash_flow(recent_data)

            # Cash Flow Quality Metrics
            cf_quality = self._analyze_cash_flow_quality(recent_data)

            # Free Cash Flow Analysis
            free_cf_analysis = self._analyze_free_cash_flow(recent_data)

            return {
                'ticker': ticker,
                'analysis_date': datetime.now(),
                'years_analyzed': min(years, recent_data.shape[1]),
                'operating_cash_flow': operating_cf_analysis,
                'investing_cash_flow': investing_cf_analysis,
                'financing_cash_flow': financing_cf_analysis,
                'cash_flow_quality': cf_quality,
                'free_cash_flow': free_cf_analysis,
                'cash_flow_data': recent_data.to_dict(),
                'analysis_type': 'cash_flow'
            }

        except Exception as e:
            return {'error': str(e)}

    def comprehensive_fundamental_analysis(self, ticker: str, years: int = None) -> Dict[str, Any]:
        """
        Perform comprehensive fundamental analysis combining all financial statements.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        years : int
            Number of years to analyze

        Returns:
        --------
        Dict with comprehensive fundamental analysis
        """
        try:
            # Perform individual analyses
            balance_sheet_analysis = self.analyze_balance_sheet(ticker, years)
            income_analysis = self.analyze_income_statement(ticker, years)
            cash_flow_analysis = self.analyze_cash_flow(ticker, years)

            # Check for errors
            if any('error' in analysis for analysis in [balance_sheet_analysis, income_analysis, cash_flow_analysis]):
                errors = [analysis.get('error') for analysis in [balance_sheet_analysis, income_analysis, cash_flow_analysis] if 'error' in analysis]
                return {'error': f'Analysis errors: {"; ".join(errors)}'}

            # Calculate composite scores
            financial_health_score = self._calculate_financial_health_score(
                balance_sheet_analysis, income_analysis, cash_flow_analysis
            )

            # Growth consistency analysis
            growth_consistency = self._analyze_growth_consistency(
                balance_sheet_analysis, income_analysis, cash_flow_analysis
            )

            # Risk assessment
            risk_assessment = self._assess_financial_risk(
                balance_sheet_analysis, income_analysis, cash_flow_analysis
            )

            # Investment recommendation
            investment_recommendation = self._generate_investment_recommendation(
                financial_health_score, growth_consistency, risk_assessment
            )

            return {
                'ticker': ticker,
                'analysis_date': datetime.now(),
                'years_analyzed': years or self.config.growth_lookback_years,
                'balance_sheet_analysis': balance_sheet_analysis,
                'income_statement_analysis': income_analysis,
                'cash_flow_analysis': cash_flow_analysis,
                'financial_health_score': financial_health_score,
                'growth_consistency': growth_consistency,
                'risk_assessment': risk_assessment,
                'investment_recommendation': investment_recommendation,
                'analysis_type': 'comprehensive_fundamental'
            }

        except Exception as e:
            return {'error': str(e)}

    def _calculate_growth_rate(self, data_series: pd.Series) -> float:
        """Calculate compound annual growth rate."""
        if len(data_series) < 2:
            return 0.0

        # Sort by date (most recent first in yfinance data)
        sorted_data = data_series.sort_index()
        start_value = sorted_data.iloc[0]
        end_value = sorted_data.iloc[-1]
        periods = len(sorted_data) - 1

        if start_value <= 0 or end_value <= 0:
            return 0.0

        cagr = (end_value / start_value) ** (1 / periods) - 1
        return cagr

    def _analyze_trend(self, values: np.ndarray) -> str:
        """Analyze trend direction using linear regression."""
        if len(values) < 2:
            return TrendDirection.NEUTRAL.value

        x = np.arange(len(values))
        slope, _, r_value, _, _ = stats.linregress(x, values)

        # Determine trend based on slope and R-squared
        r_squared = r_value ** 2

        if r_squared < 0.5:  # Low correlation, volatile
            return TrendDirection.VOLATILE.value
        elif slope > 0:
            return TrendDirection.BULLISH.value
        elif slope < 0:
            return TrendDirection.BEARISH.value
        else:
            return TrendDirection.NEUTRAL.value

    def _calculate_liquidity_ratios(self, balance_sheet: pd.DataFrame) -> Dict[str, Any]:
        """Calculate liquidity ratios."""
        ratios = {}

        try:
            # Current Ratio
            if 'Total Current Assets' in balance_sheet.index and 'Total Current Liabilities' in balance_sheet.index:
                current_assets = balance_sheet.loc['Total Current Assets'].dropna()
                current_liabilities = balance_sheet.loc['Total Current Liabilities'].dropna()

                if len(current_assets) > 0 and len(current_liabilities) > 0:
                    current_ratio = current_assets / current_liabilities
                    ratios['current_ratio'] = {
                        'latest': round(current_ratio.iloc[0], self.config.precision),
                        'trend': self._analyze_trend(current_ratio.values),
                        'average': round(current_ratio.mean(), self.config.precision)
                    }

            # Quick Ratio (if data available)
            if 'Cash And Cash Equivalents' in balance_sheet.index and 'Total Current Liabilities' in balance_sheet.index:
                cash = balance_sheet.loc['Cash And Cash Equivalents'].dropna()
                current_liabilities = balance_sheet.loc['Total Current Liabilities'].dropna()

                if len(cash) > 0 and len(current_liabilities) > 0:
                    cash_ratio = cash / current_liabilities
                    ratios['cash_ratio'] = {
                        'latest': round(cash_ratio.iloc[0], self.config.precision),
                        'trend': self._analyze_trend(cash_ratio.values),
                        'average': round(cash_ratio.mean(), self.config.precision)
                    }

        except Exception as e:
            ratios['error'] = str(e)

        return ratios

    def _calculate_leverage_ratios(self, balance_sheet: pd.DataFrame) -> Dict[str, Any]:
        """Calculate leverage ratios."""
        ratios = {}

        try:
            # Debt-to-Equity Ratio
            if 'Total Liab' in balance_sheet.index and 'Total Stockholder Equity' in balance_sheet.index:
                total_debt = balance_sheet.loc['Total Liab'].dropna()
                total_equity = balance_sheet.loc['Total Stockholder Equity'].dropna()

                if len(total_debt) > 0 and len(total_equity) > 0:
                    # Filter out zero or negative equity values
                    valid_data = (total_equity > 0) & (total_debt >= 0)
                    if valid_data.any():
                        debt_to_equity = total_debt[valid_data] / total_equity[valid_data]
                        ratios['debt_to_equity'] = {
                            'latest': round(debt_to_equity.iloc[0], self.config.precision),
                            'trend': self._analyze_trend(debt_to_equity.values),
                            'average': round(debt_to_equity.mean(), self.config.precision)
                        }

            # Debt-to-Assets Ratio
            if 'Total Liab' in balance_sheet.index and 'Total Assets' in balance_sheet.index:
                total_debt = balance_sheet.loc['Total Liab'].dropna()
                total_assets = balance_sheet.loc['Total Assets'].dropna()

                if len(total_debt) > 0 and len(total_assets) > 0:
                    # Filter out zero or negative asset values
                    valid_data = (total_assets > 0) & (total_debt >= 0)
                    if valid_data.any():
                        debt_to_assets = total_debt[valid_data] / total_assets[valid_data]
                        ratios['debt_to_assets'] = {
                            'latest': round(debt_to_assets.iloc[0], self.config.precision),
                            'trend': self._analyze_trend(debt_to_assets.values),
                            'average': round(debt_to_assets.mean(), self.config.precision)
                        }

        except Exception as e:
            ratios['error'] = str(e)

        return ratios

    def _analyze_asset_composition(self, balance_sheet: pd.DataFrame) -> Dict[str, Any]:
        """Analyze asset composition and quality."""
        composition = {}

        try:
            if 'Total Assets' in balance_sheet.index:
                total_assets = balance_sheet.loc['Total Assets'].dropna()
                latest_total = total_assets.iloc[0] if len(total_assets) > 0 else 0

                if latest_total > 0:
                    # Current vs Non-current assets
                    if 'Total Current Assets' in balance_sheet.index:
                        current_assets = balance_sheet.loc['Total Current Assets'].iloc[0]
                        composition['current_assets_percentage'] = round((current_assets / latest_total) * 100, self.config.precision)

                    # Cash percentage
                    if 'Cash And Cash Equivalents' in balance_sheet.index:
                        cash = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0]
                        composition['cash_percentage'] = round((cash / latest_total) * 100, self.config.precision)

                    # Property, Plant & Equipment
                    if 'Property Plant Equipment' in balance_sheet.index:
                        ppe = balance_sheet.loc['Property Plant Equipment'].iloc[0]
                        composition['ppe_percentage'] = round((ppe / latest_total) * 100, self.config.precision)

        except Exception as e:
            composition['error'] = str(e)

        return composition

    def _analyze_profitability(self, income_stmt: pd.DataFrame) -> Dict[str, Any]:
        """Analyze profitability metrics."""
        profitability = {}

        try:
            # Net Income Analysis
            if 'Net Income' in income_stmt.index:
                net_income = income_stmt.loc['Net Income'].dropna()
                if len(net_income) > 1:
                    ni_growth = self._calculate_growth_rate(net_income)
                    profitability['net_income_growth'] = round(ni_growth, self.config.precision)
                    profitability['net_income_trend'] = self._analyze_trend(net_income.values)
                    profitability['net_income_volatility'] = round(np.std(net_income.pct_change().dropna()), self.config.precision)

            # Operating Income Analysis
            if 'Operating Income' in income_stmt.index:
                operating_income = income_stmt.loc['Operating Income'].dropna()
                if len(operating_income) > 1:
                    oi_growth = self._calculate_growth_rate(operating_income)
                    profitability['operating_income_growth'] = round(oi_growth, self.config.precision)
                    profitability['operating_income_trend'] = self._analyze_trend(operating_income.values)

            # Gross Profit Analysis
            if 'Gross Profit' in income_stmt.index:
                gross_profit = income_stmt.loc['Gross Profit'].dropna()
                if len(gross_profit) > 1:
                    gp_growth = self._calculate_growth_rate(gross_profit)
                    profitability['gross_profit_growth'] = round(gp_growth, self.config.precision)
                    profitability['gross_profit_trend'] = self._analyze_trend(gross_profit.values)

        except Exception as e:
            profitability['error'] = str(e)

        return profitability

    def _analyze_operating_performance(self, income_stmt: pd.DataFrame) -> Dict[str, Any]:
        """Analyze operating performance metrics."""
        operating = {}

        try:
            # Operating Leverage
            if 'Total Revenue' in income_stmt.index and 'Operating Income' in income_stmt.index:
                revenue = income_stmt.loc['Total Revenue'].dropna()
                operating_income = income_stmt.loc['Operating Income'].dropna()

                if len(revenue) > 1 and len(operating_income) > 1:
                    revenue_change = revenue.pct_change().dropna()
                    oi_change = operating_income.pct_change().dropna()

                    # Calculate operating leverage where both changes are non-zero
                    valid_periods = (revenue_change != 0) & (oi_change != 0) & ~np.isinf(oi_change / revenue_change)
                    if valid_periods.any():
                        operating_leverage = oi_change[valid_periods] / revenue_change[valid_periods]
                        operating['average_operating_leverage'] = round(operating_leverage.mean(), self.config.precision)
                        operating['operating_leverage_volatility'] = round(operating_leverage.std(), self.config.precision)

            # Cost Structure Analysis
            cost_structure = self._analyze_cost_structure(income_stmt)
            operating['cost_structure'] = cost_structure

        except Exception as e:
            operating['error'] = str(e)

        return operating

    def _analyze_earnings_quality(self, income_stmt: pd.DataFrame) -> Dict[str, Any]:
        """Analyze earnings quality indicators."""
        quality = {}

        try:
            # Earnings Consistency
            if 'Net Income' in income_stmt.index:
                net_income = income_stmt.loc['Net Income'].dropna()
                if len(net_income) > 1:
                    # Calculate coefficient of variation
                    if net_income.mean() != 0:
                        cv = net_income.std() / abs(net_income.mean())
                        quality['earnings_consistency'] = round(1 / (1 + cv), self.config.precision)  # Higher is better

                    # Positive earnings ratio
                    positive_earnings = (net_income > 0).sum()
                    quality['positive_earnings_ratio'] = round(positive_earnings / len(net_income), self.config.precision)

            # Revenue vs Earnings Growth Comparison
            if 'Total Revenue' in income_stmt.index and 'Net Income' in income_stmt.index:
                revenue = income_stmt.loc['Total Revenue'].dropna()
                net_income = income_stmt.loc['Net Income'].dropna()

                if len(revenue) > 1 and len(net_income) > 1:
                    revenue_growth = self._calculate_growth_rate(revenue)
                    earnings_growth = self._calculate_growth_rate(net_income)

                    quality['revenue_growth'] = round(revenue_growth, self.config.precision)
                    quality['earnings_growth'] = round(earnings_growth, self.config.precision)

                    # Quality indicator: earnings should grow with revenue
                    if revenue_growth > 0:
                        quality['earnings_revenue_ratio'] = round(earnings_growth / revenue_growth, self.config.precision)

        except Exception as e:
            quality['error'] = str(e)

        return quality

    def _calculate_margin_trends(self, income_stmt: pd.DataFrame) -> Dict[str, Any]:
        """Calculate and analyze margin trends."""
        margins = {}

        try:
            if 'Total Revenue' in income_stmt.index:
                revenue = income_stmt.loc['Total Revenue'].dropna()

                # Gross Margin
                if 'Gross Profit' in income_stmt.index:
                    gross_profit = income_stmt.loc['Gross Profit'].dropna()
                    if len(gross_profit) > 0 and len(revenue) > 0:
                        # Align data
                        common_dates = gross_profit.index.intersection(revenue.index)
                        if len(common_dates) > 0:
                            gross_margin = (gross_profit[common_dates] / revenue[common_dates]) * 100
                            margins['gross_margin'] = {
                                'latest': round(gross_margin.iloc[0], self.config.precision),
                                'trend': self._analyze_trend(gross_margin.values),
                                'average': round(gross_margin.mean(), self.config.precision),
                                'volatility': round(gross_margin.std(), self.config.precision)
                            }

                # Operating Margin
                if 'Operating Income' in income_stmt.index:
                    operating_income = income_stmt.loc['Operating Income'].dropna()
                    if len(operating_income) > 0 and len(revenue) > 0:
                        common_dates = operating_income.index.intersection(revenue.index)
                        if len(common_dates) > 0:
                            operating_margin = (operating_income[common_dates] / revenue[common_dates]) * 100
                            margins['operating_margin'] = {
                                'latest': round(operating_margin.iloc[0], self.config.precision),
                                'trend': self._analyze_trend(operating_margin.values),
                                'average': round(operating_margin.mean(), self.config.precision),
                                'volatility': round(operating_margin.std(), self.config.precision)
                            }

                # Net Margin
                if 'Net Income' in income_stmt.index:
                    net_income = income_stmt.loc['Net Income'].dropna()
                    if len(net_income) > 0 and len(revenue) > 0:
                        common_dates = net_income.index.intersection(revenue.index)
                        if len(common_dates) > 0:
                            net_margin = (net_income[common_dates] / revenue[common_dates]) * 100
                            margins['net_margin'] = {
                                'latest': round(net_margin.iloc[0], self.config.precision),
                                'trend': self._analyze_trend(net_margin.values),
                                'average': round(net_margin.mean(), self.config.precision),
                                'volatility': round(net_margin.std(), self.config.precision)
                            }

        except Exception as e:
            margins['error'] = str(e)

        return margins

    def _analyze_cost_structure(self, income_stmt: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cost structure and efficiency."""
        cost_structure = {}

        try:
            if 'Total Revenue' in income_stmt.index:
                revenue = income_stmt.loc['Total Revenue'].dropna()

                # Cost of Revenue Analysis
                if 'Cost Of Revenue' in income_stmt.index:
                    cogs = income_stmt.loc['Cost Of Revenue'].dropna()
                    if len(cogs) > 0 and len(revenue) > 0:
                        common_dates = cogs.index.intersection(revenue.index)
                        if len(common_dates) > 0:
                            cogs_ratio = (cogs[common_dates] / revenue[common_dates]) * 100
                            cost_structure['cogs_percentage'] = {
                                'latest': round(cogs_ratio.iloc[0], self.config.precision),
                                'trend': self._analyze_trend(cogs_ratio.values),
                                'average': round(cogs_ratio.mean(), self.config.precision)
                            }

                # SG&A Analysis (if available)
                possible_sga_fields = ['Selling General Administrative', 'Total Operating Expenses']
                for field in possible_sga_fields:
                    if field in income_stmt.index:
                        sga = income_stmt.loc[field].dropna()
                        if len(sga) > 0 and len(revenue) > 0:
                            common_dates = sga.index.intersection(revenue.index)
                            if len(common_dates) > 0:
                                sga_ratio = (sga[common_dates] / revenue[common_dates]) * 100
                                cost_structure['sga_percentage'] = {
                                    'latest': round(sga_ratio.iloc[0], self.config.precision),
                                    'trend': self._analyze_trend(sga_ratio.values),
                                    'average': round(sga_ratio.mean(), self.config.precision)
                                }
                        break

        except Exception as e:
            cost_structure['error'] = str(e)

        return cost_structure

    def _analyze_operating_cash_flow(self, cash_flow: pd.DataFrame) -> Dict[str, Any]:
        """Analyze operating cash flow metrics."""
        operating_cf = {}

        try:
            if 'Total Cash From Operating Activities' in cash_flow.index:
                ocf = cash_flow.loc['Total Cash From Operating Activities'].dropna()
                if len(ocf) > 1:
                    ocf_growth = self._calculate_growth_rate(ocf)
                    operating_cf['growth_rate'] = round(ocf_growth, self.config.precision)
                    operating_cf['trend'] = self._analyze_trend(ocf.values)
                    operating_cf['average'] = round(ocf.mean(), self.config.precision)
                    operating_cf['volatility'] = round(ocf.std(), self.config.precision)

                    # Cash flow consistency
                    positive_periods = (ocf > 0).sum()
                    operating_cf['positive_periods_ratio'] = round(positive_periods / len(ocf), self.config.precision)

        except Exception as e:
            operating_cf['error'] = str(e)

        return operating_cf

    def _analyze_investing_cash_flow(self, cash_flow: pd.DataFrame) -> Dict[str, Any]:
        """Analyze investing cash flow metrics."""
        investing_cf = {}

        try:
            if 'Total Cash From Investing Activities' in cash_flow.index:
                icf = cash_flow.loc['Total Cash From Investing Activities'].dropna()
                if len(icf) > 1:
                    investing_cf['average'] = round(icf.mean(), self.config.precision)
                    investing_cf['trend'] = self._analyze_trend(icf.values)
                    investing_cf['volatility'] = round(icf.std(), self.config.precision)

                    # Investment intensity (typically negative for growing companies)
                    negative_periods = (icf < 0).sum()
                    investing_cf['investment_periods_ratio'] = round(negative_periods / len(icf), self.config.precision)

        except Exception as e:
            investing_cf['error'] = str(e)

        return investing_cf

    def _analyze_financing_cash_flow(self, cash_flow: pd.DataFrame) -> Dict[str, Any]:
        """Analyze financing cash flow metrics."""
        financing_cf = {}

        try:
            if 'Total Cash From Financing Activities' in cash_flow.index:
                fcf = cash_flow.loc['Total Cash From Financing Activities'].dropna()
                if len(fcf) > 1:
                    financing_cf['average'] = round(fcf.mean(), self.config.precision)
                    financing_cf['trend'] = self._analyze_trend(fcf.values)
                    financing_cf['volatility'] = round(fcf.std(), self.config.precision)

        except Exception as e:
            financing_cf['error'] = str(e)

        return financing_cf

    def _analyze_cash_flow_quality(self, cash_flow: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cash flow quality metrics."""
        cf_quality = {}

        try:
            # Operating Cash Flow to Net Income Ratio (if we have net income)
            if 'Total Cash From Operating Activities' in cash_flow.index:
                ocf = cash_flow.loc['Total Cash From Operating Activities'].dropna()

                # Cash flow predictability
                if len(ocf) > 2:
                    # Calculate coefficient of variation
                    if ocf.mean() != 0:
                        cv = ocf.std() / abs(ocf.mean())
                        cf_quality['predictability'] = round(1 / (1 + cv), self.config.precision)

                # Cash conversion quality
                cf_quality['operating_cash_flow_quality'] = {
                    'average': round(ocf.mean(), self.config.precision),
                    'consistency': round(1 - (ocf.std() / abs(ocf.mean())) if ocf.mean() != 0 else 0, self.config.precision)
                }

        except Exception as e:
            cf_quality['error'] = str(e)

        return cf_quality

    def _analyze_free_cash_flow(self, cash_flow: pd.DataFrame) -> Dict[str, Any]:
        """Analyze free cash flow metrics."""
        free_cf = {}

        try:
            ocf_key = 'Total Cash From Operating Activities'
            capex_key = 'Capital Expenditures'

            if ocf_key in cash_flow.index:
                ocf = cash_flow.loc[ocf_key].dropna()

                # Try to find capital expenditures
                capex = None
                for possible_capex in ['Capital Expenditures', 'Capital Expenditure', 'Capex']:
                    if possible_capex in cash_flow.index:
                        capex = cash_flow.loc[possible_capex].dropna()
                        break

                if capex is not None and len(capex) > 0:
                    # Align data
                    common_dates = ocf.index.intersection(capex.index)
                    if len(common_dates) > 0:
                        # Free Cash Flow = Operating Cash Flow - Capital Expenditures
                        # Note: CapEx is typically negative, so we add it
                        fcf = ocf[common_dates] + capex[common_dates]  # Since capex is negative

                        if len(fcf) > 1:
                            fcf_growth = self._calculate_growth_rate(fcf)
                            free_cf['growth_rate'] = round(fcf_growth, self.config.precision)
                            free_cf['trend'] = self._analyze_trend(fcf.values)
                            free_cf['average'] = round(fcf.mean(), self.config.precision)
                            free_cf['volatility'] = round(fcf.std(), self.config.precision)

                            # Free cash flow yield
                            positive_fcf = (fcf > 0).sum()
                            free_cf['positive_periods_ratio'] = round(positive_fcf / len(fcf), self.config.precision)
                else:
                    free_cf['note'] = 'Capital expenditure data not available for free cash flow calculation'

        except Exception as e:
            free_cf['error'] = str(e)

        return free_cf

    def _calculate_financial_health_score(self, balance_sheet: Dict, income_stmt: Dict, cash_flow: Dict) -> Dict[str, Any]:
        """Calculate overall financial health score."""
        try:
            scores = {}
            weights = self.config.financial_health_weights

            # Liquidity Score (25%)
            liquidity_score = 0
            if 'liquidity_ratios' in balance_sheet and 'current_ratio' in balance_sheet['liquidity_ratios']:
                current_ratio = balance_sheet['liquidity_ratios']['current_ratio']['latest']
                # Score based on current ratio (1.5-2.5 is good)
                if current_ratio >= 1.5:
                    liquidity_score = min(100, (current_ratio / 2.5) * 100)
                else:
                    liquidity_score = (current_ratio / 1.5) * 60  # Below 1.5 is concerning
            scores['liquidity_score'] = round(liquidity_score, self.config.precision)

            # Profitability Score (30%)
            profitability_score = 0
            if 'margin_analysis' in income_stmt and 'net_margin' in income_stmt['margin_analysis']:
                net_margin = income_stmt['margin_analysis']['net_margin']['latest']
                # Score based on net margin (>10% is excellent)
                if net_margin > 0:
                    profitability_score = min(100, (net_margin / 10) * 100)
                else:
                    profitability_score = 0
            scores['profitability_score'] = round(profitability_score, self.config.precision)

            # Leverage Score (20%)
            leverage_score = 100  # Start with perfect score
            if 'leverage_ratios' in balance_sheet and 'debt_to_equity' in balance_sheet['leverage_ratios']:
                debt_to_equity = balance_sheet['leverage_ratios']['debt_to_equity']['latest']
                # Lower debt is better (D/E < 0.5 is excellent, > 2.0 is concerning)
                if debt_to_equity <= 0.5:
                    leverage_score = 100
                elif debt_to_equity <= 1.0:
                    leverage_score = 80
                elif debt_to_equity <= 2.0:
                    leverage_score = 60
                else:
                    leverage_score = 40
            scores['leverage_score'] = round(leverage_score, self.config.precision)

            # Efficiency Score (15%)
            efficiency_score = 50  # Default neutral score
            if ('margin_analysis' in income_stmt and
                'operating_margin' in income_stmt['margin_analysis'] and
                income_stmt['margin_analysis']['operating_margin']['trend'] == TrendDirection.BULLISH.value):
                efficiency_score = 75
            scores['efficiency_score'] = round(efficiency_score, self.config.precision)

            # Growth Score (10%)
            growth_score = 50  # Default neutral score
            if 'revenue_metrics' in income_stmt and 'revenue_growth_rate' in income_stmt['revenue_metrics']:
                revenue_growth = income_stmt['revenue_metrics']['revenue_growth_rate']
                if revenue_growth > 0.1:  # 10% growth is good
                    growth_score = min(100, (revenue_growth / 0.2) * 100)  # 20% growth = 100 points
                elif revenue_growth > 0:
                    growth_score = (revenue_growth / 0.1) * 70  # Scale to 70 points for 10% growth
                else:
                    growth_score = 30  # Negative growth
            scores['growth_score'] = round(growth_score, self.config.precision)

            # Calculate weighted overall score
            overall_score = (
                scores['liquidity_score'] * weights['liquidity'] +
                scores['profitability_score'] * weights['profitability'] +
                scores['leverage_score'] * weights['leverage'] +
                scores['efficiency_score'] * weights['efficiency'] +
                scores['growth_score'] * weights['growth']
            )

            scores['overall_score'] = round(overall_score, self.config.precision)
            scores['rating'] = self._get_rating_from_score(overall_score)

            return scores

        except Exception as e:
            return {'error': str(e)}

    def _analyze_growth_consistency(self, balance_sheet: Dict, income_stmt: Dict, cash_flow: Dict) -> Dict[str, Any]:
        """Analyze growth consistency across different metrics."""
        try:
            consistency = {}

            # Collect growth rates
            growth_rates = {}

            if 'revenue_metrics' in income_stmt and 'revenue_growth_rate' in income_stmt['revenue_metrics']:
                growth_rates['revenue'] = income_stmt['revenue_metrics']['revenue_growth_rate']

            if 'growth_metrics' in balance_sheet and 'asset_growth_rate' in balance_sheet['growth_metrics']:
                growth_rates['assets'] = balance_sheet['growth_metrics']['asset_growth_rate']

            if 'operating_cash_flow' in cash_flow and 'growth_rate' in cash_flow['operating_cash_flow']:
                growth_rates['operating_cash_flow'] = cash_flow['operating_cash_flow']['growth_rate']

            if len(growth_rates) >= 2:
                growth_values = list(growth_rates.values())
                consistency['growth_rates'] = growth_rates
                consistency['growth_variance'] = round(np.var(growth_values), self.config.precision)
                consistency['growth_consistency_score'] = round(max(0, 100 - (np.var(growth_values) * 1000)), self.config.precision)
            else:
                consistency['note'] = 'Insufficient data for growth consistency analysis'

            return consistency

        except Exception as e:
            return {'error': str(e)}

    def _assess_financial_risk(self, balance_sheet: Dict, income_stmt: Dict, cash_flow: Dict) -> Dict[str, Any]:
        """Assess financial risk factors."""
        try:
            risk_factors = {}
            risk_score = 0  # Lower is better

            # Liquidity Risk
            if ('liquidity_ratios' in balance_sheet and
                'current_ratio' in balance_sheet['liquidity_ratios']):
                current_ratio = balance_sheet['liquidity_ratios']['current_ratio']['latest']
                if current_ratio < 1.0:
                    risk_score += 30  # High liquidity risk
                    risk_factors['liquidity_risk'] = 'High'
                elif current_ratio < 1.5:
                    risk_score += 15  # Moderate liquidity risk
                    risk_factors['liquidity_risk'] = 'Moderate'
                else:
                    risk_factors['liquidity_risk'] = 'Low'

            # Leverage Risk
            if ('leverage_ratios' in balance_sheet and
                'debt_to_equity' in balance_sheet['leverage_ratios']):
                debt_to_equity = balance_sheet['leverage_ratios']['debt_to_equity']['latest']
                if debt_to_equity > 2.0:
                    risk_score += 25  # High leverage risk
                    risk_factors['leverage_risk'] = 'High'
                elif debt_to_equity > 1.0:
                    risk_score += 10  # Moderate leverage risk
                    risk_factors['leverage_risk'] = 'Moderate'
                else:
                    risk_factors['leverage_risk'] = 'Low'

            # Profitability Risk
            if ('margin_analysis' in income_stmt and
                'net_margin' in income_stmt['margin_analysis']):
                net_margin = income_stmt['margin_analysis']['net_margin']['latest']
                if net_margin < 0:
                    risk_score += 20  # Negative margins
                    risk_factors['profitability_risk'] = 'High'
                elif net_margin < 5:
                    risk_score += 10  # Low margins
                    risk_factors['profitability_risk'] = 'Moderate'
                else:
                    risk_factors['profitability_risk'] = 'Low'

            # Cash Flow Risk
            if ('operating_cash_flow' in cash_flow and
                'positive_periods_ratio' in cash_flow['operating_cash_flow']):
                positive_ratio = cash_flow['operating_cash_flow']['positive_periods_ratio']
                if positive_ratio < 0.6:
                    risk_score += 15  # Inconsistent cash flow
                    risk_factors['cash_flow_risk'] = 'High'
                elif positive_ratio < 0.8:
                    risk_score += 8  # Somewhat inconsistent
                    risk_factors['cash_flow_risk'] = 'Moderate'
                else:
                    risk_factors['cash_flow_risk'] = 'Low'

            # Overall Risk Assessment
            if risk_score <= 20:
                overall_risk = 'Low'
            elif risk_score <= 50:
                overall_risk = 'Moderate'
            else:
                overall_risk = 'High'

            return {
                'risk_factors': risk_factors,
                'risk_score': round(risk_score, self.config.precision),
                'overall_risk': overall_risk
            }

        except Exception as e:
            return {'error': str(e)}

    def _generate_investment_recommendation(self, health_score: Dict, growth_consistency: Dict, risk_assessment: Dict) -> Dict[str, Any]:
        """Generate investment recommendation based on analysis."""
        try:
            recommendation = {}

            if 'error' in health_score or 'error' in risk_assessment:
                return {'recommendation': 'HOLD', 'reason': 'Insufficient data for recommendation'}

            overall_score = health_score.get('overall_score', 50)
            risk_level = risk_assessment.get('overall_risk', 'Moderate')

            # Decision matrix
            if overall_score >= 80 and risk_level == 'Low':
                recommendation['action'] = 'STRONG BUY'
                recommendation['confidence'] = 'High'
            elif overall_score >= 70 and risk_level in ['Low', 'Moderate']:
                recommendation['action'] = 'BUY'
                recommendation['confidence'] = 'High' if risk_level == 'Low' else 'Moderate'
            elif overall_score >= 60 and risk_level == 'Low':
                recommendation['action'] = 'BUY'
                recommendation['confidence'] = 'Moderate'
            elif overall_score >= 50 and risk_level == 'Moderate':
                recommendation['action'] = 'HOLD'
                recommendation['confidence'] = 'Moderate'
            elif overall_score >= 40:
                recommendation['action'] = 'HOLD'
                recommendation['confidence'] = 'Low'
            else:
                recommendation['action'] = 'SELL'
                recommendation['confidence'] = 'High' if risk_level == 'High' else 'Moderate'

            # Add reasoning
            reasons = []
            if overall_score >= 70:
                reasons.append('Strong financial health')
            if risk_level == 'Low':
                reasons.append('Low risk profile')
            if 'growth_consistency_score' in growth_consistency and growth_consistency['growth_consistency_score'] > 70:
                reasons.append('Consistent growth pattern')

            recommendation['reasons'] = reasons if reasons else ['Mixed financial indicators']
            recommendation['health_score'] = overall_score
            recommendation['risk_level'] = risk_level

            return recommendation

        except Exception as e:
            return {'error': str(e)}

    def _get_rating_from_score(self, score: float) -> str:
        """Convert numerical score to letter rating."""
        if score >= 90:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 70:
            return 'B+'
        elif score >= 60:
            return 'B'
        elif score >= 50:
            return 'C+'
        elif score >= 40:
            return 'C'
        elif score >= 30:
            return 'D'
        else:
            return 'F'


class TechnicalAnalysis:
    """
    Technical analysis indicators and chart pattern recognition.
    """

    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()

    def get_price_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Get price data for technical analysis.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        period : str
            Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

        Returns:
        --------
        DataFrame with OHLCV data
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            return data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

    def calculate_rsi(self, data: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Parameters:
        -----------
        data : DataFrame
            Price data with 'Close' column
        period : int
            RSI period (default from config)

        Returns:
        --------
        Series with RSI values
        """
        if period is None:
            period = self.config.rsi_period

        try:
            close = data['Close']
            delta = close.diff()

            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return rsi
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            return pd.Series()

    def calculate_macd(self, data: pd.DataFrame, fast: int = None, slow: int = None, signal: int = None) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Parameters:
        -----------
        data : DataFrame
            Price data with 'Close' column
        fast : int
            Fast EMA period
        slow : int
            Slow EMA period
        signal : int
            Signal line EMA period

        Returns:
        --------
        Dict with MACD line, signal line, and histogram
        """
        if fast is None:
            fast = self.config.macd_fast
        if slow is None:
            slow = self.config.macd_slow
        if signal is None:
            signal = self.config.macd_signal

        try:
            close = data['Close']

            ema_fast = close.ewm(span=fast).mean()
            ema_slow = close.ewm(span=slow).mean()

            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line

            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
        except Exception as e:
            print(f"Error calculating MACD: {e}")
            return {}

    def calculate_stochastic(self, data: pd.DataFrame, k_period: int = None, d_period: int = None) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator.

        Parameters:
        -----------
        data : DataFrame
            Price data with High, Low, Close columns
        k_period : int
            %K period
        d_period : int
            %D period

        Returns:
        --------
        Dict with %K and %D values
        """
        if k_period is None:
            k_period = self.config.stoch_k_period
        if d_period is None:
            d_period = self.config.stoch_d_period

        try:
            high = data['High']
            low = data['Low']
            close = data['Close']

            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()

            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()

            return {
                'k_percent': k_percent,
                'd_percent': d_percent
            }
        except Exception as e:
            print(f"Error calculating Stochastic: {e}")
            return {}

    def calculate_adx(self, data: pd.DataFrame, period: int = None) -> Dict[str, pd.Series]:
        """
        Calculate Average Directional Index (ADX).

        Parameters:
        -----------
        data : DataFrame
            Price data with High, Low, Close columns
        period : int
            ADX period

        Returns:
        --------
        Dict with ADX, +DI, -DI values
        """
        if period is None:
            period = self.config.adx_period

        try:
            high = data['High']
            low = data['Low']
            close = data['Close']

            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Directional Movement
            dm_plus = high.diff()
            dm_minus = low.diff()

            dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
            dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
            dm_minus = abs(dm_minus)

            # Smoothed averages
            atr = true_range.rolling(window=period).mean()
            di_plus = 100 * (dm_plus.rolling(window=period).mean() / atr)
            di_minus = 100 * (dm_minus.rolling(window=period).mean() / atr)

            # ADX calculation
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = dx.rolling(window=period).mean()

            return {
                'adx': adx,
                'di_plus': di_plus,
                'di_minus': di_minus
            }
        except Exception as e:
            print(f"Error calculating ADX: {e}")
            return {}

    def calculate_parabolic_sar(self, data: pd.DataFrame, acceleration: float = None, maximum: float = None) -> pd.Series:
        """
        Calculate Parabolic SAR.

        Parameters:
        -----------
        data : DataFrame
            Price data with High, Low columns
        acceleration : float
            Acceleration factor
        maximum : float
            Maximum acceleration factor

        Returns:
        --------
        Series with Parabolic SAR values
        """
        if acceleration is None:
            acceleration = self.config.sar_acceleration
        if maximum is None:
            maximum = self.config.sar_maximum

        try:
            high = data['High'].values
            low = data['Low'].values

            sar = np.zeros(len(data))
            trend = np.zeros(len(data))
            ep = np.zeros(len(data))
            af = np.zeros(len(data))

            # Initialize
            sar[0] = low[0]
            trend[0] = 1  # 1 for uptrend, -1 for downtrend
            ep[0] = high[0]
            af[0] = acceleration

            for i in range(1, len(data)):
                if trend[i-1] == 1:  # Uptrend
                    sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])

                    if low[i] <= sar[i]:
                        # Trend reversal
                        trend[i] = -1
                        sar[i] = ep[i-1]
                        ep[i] = low[i]
                        af[i] = acceleration
                    else:
                        trend[i] = 1
                        if high[i] > ep[i-1]:
                            ep[i] = high[i]
                            af[i] = min(af[i-1] + acceleration, maximum)
                        else:
                            ep[i] = ep[i-1]
                            af[i] = af[i-1]

                else:  # Downtrend
                    sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])

                    if high[i] >= sar[i]:
                        # Trend reversal
                        trend[i] = 1
                        sar[i] = ep[i-1]
                        ep[i] = high[i]
                        af[i] = acceleration
                    else:
                        trend[i] = -1
                        if low[i] < ep[i-1]:
                            ep[i] = low[i]
                            af[i] = min(af[i-1] + acceleration, maximum)
                        else:
                            ep[i] = ep[i-1]
                            af[i] = af[i-1]

            return pd.Series(sar, index=data.index)
        except Exception as e:
            print(f"Error calculating Parabolic SAR: {e}")
            return pd.Series()

    def comprehensive_technical_analysis(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        period : str
            Data period

        Returns:
        --------
        Dict with comprehensive technical analysis
        """
        try:
            # Get price data
            data = self.get_price_data(ticker, period)
            if data.empty:
                return {'error': f'No price data available for {ticker}'}

            # Calculate all indicators
            rsi = self.calculate_rsi(data)
            macd = self.calculate_macd(data)
            stochastic = self.calculate_stochastic(data)
            adx = self.calculate_adx(data)
            sar = self.calculate_parabolic_sar(data)

            # Get latest values
            latest_values = {
                'rsi': round(rsi.iloc[-1], self.config.precision) if not rsi.empty else None,
                'macd': {
                    'macd': round(macd['macd'].iloc[-1], self.config.precision) if 'macd' in macd and not macd['macd'].empty else None,
                    'signal': round(macd['signal'].iloc[-1], self.config.precision) if 'signal' in macd and not macd['signal'].empty else None,
                    'histogram': round(macd['histogram'].iloc[-1], self.config.precision) if 'histogram' in macd and not macd['histogram'].empty else None
                },
                'stochastic': {
                    'k_percent': round(stochastic['k_percent'].iloc[-1], self.config.precision) if 'k_percent' in stochastic and not stochastic['k_percent'].empty else None,
                    'd_percent': round(stochastic['d_percent'].iloc[-1], self.config.precision) if 'd_percent' in stochastic and not stochastic['d_percent'].empty else None
                },
                'adx': {
                    'adx': round(adx['adx'].iloc[-1], self.config.precision) if 'adx' in adx and not adx['adx'].empty else None,
                    'di_plus': round(adx['di_plus'].iloc[-1], self.config.precision) if 'di_plus' in adx and not adx['di_plus'].empty else None,
                    'di_minus': round(adx['di_minus'].iloc[-1], self.config.precision) if 'di_minus' in adx and not adx['di_minus'].empty else None
                },
                'parabolic_sar': round(sar.iloc[-1], self.config.precision) if not sar.empty else None,
                'current_price': round(data['Close'].iloc[-1], self.config.precision)
            }

            # Generate signals
            signals = self._generate_technical_signals(latest_values)

            # Calculate trend strength
            trend_analysis = self._analyze_trend_strength(data, latest_values)

            # Support and resistance levels
            support_resistance = self._calculate_support_resistance(data)

            return {
                'ticker': ticker,
                'analysis_date': datetime.now(),
                'period': period,
                'latest_indicators': latest_values,
                'signals': signals,
                'trend_analysis': trend_analysis,
                'support_resistance': support_resistance,
                'price_data': data.tail(10).to_dict(),  # Last 10 days
                'analysis_type': 'comprehensive_technical'
            }

        except Exception as e:
            return {'error': str(e)}

    def _generate_technical_signals(self, indicators: Dict) -> Dict[str, Any]:
        """Generate trading signals from technical indicators."""
        signals = {}

        try:
            # RSI Signals
            if indicators['rsi'] is not None:
                rsi = indicators['rsi']
                if rsi > 70:
                    signals['rsi_signal'] = 'SELL - Overbought'
                elif rsi < 30:
                    signals['rsi_signal'] = 'BUY - Oversold'
                else:
                    signals['rsi_signal'] = 'NEUTRAL'

            # MACD Signals
            if (indicators['macd']['macd'] is not None and
                indicators['macd']['signal'] is not None):
                macd_val = indicators['macd']['macd']
                signal_val = indicators['macd']['signal']

                if macd_val > signal_val:
                    signals['macd_signal'] = 'BUY - MACD above signal'
                else:
                    signals['macd_signal'] = 'SELL - MACD below signal'

            # Stochastic Signals
            if (indicators['stochastic']['k_percent'] is not None and
                indicators['stochastic']['d_percent'] is not None):
                k = indicators['stochastic']['k_percent']
                d = indicators['stochastic']['d_percent']

                if k > 80 and d > 80:
                    signals['stochastic_signal'] = 'SELL - Overbought'
                elif k < 20 and d < 20:
                    signals['stochastic_signal'] = 'BUY - Oversold'
                elif k > d:
                    signals['stochastic_signal'] = 'BUY - %K above %D'
                else:
                    signals['stochastic_signal'] = 'SELL - %K below %D'

            # ADX Signals
            if indicators['adx']['adx'] is not None:
                adx_val = indicators['adx']['adx']
                if adx_val > 25:
                    signals['adx_signal'] = 'STRONG TREND'
                elif adx_val > 20:
                    signals['adx_signal'] = 'MODERATE TREND'
                else:
                    signals['adx_signal'] = 'WEAK TREND'

            # Parabolic SAR Signal
            if (indicators['parabolic_sar'] is not None and
                indicators['current_price'] is not None):
                if indicators['current_price'] > indicators['parabolic_sar']:
                    signals['sar_signal'] = 'BUY - Price above SAR'
                else:
                    signals['sar_signal'] = 'SELL - Price below SAR'

            # Overall Signal
            buy_signals = sum(1 for signal in signals.values() if 'BUY' in str(signal))
            sell_signals = sum(1 for signal in signals.values() if 'SELL' in str(signal))

            if buy_signals > sell_signals:
                signals['overall_signal'] = 'BUY'
            elif sell_signals > buy_signals:
                signals['overall_signal'] = 'SELL'
            else:
                signals['overall_signal'] = 'NEUTRAL'

            signals['signal_strength'] = abs(buy_signals - sell_signals) / len(signals) if len(signals) > 0 else 0

            return signals

        except Exception as e:
            return {'error': str(e)}

    def _analyze_trend_strength(self, data: pd.DataFrame, indicators: Dict) -> Dict[str, Any]:
        """Analyze trend strength and direction."""
        try:
            trend_analysis = {}

            # Price trend (20-day moving average)
            if len(data) >= 20:
                ma_20 = data['Close'].rolling(window=20).mean()
                current_price = data['Close'].iloc[-1]
                ma_20_current = ma_20.iloc[-1]

                if current_price > ma_20_current:
                    trend_analysis['price_vs_ma20'] = 'BULLISH'
                else:
                    trend_analysis['price_vs_ma20'] = 'BEARISH'

            # Volume trend
            if 'Volume' in data.columns and len(data) >= 10:
                recent_volume = data['Volume'].tail(5).mean()
                previous_volume = data['Volume'].iloc[-15:-5].mean() if len(data) >= 15 else recent_volume

                if recent_volume > previous_volume * 1.2:
                    trend_analysis['volume_trend'] = 'INCREASING'
                elif recent_volume < previous_volume * 0.8:
                    trend_analysis['volume_trend'] = 'DECREASING'
                else:
                    trend_analysis['volume_trend'] = 'STABLE'

            # Volatility analysis
            if len(data) >= 20:
                returns = data['Close'].pct_change().dropna()
                volatility = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252)  # Annualized
                trend_analysis['volatility'] = round(volatility * 100, self.config.precision)

                if volatility > 0.3:
                    trend_analysis['volatility_level'] = 'HIGH'
                elif volatility > 0.2:
                    trend_analysis['volatility_level'] = 'MODERATE'
                else:
                    trend_analysis['volatility_level'] = 'LOW'

            # ADX trend strength
            if indicators['adx']['adx'] is not None:
                adx_val = indicators['adx']['adx']
                if adx_val > 40:
                    trend_analysis['trend_strength'] = 'VERY STRONG'
                elif adx_val > 25:
                    trend_analysis['trend_strength'] = 'STRONG'
                elif adx_val > 20:
                    trend_analysis['trend_strength'] = 'MODERATE'
                else:
                    trend_analysis['trend_strength'] = 'WEAK'

            return trend_analysis

        except Exception as e:
            return {'error': str(e)}

    def _calculate_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
        """Calculate support and resistance levels."""
        try:
            support_resistance = {}

            if len(data) >= window * 2:
                # Recent highs and lows
                recent_high = data['High'].tail(window).max()
                recent_low = data['Low'].tail(window).min()

                # Pivot points
                highs = data['High'].rolling(window=window, center=True).max()
                lows = data['Low'].rolling(window=window, center=True).min()

                # Find pivot highs and lows
                pivot_highs = data['High'][data['High'] == highs].dropna()
                pivot_lows = data['Low'][data['Low'] == lows].dropna()

                # Get most recent levels
                resistance_levels = sorted(pivot_highs.tail(3).values, reverse=True)
                support_levels = sorted(pivot_lows.tail(3).values)

                support_resistance.update({
                    'immediate_resistance': round(recent_high, 2),
                    'immediate_support': round(recent_low, 2),
                    'resistance_levels': [round(level, 2) for level in resistance_levels[:3]],
                    'support_levels': [round(level, 2) for level in support_levels[:3]],
                    'current_price': round(data['Close'].iloc[-1], 2)
                })

            return support_resistance

        except Exception as e:
            return {'error': str(e)}

    def plot_technical_analysis(self, ticker: str, period: str = "6mo") -> Dict[str, Any]:
        """
        Create comprehensive technical analysis charts.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        period : str
            Data period for plotting

        Returns:
        --------
        Dict with plot information
        """
        try:
            # Get data and indicators
            data = self.get_price_data(ticker, period)
            if data.empty:
                return {'error': f'No data available for {ticker}'}

            # Calculate indicators
            rsi = self.calculate_rsi(data)
            macd = self.calculate_macd(data)
            stochastic = self.calculate_stochastic(data)
            sar = self.calculate_parabolic_sar(data)

            # Create subplots
            plt.style.use(self.config.plot_style)
            fig, axes = plt.subplots(4, 1, figsize=(15, 20))

            # Price and SAR
            axes[0].plot(data.index, data['Close'], label='Close Price', linewidth=2)
            if not sar.empty:
                axes[0].scatter(data.index, sar, c=['red' if data['Close'].iloc[i] < sar.iloc[i] else 'green' for i in range(len(sar))],
                               s=20, alpha=0.7, label='Parabolic SAR')
            axes[0].set_title(f'{ticker} - Price and Parabolic SAR')
            axes[0].set_ylabel('Price')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # RSI
            if not rsi.empty:
                axes[1].plot(data.index, rsi, label='RSI', color='purple', linewidth=2)
                axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
                axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
                axes[1].axhline(y=50, color='gray', linestyle='-', alpha=0.5)
                axes[1].set_title('RSI (Relative Strength Index)')
                axes[1].set_ylabel('RSI')
                axes[1].set_ylim(0, 100)
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            # MACD
            if macd:
                axes[2].plot(data.index, macd['macd'], label='MACD', linewidth=2)
                axes[2].plot(data.index, macd['signal'], label='Signal', linewidth=2)
                axes[2].bar(data.index, macd['histogram'], label='Histogram', alpha=0.7)
                axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                axes[2].set_title('MACD (Moving Average Convergence Divergence)')
                axes[2].set_ylabel('MACD')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)

            # Stochastic
            if stochastic:
                axes[3].plot(data.index, stochastic['k_percent'], label='%K', linewidth=2)
                axes[3].plot(data.index, stochastic['d_percent'], label='%D', linewidth=2)
                axes[3].axhline(y=80, color='r', linestyle='--', alpha=0.7, label='Overbought (80)')
                axes[3].axhline(y=20, color='g', linestyle='--', alpha=0.7, label='Oversold (20)')
                axes[3].set_title('Stochastic Oscillator')
                axes[3].set_ylabel('Stochastic %')
                axes[3].set_xlabel('Date')
                axes[3].set_ylim(0, 100)
                axes[3].legend()
                axes[3].grid(True, alpha=0.3)

            plt.tight_layout()

            return {
                'ticker': ticker,
                'period': period,
                'plot_created': True,
                'indicators_plotted': ['price', 'parabolic_sar', 'rsi', 'macd', 'stochastic'],
                'data_points': len(data)
            }

        except Exception as e:
            return {'error': str(e)}


class BehavioralAnalysis:
    """
    Behavioral and sentiment analysis for cognitive biases and market psychology.
    """

    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def analyze_overconfidence_bias(self, analyst_estimates: List[float], actual_results: List[float]) -> Dict[str, Any]:
        """
        Analyze overconfidence bias in analyst estimates.

        Parameters:
        -----------
        analyst_estimates : List[float]
            Analyst earnings estimates
        actual_results : List[float]
            Actual earnings results

        Returns:
        --------
        Dict with overconfidence analysis
        """
        try:
            if len(analyst_estimates) != len(actual_results):
                return {'error': 'Estimates and results must have the same length'}

            estimates = np.array(analyst_estimates)
            actuals = np.array(actual_results)

            # Calculate forecast errors
            errors = estimates - actuals
            absolute_errors = np.abs(errors)
            percentage_errors = np.abs(errors / actuals) * 100

            # Overconfidence indicators
            mean_error = np.mean(errors)
            mean_absolute_error = np.mean(absolute_errors)
            mean_absolute_percentage_error = np.mean(percentage_errors[np.isfinite(percentage_errors)])

            # Bias direction
            positive_bias = np.sum(errors > 0) / len(errors)  # Proportion of overestimates

            # Confidence intervals (analysts tend to be overconfident if actual volatility exceeds predicted)
            estimate_volatility = np.std(estimates)
            actual_volatility = np.std(actuals)
            volatility_ratio = actual_volatility / estimate_volatility if estimate_volatility > 0 else float('inf')

            # Overconfidence score (higher means more overconfident)
            # Based on MAPE and volatility underestimation
            overconfidence_score = min(100, (mean_absolute_percentage_error * 2 + max(0, (volatility_ratio - 1) * 50)))

            # Determine overconfidence level
            if overconfidence_score > self.config.overconfidence_threshold * 100:
                confidence_level = 'HIGH'
            elif overconfidence_score > (self.config.overconfidence_threshold * 50):
                confidence_level = 'MODERATE'
            else:
                confidence_level = 'LOW'

            return {
                'analysis_type': 'overconfidence_bias',
                'sample_size': len(estimates),
                'mean_error': round(mean_error, self.config.precision),
                'mean_absolute_error': round(mean_absolute_error, self.config.precision),
                'mean_absolute_percentage_error': round(mean_absolute_percentage_error, self.config.precision),
                'positive_bias_ratio': round(positive_bias, self.config.precision),
                'estimate_volatility': round(estimate_volatility, self.config.precision),
                'actual_volatility': round(actual_volatility, self.config.precision),
                'volatility_ratio': round(volatility_ratio, self.config.precision),
                'overconfidence_score': round(overconfidence_score, self.config.precision),
                'confidence_level': confidence_level,
                'interpretation': self._interpret_overconfidence(overconfidence_score, positive_bias)
            }

        except Exception as e:
            return {'error': str(e)}

    def analyze_anchoring_bias(self, prices: List[float], reference_points: List[float] = None) -> Dict[str, Any]:
        """
        Analyze anchoring bias in price movements.

        Parameters:
        -----------
        prices : List[float]
            Historical price series
        reference_points : List[float]
            Reference anchoring points (e.g., 52-week high, analyst targets)

        Returns:
        --------
        Dict with anchoring bias analysis
        """
        try:
            prices = np.array(prices)

            if reference_points is None:
                # Use common anchoring points
                reference_points = [
                    np.max(prices),  # 52-week high
                    np.min(prices),  # 52-week low
                    prices[0],      # Starting price
                    np.mean(prices)  # Average price
                ]

            anchoring_analysis = {}

            # Calculate distance from anchoring points
            current_price = prices[-1]

            for i, anchor in enumerate(reference_points):
                anchor_name = ['52w_high', '52w_low', 'initial_price', 'average_price'][i] if i < 4 else f'anchor_{i}'

                distance_from_anchor = abs(current_price - anchor) / anchor * 100
                anchoring_analysis[f'{anchor_name}_distance'] = round(distance_from_anchor, self.config.precision)

                # Check if price clusters around this anchor
                tolerance = 0.05  # 5% tolerance
                near_anchor_periods = np.sum(np.abs(prices - anchor) / anchor < tolerance)
                clustering_ratio = near_anchor_periods / len(prices)
                anchoring_analysis[f'{anchor_name}_clustering'] = round(clustering_ratio, self.config.precision)

            # Price momentum vs anchoring
            recent_prices = prices[-10:] if len(prices) >= 10 else prices
            price_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

            # Resistance/support at psychological levels
            psychological_levels = self._identify_psychological_levels(prices)

            # Overall anchoring bias score
            max_clustering = max([val for key, val in anchoring_analysis.items() if 'clustering' in key])
            anchoring_score = max_clustering * 100

            if anchoring_score > 30:
                bias_level = 'HIGH'
            elif anchoring_score > 15:
                bias_level = 'MODERATE'
            else:
                bias_level = 'LOW'

            return {
                'analysis_type': 'anchoring_bias',
                'current_price': round(current_price, self.config.precision),
                'reference_points': {f'point_{i}': round(point, self.config.precision) for i, point in enumerate(reference_points)},
                'anchoring_distances': {k: v for k, v in anchoring_analysis.items() if 'distance' in k},
                'clustering_ratios': {k: v for k, v in anchoring_analysis.items() if 'clustering' in k},
                'price_momentum': round(price_momentum, self.config.precision),
                'psychological_levels': psychological_levels,
                'anchoring_score': round(anchoring_score, self.config.precision),
                'bias_level': bias_level,
                'interpretation': self._interpret_anchoring(anchoring_score, max_clustering)
            }

        except Exception as e:
            return {'error': str(e)}

    def analyze_disposition_effect(self, trades: List[Dict]) -> Dict[str, Any]:
        """
        Analyze disposition effect in trading behavior.

        Parameters:
        -----------
        trades : List[Dict]
            List of trade dictionaries with keys: 'entry_price', 'exit_price', 'holding_period'

        Returns:
        --------
        Dict with disposition effect analysis
        """
        try:
            if not trades:
                return {'error': 'No trade data provided'}

            gains = []
            losses = []
            gain_holding_periods = []
            loss_holding_periods = []

            for trade in trades:
                entry_price = trade['entry_price']
                exit_price = trade['exit_price']
                holding_period = trade.get('holding_period', 1)

                pnl = (exit_price - entry_price) / entry_price

                if pnl > 0:
                    gains.append(pnl)
                    gain_holding_periods.append(holding_period)
                else:
                    losses.append(pnl)
                    loss_holding_periods.append(holding_period)

            if not gains and not losses:
                return {'error': 'No valid trades found'}

            # Calculate key metrics
            total_trades = len(trades)
            winning_trades = len(gains)
            losing_trades = len(losses)
            win_rate = winning_trades / total_trades

            # Average holding periods
            avg_gain_holding = np.mean(gain_holding_periods) if gain_holding_periods else 0
            avg_loss_holding = np.mean(loss_holding_periods) if loss_holding_periods else 0

            # Disposition effect ratio (higher means more disposition effect)
            # Disposition effect: tendency to sell winners too early and hold losers too long
            if avg_gain_holding > 0 and avg_loss_holding > 0:
                disposition_ratio = avg_loss_holding / avg_gain_holding
            else:
                disposition_ratio = 1.0

            # Average gains vs losses
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0

            # Disposition effect score
            # Higher score indicates stronger disposition effect
            if disposition_ratio > 1:
                disposition_score = min(100, (disposition_ratio - 1) * 50 +
                                      (1 - min(avg_gain / abs(avg_loss) if avg_loss != 0 else 1, 2)) * 25)
            else:
                disposition_score = max(0, 50 - (1 - disposition_ratio) * 25)

            # Determine disposition effect level
            if disposition_score > 70:
                effect_level = 'HIGH'
            elif disposition_score > 40:
                effect_level = 'MODERATE'
            else:
                effect_level = 'LOW'

            return {
                'analysis_type': 'disposition_effect',
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': round(win_rate, self.config.precision),
                'average_gain': round(avg_gain * 100, self.config.precision),  # Percentage
                'average_loss': round(avg_loss * 100, self.config.precision),   # Percentage
                'avg_gain_holding_period': round(avg_gain_holding, self.config.precision),
                'avg_loss_holding_period': round(avg_loss_holding, self.config.precision),
                'disposition_ratio': round(disposition_ratio, self.config.precision),
                'disposition_score': round(disposition_score, self.config.precision),
                'effect_level': effect_level,
                'interpretation': self._interpret_disposition_effect(disposition_score, disposition_ratio)
            }

        except Exception as e:
            return {'error': str(e)}

    def analyze_herding_bias(self, price_data: pd.DataFrame, volume_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analyze herding bias using price and volume patterns.

        Parameters:
        -----------
        price_data : DataFrame
            Price data with datetime index
        volume_data : DataFrame
            Volume data (optional)

        Returns:
        --------
        Dict with herding bias analysis
        """
        try:
            if price_data.empty:
                return {'error': 'No price data provided'}

            # Convert to Series if DataFrame
            if isinstance(price_data, pd.DataFrame):
                if 'Close' in price_data.columns:
                    prices = price_data['Close']
                else:
                    prices = price_data.iloc[:, 0]  # First column
            else:
                prices = price_data

            # Calculate returns
            returns = prices.pct_change().dropna()

            # Herding metrics
            herding_analysis = {}

            # 1. Return autocorrelation (herding leads to momentum)
            if len(returns) > 1:
                autocorr_1 = returns.autocorr(lag=1)
                autocorr_5 = returns.autocorr(lag=5) if len(returns) > 5 else 0
                herding_analysis['return_autocorr_1day'] = round(autocorr_1, self.config.precision)
                herding_analysis['return_autocorr_5day'] = round(autocorr_5, self.config.precision)

            # 2. Clustering of extreme movements (herding in stress)
            extreme_threshold = np.percentile(np.abs(returns), 90)
            extreme_days = np.abs(returns) > extreme_threshold

            # Count consecutive extreme days
            consecutive_extremes = []
            count = 0
            for is_extreme in extreme_days:
                if is_extreme:
                    count += 1
                else:
                    if count > 0:
                        consecutive_extremes.append(count)
                    count = 0
            if count > 0:
                consecutive_extremes.append(count)

            avg_consecutive_extremes = np.mean(consecutive_extremes) if consecutive_extremes else 0
            herding_analysis['avg_consecutive_extreme_days'] = round(avg_consecutive_extremes, self.config.precision)

            # 3. Volatility clustering
            rolling_vol = returns.rolling(window=20).std()
            vol_autocorr = rolling_vol.dropna().autocorr(lag=1) if len(rolling_vol.dropna()) > 1 else 0
            herding_analysis['volatility_clustering'] = round(vol_autocorr, self.config.precision)

            # 4. Volume-price relationship (if volume data available)
            if volume_data is not None and not volume_data.empty:
                if isinstance(volume_data, pd.DataFrame):
                    volume = volume_data.iloc[:, 0]
                else:
                    volume = volume_data

                # Align price and volume data
                common_index = prices.index.intersection(volume.index)
                if len(common_index) > 10:
                    aligned_returns = returns.loc[common_index]
                    aligned_volume = volume.loc[common_index]

                    # Volume spike correlation with large price moves
                    volume_changes = aligned_volume.pct_change().dropna()
                    price_vol_corr = np.corrcoef(np.abs(aligned_returns[1:]), volume_changes)[0, 1]
                    herding_analysis['volume_price_correlation'] = round(price_vol_corr, self.config.precision)

            # 5. Cross-sectional return dispersion (proxy using rolling correlation with market)
            # This would require market data - simplified version using return consistency
            return_consistency = 1 - (returns.std() / abs(returns.mean())) if returns.mean() != 0 else 0
            herding_analysis['return_consistency'] = round(max(0, return_consistency), self.config.precision)

            # Calculate overall herding score
            herding_indicators = [
                abs(autocorr_1) if 'return_autocorr_1day' in herding_analysis else 0,
                avg_consecutive_extremes / 5,  # Normalize
                abs(vol_autocorr),
                herding_analysis.get('volume_price_correlation', 0),
                herding_analysis.get('return_consistency', 0)
            ]

            herding_score = np.mean([indicator for indicator in herding_indicators if not np.isnan(indicator)]) * 100

            # Determine herding level
            if herding_score > self.config.herding_threshold * 100:
                herding_level = 'HIGH'
            elif herding_score > (self.config.herding_threshold * 50):
                herding_level = 'MODERATE'
            else:
                herding_level = 'LOW'

            return {
                'analysis_type': 'herding_bias',
                'sample_size': len(returns),
                'herding_indicators': herding_analysis,
                'herding_score': round(herding_score, self.config.precision),
                'herding_level': herding_level,
                'interpretation': self._interpret_herding(herding_score, herding_analysis)
            }

        except Exception as e:
            return {'error': str(e)}

    def sentiment_analysis_from_text(self, text_data: List[str], source: str = 'news') -> Dict[str, Any]:
        """
        Perform sentiment analysis on text data (news, social media, etc.).

        Parameters:
        -----------
        text_data : List[str]
            List of text strings to analyze
        source : str
            Source of the text data

        Returns:
        --------
        Dict with sentiment analysis results
        """
        try:
            if not text_data:
                return {'error': 'No text data provided'}

            sentiments = []
            word_counts = []

            for text in text_data:
                if not text or not isinstance(text, str):
                    continue

                # VADER sentiment analysis
                vader_scores = self.sentiment_analyzer.polarity_scores(text)

                # TextBlob sentiment analysis
                blob = TextBlob(text)
                textblob_polarity = blob.sentiment.polarity
                textblob_subjectivity = blob.sentiment.subjectivity

                # Combine scores
                sentiment_score = (vader_scores['compound'] + textblob_polarity) / 2

                sentiments.append({
                    'text_length': len(text),
                    'vader_compound': vader_scores['compound'],
                    'vader_positive': vader_scores['pos'],
                    'vader_negative': vader_scores['neg'],
                    'vader_neutral': vader_scores['neu'],
                    'textblob_polarity': textblob_polarity,
                    'textblob_subjectivity': textblob_subjectivity,
                    'combined_sentiment': sentiment_score
                })

                # Count words
                words = text.lower().split()
                word_counts.extend(words)

            if not sentiments:
                return {'error': 'No valid text data found'}

            # Aggregate sentiment metrics
            sentiment_df = pd.DataFrame(sentiments)

            aggregate_sentiment = {
                'total_texts': len(sentiments),
                'average_sentiment': round(sentiment_df['combined_sentiment'].mean(), self.config.precision),
                'sentiment_std': round(sentiment_df['combined_sentiment'].std(), self.config.precision),
                'positive_ratio': round((sentiment_df['combined_sentiment'] > 0.1).mean(), self.config.precision),
                'negative_ratio': round((sentiment_df['combined_sentiment'] < -0.1).mean(), self.config.precision),
                'neutral_ratio': round(((sentiment_df['combined_sentiment'] >= -0.1) &
                                      (sentiment_df['combined_sentiment'] <= 0.1)).mean(), self.config.precision),
                'subjectivity_avg': round(sentiment_df['textblob_subjectivity'].mean(), self.config.precision)
            }

            # Most common words
            word_frequency = Counter(word_counts)
            most_common_words = word_frequency.most_common(20)

            # Sentiment trend (if more than 10 texts)
            if len(sentiments) >= 10:
                sentiment_trend = self._analyze_sentiment_trend(sentiment_df['combined_sentiment'])
                aggregate_sentiment['sentiment_trend'] = sentiment_trend

            # Market sentiment interpretation
            avg_sentiment = aggregate_sentiment['average_sentiment']
            if avg_sentiment > 0.3:
                market_sentiment = 'VERY POSITIVE'
            elif avg_sentiment > 0.1:
                market_sentiment = 'POSITIVE'
            elif avg_sentiment > -0.1:
                market_sentiment = 'NEUTRAL'
            elif avg_sentiment > -0.3:
                market_sentiment = 'NEGATIVE'
            else:
                market_sentiment = 'VERY NEGATIVE'

            return {
                'analysis_type': 'sentiment_analysis',
                'source': source,
                'aggregate_sentiment': aggregate_sentiment,
                'market_sentiment': market_sentiment,
                'most_common_words': most_common_words[:10],
                'individual_sentiments': sentiments[:10],  # First 10 for reference
                'sentiment_distribution': {
                    'very_positive': round((sentiment_df['combined_sentiment'] > 0.5).mean() * 100, 1),
                    'positive': round(((sentiment_df['combined_sentiment'] > 0.1) &
                                    (sentiment_df['combined_sentiment'] <= 0.5)).mean() * 100, 1),
                    'neutral': round(((sentiment_df['combined_sentiment'] >= -0.1) &
                                   (sentiment_df['combined_sentiment'] <= 0.1)).mean() * 100, 1),
                    'negative': round(((sentiment_df['combined_sentiment'] >= -0.5) &
                                     (sentiment_df['combined_sentiment'] < -0.1)).mean() * 100, 1),
                    'very_negative': round((sentiment_df['combined_sentiment'] < -0.5).mean() * 100, 1)
                }
            }

        except Exception as e:
            return {'error': str(e)}

    def comprehensive_behavioral_analysis(self, ticker: str, price_data: pd.DataFrame = None,
                                        news_data: List[str] = None, trade_data: List[Dict] = None) -> Dict[str, Any]:
        """
        Perform comprehensive behavioral analysis combining multiple bias analyses.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        price_data : DataFrame
            Price data for herding and anchoring analysis
        news_data : List[str]
            News articles for sentiment analysis
        trade_data : List[Dict]
            Trading data for disposition effect analysis

        Returns:
        --------
        Dict with comprehensive behavioral analysis
        """
        try:
            analysis_results = {
                'ticker': ticker,
                'analysis_date': datetime.now(),
                'analysis_type': 'comprehensive_behavioral'
            }

            # Price-based analyses
            if price_data is not None and not price_data.empty:
                # Herding bias analysis
                herding_analysis = self.analyze_herding_bias(price_data)
                analysis_results['herding_analysis'] = herding_analysis

                # Anchoring bias analysis
                if 'Close' in price_data.columns:
                    prices = price_data['Close'].values
                    anchoring_analysis = self.analyze_anchoring_bias(prices)
                    analysis_results['anchoring_analysis'] = anchoring_analysis

            # Sentiment analysis
            if news_data:
                sentiment_analysis = self.sentiment_analysis_from_text(news_data, 'news')
                analysis_results['sentiment_analysis'] = sentiment_analysis

            # Disposition effect analysis
            if trade_data:
                disposition_analysis = self.analyze_disposition_effect(trade_data)
                analysis_results['disposition_analysis'] = disposition_analysis

            # Generate overall behavioral score
            behavioral_score = self._calculate_behavioral_score(analysis_results)
            analysis_results['behavioral_score'] = behavioral_score

            # Investment implications
            implications = self._generate_behavioral_implications(analysis_results)
            analysis_results['investment_implications'] = implications

            return analysis_results

        except Exception as e:
            return {'error': str(e)}

    def _interpret_overconfidence(self, score: float, positive_bias: float) -> str:
        """Interpret overconfidence analysis results."""
        if score > 60:
            return f"High overconfidence detected. Forecasts show {positive_bias:.1%} positive bias with significant accuracy issues."
        elif score > 30:
            return f"Moderate overconfidence present. Analysts show some forecasting bias with {positive_bias:.1%} tendency to overestimate."
        else:
            return "Low overconfidence. Forecasts appear reasonably calibrated."

    def _interpret_anchoring(self, score: float, max_clustering: float) -> str:
        """Interpret anchoring bias analysis results."""
        if score > 30:
            return f"Strong anchoring bias detected. Price clusters around reference points {max_clustering:.1%} of the time."
        elif score > 15:
            return f"Moderate anchoring bias. Some price clustering around reference levels ({max_clustering:.1%})."
        else:
            return "Low anchoring bias. Prices move relatively independently of reference points."

    def _interpret_disposition_effect(self, score: float, ratio: float) -> str:
        """Interpret disposition effect analysis results."""
        if score > 70:
            return f"Strong disposition effect. Losing positions held {ratio:.1f}x longer than winning positions."
        elif score > 40:
            return f"Moderate disposition effect detected. Some tendency to hold losers longer (ratio: {ratio:.1f})."
        else:
            return "Low disposition effect. Trading behavior shows disciplined position management."

    def _interpret_herding(self, score: float, indicators: Dict) -> str:
        """Interpret herding bias analysis results."""
        if score > 70:
            return "Strong herding behavior detected. High correlation in movements and volatility clustering."
        elif score > 35:
            return "Moderate herding tendencies. Some momentum and clustering in price movements."
        else:
            return "Low herding behavior. Price movements appear relatively independent."

    def _identify_psychological_levels(self, prices: np.ndarray) -> List[float]:
        """Identify psychological price levels (round numbers)."""
        current_price = prices[-1]

        # Generate round number levels around current price
        magnitude = 10 ** (len(str(int(current_price))) - 1)

        psychological_levels = []
        for multiplier in [0.5, 1, 1.5, 2, 2.5]:
            level = magnitude * multiplier
            if abs(level - current_price) / current_price < 0.5:  # Within 50% of current price
                psychological_levels.append(round(level, 2))

        return sorted(psychological_levels)

    def _analyze_sentiment_trend(self, sentiments: pd.Series) -> str:
        """Analyze trend in sentiment data."""
        if len(sentiments) < 5:
            return 'INSUFFICIENT_DATA'

        # Simple linear trend
        x = np.arange(len(sentiments))
        slope, _, r_value, _, _ = stats.linregress(x, sentiments)

        r_squared = r_value ** 2

        if r_squared < 0.3:
            return 'VOLATILE'
        elif slope > 0.01:
            return 'IMPROVING'
        elif slope < -0.01:
            return 'DETERIORATING'
        else:
            return 'STABLE'

    def _calculate_behavioral_score(self, analysis_results: Dict) -> Dict[str, Any]:
        """Calculate overall behavioral bias score."""
        try:
            scores = {}
            weights = {'herding': 0.25, 'anchoring': 0.25, 'sentiment': 0.25, 'disposition': 0.25}

            # Herding score
            if 'herding_analysis' in analysis_results and 'herding_score' in analysis_results['herding_analysis']:
                scores['herding'] = analysis_results['herding_analysis']['herding_score']

            # Anchoring score
            if 'anchoring_analysis' in analysis_results and 'anchoring_score' in analysis_results['anchoring_analysis']:
                scores['anchoring'] = analysis_results['anchoring_analysis']['anchoring_score']

            # Sentiment score (convert to bias score)
            if 'sentiment_analysis' in analysis_results:
                avg_sentiment = analysis_results['sentiment_analysis']['aggregate_sentiment']['average_sentiment']
                # Convert sentiment to bias score (extreme sentiment = higher bias)
                sentiment_bias = abs(avg_sentiment) * 100
                scores['sentiment'] = sentiment_bias

            # Disposition score
            if 'disposition_analysis' in analysis_results and 'disposition_score' in analysis_results['disposition_analysis']:
                scores['disposition'] = analysis_results['disposition_analysis']['disposition_score']

            # Calculate weighted average
            if scores:
                total_weight = sum(weights[key] for key in scores.keys())
                weighted_score = sum(scores[key] * weights.get(key, 0) for key in scores.keys()) / total_weight

                # Overall bias level
                if weighted_score > 70:
                    bias_level = 'HIGH'
                elif weighted_score > 40:
                    bias_level = 'MODERATE'
                else:
                    bias_level = 'LOW'

                return {
                    'individual_scores': scores,
                    'overall_score': round(weighted_score, self.config.precision),
                    'bias_level': bias_level,
                    'components_analyzed': len(scores)
                }
            else:
                return {'error': 'No behavioral scores calculated'}

        except Exception as e:
            return {'error': str(e)}

    def _generate_behavioral_implications(self, analysis_results: Dict) -> Dict[str, Any]:
        """Generate investment implications from behavioral analysis."""
        try:
            implications = {
                'risk_factors': [],
                'opportunities': [],
                'recommendations': []
            }

            # Herding implications
            if 'herding_analysis' in analysis_results:
                herding_level = analysis_results['herding_analysis'].get('herding_level', 'LOW')
                if herding_level == 'HIGH':
                    implications['risk_factors'].append('High herding behavior increases crash risk')
                    implications['recommendations'].append('Consider contrarian strategies')

            # Anchoring implications
            if 'anchoring_analysis' in analysis_results:
                anchoring_level = analysis_results['anchoring_analysis'].get('bias_level', 'LOW')
                if anchoring_level == 'HIGH':
                    implications['opportunities'].append('Price anchoring creates predictable support/resistance')
                    implications['recommendations'].append('Use technical analysis around psychological levels')

            # Sentiment implications
            if 'sentiment_analysis' in analysis_results:
                market_sentiment = analysis_results['sentiment_analysis'].get('market_sentiment', 'NEUTRAL')
                if market_sentiment in ['VERY POSITIVE', 'VERY NEGATIVE']:
                    implications['risk_factors'].append(f'Extreme sentiment ({market_sentiment}) suggests reversal risk')
                    implications['recommendations'].append('Consider contrarian positioning')

            # Disposition effect implications
            if 'disposition_analysis' in analysis_results:
                effect_level = analysis_results['disposition_analysis'].get('effect_level', 'LOW')
                if effect_level == 'HIGH':
                    implications['opportunities'].append('Strong disposition effect creates momentum opportunities')
                    implications['recommendations'].append('Use momentum strategies and strict stop-losses')

            # Overall implications
            if 'behavioral_score' in analysis_results:
                overall_bias = analysis_results['behavioral_score'].get('bias_level', 'LOW')
                if overall_bias == 'HIGH':
                    implications['recommendations'].append('High behavioral bias environment - exercise caution')
                    implications['recommendations'].append('Consider systematic/algorithmic approaches')
                elif overall_bias == 'LOW':
                    implications['opportunities'].append('Low bias environment - fundamental analysis more reliable')

            return implications

        except Exception as e:
            return {'error': str(e)}


def create_word_cloud(text_data: List[str], title: str = "Word Cloud") -> Dict[str, Any]:
    """
    Create a word cloud from text data.

    Parameters:
    -----------
    text_data : List[str]
        List of text strings
    title : str
        Title for the word cloud

    Returns:
    --------
    Dict with word cloud information
    """
    try:
        if not text_data:
            return {'error': 'No text data provided'}

        # Combine all text
        combined_text = ' '.join(text_data)

        # Clean text
        cleaned_text = re.sub(r'[^\w\s]', '', combined_text.lower())

        # Create word cloud
        wordcloud = WordCloud(width=800, height=400,
                             background_color='white',
                             max_words=100,
                             colormap='viridis').generate(cleaned_text)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Get word frequencies
        word_freq = wordcloud.words_

        return {
            'word_cloud_created': True,
            'total_words': len(cleaned_text.split()),
            'unique_words': len(word_freq),
            'top_words': dict(list(word_freq.items())[:20])
        }

    except Exception as e:
        return {'error': str(e)}


# Example usage and testing functions
def example_fundamental_analysis():
    """Example of how to use fundamental analysis."""
    config = AnalysisConfig()
    fa = FundamentalAnalysis(config)

    # Analyze a stock
    result = fa.comprehensive_fundamental_analysis('AAPL', years=5)
    return result


def example_technical_analysis():
    """Example of how to use technical analysis."""
    config = AnalysisConfig()
    ta = TechnicalAnalysis(config)

    # Analyze a stock
    result = ta.comprehensive_technical_analysis('AAPL', period='1y')
    return result


def example_behavioral_analysis():
    """Example of how to use behavioral analysis."""
    config = AnalysisConfig()
    ba = BehavioralAnalysis(config)

    # Example price data
    ta = TechnicalAnalysis(config)
    price_data = ta.get_price_data('AAPL', '6mo')

    # Example news data
    news_data = [
        "Apple reports strong quarterly earnings beating expectations",
        "Investors are optimistic about Apple's new product launches",
        "Market analysts raise price targets for Apple stock"
    ]

    # Comprehensive analysis
    result = ba.comprehensive_behavioral_analysis('AAPL', price_data=price_data, news_data=news_data)
    return result


if __name__ == "__main__":
    # Run examples
    print("Stock Picking Analysis Framework")
    print("="*50)

    # Test fundamental analysis
    print("\n1. Fundamental Analysis Example:")
    fa_result = example_fundamental_analysis()
    print(f"Analysis completed for: {fa_result.get('ticker', 'N/A')}")

    # Test technical analysis
    print("\n2. Technical Analysis Example:")
    ta_result = example_technical_analysis()
    print(f"Analysis completed for: {ta_result.get('ticker', 'N/A')}")

    # Test behavioral analysis
    print("\n3. Behavioral Analysis Example:")
    ba_result = example_behavioral_analysis()
    print(f"Analysis completed for: {ba_result.get('ticker', 'N/A')}")

    print("\nAll analyses completed successfully!")