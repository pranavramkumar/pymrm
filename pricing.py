"""
Pricing: Comprehensive Financial Mathematics and Options Pricing Module

This module provides extensive financial mathematics functionality including:
- Time value of money calculations (simple/compound interest, yield conversions)
- Bond pricing, duration, and convexity
- Option payoffs and P&L for various option types
- Option pricing models (Black-Scholes, Binomial, Monte Carlo, etc.)
- Option Greeks calculations
- Option strategies and portfolio analysis
- Advanced derivatives pricing and risk management

Author: Claude AI
Created: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize_scalar, fsolve, newton
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from enum import Enum


class OptionType(Enum):
    """Option type enumeration."""
    CALL = "call"
    PUT = "put"


class OptionStyle(Enum):
    """Option exercise style enumeration."""
    EUROPEAN = "european"
    AMERICAN = "american"
    BERMUDAN = "bermudan"


@dataclass
class PricingConfig:
    """Configuration class for pricing calculations."""

    # General settings
    precision: int = 6
    max_iterations: int = 1000
    tolerance: float = 1e-8

    # Monte Carlo settings
    mc_simulations: int = 100000
    random_seed: int = 42

    # Binomial tree settings
    tree_steps: int = 100

    # Numerical methods
    finite_difference_h: float = 0.01

    # Plotting settings
    plot_style: str = 'seaborn-v0_8'
    figure_size: Tuple[int, int] = (12, 8)


class TimeValueMath:
    """
    Time value of money calculations and yield conversions.
    """

    def __init__(self, config: PricingConfig = None):
        self.config = config or PricingConfig()

    def simple_interest(self, principal: float, rate: float, time: float) -> Dict[str, Any]:
        """
        Calculate simple interest.

        Parameters:
        -----------
        principal : float
            Principal amount
        rate : float
            Annual interest rate (as decimal, e.g., 0.05 for 5%)
        time : float
            Time period in years

        Returns:
        --------
        Dict with interest and final amount
        """
        try:
            interest = principal * rate * time
            final_amount = principal + interest

            return {
                'principal': round(principal, self.config.precision),
                'rate': round(rate, self.config.precision),
                'time': round(time, self.config.precision),
                'interest': round(interest, self.config.precision),
                'final_amount': round(final_amount, self.config.precision),
                'method': 'simple_interest'
            }
        except Exception as e:
            return {'error': str(e)}

    def compound_interest(self, principal: float, rate: float, time: float,
                         compounding_periods: int = 1) -> Dict[str, Any]:
        """
        Calculate compound interest.

        Parameters:
        -----------
        principal : float
            Principal amount
        rate : float
            Annual interest rate (as decimal)
        time : float
            Time period in years
        compounding_periods : int
            Number of compounding periods per year

        Returns:
        --------
        Dict with compound interest calculations
        """
        try:
            periodic_rate = rate / compounding_periods
            total_periods = compounding_periods * time

            final_amount = principal * (1 + periodic_rate) ** total_periods
            compound_interest = final_amount - principal

            # Effective annual rate
            effective_rate = (1 + periodic_rate) ** compounding_periods - 1

            return {
                'principal': round(principal, self.config.precision),
                'annual_rate': round(rate, self.config.precision),
                'time': round(time, self.config.precision),
                'compounding_periods': compounding_periods,
                'periodic_rate': round(periodic_rate, self.config.precision),
                'total_periods': round(total_periods, self.config.precision),
                'compound_interest': round(compound_interest, self.config.precision),
                'final_amount': round(final_amount, self.config.precision),
                'effective_annual_rate': round(effective_rate, self.config.precision),
                'method': 'compound_interest'
            }
        except Exception as e:
            return {'error': str(e)}

    def continuous_compounding(self, principal: float, rate: float, time: float) -> Dict[str, Any]:
        """
        Calculate continuously compounded interest.

        Parameters:
        -----------
        principal : float
            Principal amount
        rate : float
            Annual interest rate (as decimal)
        time : float
            Time period in years

        Returns:
        --------
        Dict with continuous compounding calculations
        """
        try:
            final_amount = principal * np.exp(rate * time)
            interest = final_amount - principal

            return {
                'principal': round(principal, self.config.precision),
                'rate': round(rate, self.config.precision),
                'time': round(time, self.config.precision),
                'interest': round(interest, self.config.precision),
                'final_amount': round(final_amount, self.config.precision),
                'method': 'continuous_compounding'
            }
        except Exception as e:
            return {'error': str(e)}

    def holding_period_yield(self, initial_value: float, final_value: float,
                           dividends: float = 0) -> Dict[str, Any]:
        """
        Calculate holding period yield.

        Parameters:
        -----------
        initial_value : float
            Initial investment value
        final_value : float
            Final investment value
        dividends : float
            Dividends received during holding period

        Returns:
        --------
        Dict with holding period yield
        """
        try:
            if initial_value <= 0:
                return {'error': 'Initial value must be positive'}

            hpy = (final_value + dividends - initial_value) / initial_value

            return {
                'initial_value': round(initial_value, self.config.precision),
                'final_value': round(final_value, self.config.precision),
                'dividends': round(dividends, self.config.precision),
                'holding_period_yield': round(hpy, self.config.precision),
                'holding_period_return_percent': round(hpy * 100, self.config.precision)
            }
        except Exception as e:
            return {'error': str(e)}

    def annual_percentage_yield(self, rate: float, compounding_periods: int) -> Dict[str, Any]:
        """
        Calculate Annual Percentage Yield (APY).

        Parameters:
        -----------
        rate : float
            Nominal annual interest rate (as decimal)
        compounding_periods : int
            Number of compounding periods per year

        Returns:
        --------
        Dict with APY calculation
        """
        try:
            apy = (1 + rate / compounding_periods) ** compounding_periods - 1

            return {
                'nominal_rate': round(rate, self.config.precision),
                'compounding_periods': compounding_periods,
                'annual_percentage_yield': round(apy, self.config.precision),
                'apy_percent': round(apy * 100, self.config.precision)
            }
        except Exception as e:
            return {'error': str(e)}

    def effective_annual_yield(self, hpy: float, holding_period: float) -> Dict[str, Any]:
        """
        Calculate Effective Annual Yield from holding period yield.

        Parameters:
        -----------
        hpy : float
            Holding period yield (as decimal)
        holding_period : float
            Holding period in years

        Returns:
        --------
        Dict with effective annual yield
        """
        try:
            if holding_period <= 0:
                return {'error': 'Holding period must be positive'}

            eay = (1 + hpy) ** (1 / holding_period) - 1

            return {
                'holding_period_yield': round(hpy, self.config.precision),
                'holding_period_years': round(holding_period, self.config.precision),
                'effective_annual_yield': round(eay, self.config.precision),
                'eay_percent': round(eay * 100, self.config.precision)
            }
        except Exception as e:
            return {'error': str(e)}

    def present_value(self, future_value: float, rate: float, time: float,
                     compounding_periods: int = 1) -> Dict[str, Any]:
        """
        Calculate present value of future cash flow.

        Parameters:
        -----------
        future_value : float
            Future value
        rate : float
            Discount rate (as decimal)
        time : float
            Time to maturity in years
        compounding_periods : int
            Compounding periods per year

        Returns:
        --------
        Dict with present value calculation
        """
        try:
            periodic_rate = rate / compounding_periods
            total_periods = compounding_periods * time

            pv = future_value / (1 + periodic_rate) ** total_periods

            return {
                'future_value': round(future_value, self.config.precision),
                'discount_rate': round(rate, self.config.precision),
                'time': round(time, self.config.precision),
                'compounding_periods': compounding_periods,
                'present_value': round(pv, self.config.precision),
                'discount_factor': round(pv / future_value, self.config.precision)
            }
        except Exception as e:
            return {'error': str(e)}

    def future_value(self, present_value: float, rate: float, time: float,
                    compounding_periods: int = 1) -> Dict[str, Any]:
        """
        Calculate future value of present cash flow.

        Parameters:
        -----------
        present_value : float
            Present value
        rate : float
            Interest rate (as decimal)
        time : float
            Time to maturity in years
        compounding_periods : int
            Compounding periods per year

        Returns:
        --------
        Dict with future value calculation
        """
        try:
            periodic_rate = rate / compounding_periods
            total_periods = compounding_periods * time

            fv = present_value * (1 + periodic_rate) ** total_periods

            return {
                'present_value': round(present_value, self.config.precision),
                'interest_rate': round(rate, self.config.precision),
                'time': round(time, self.config.precision),
                'compounding_periods': compounding_periods,
                'future_value': round(fv, self.config.precision),
                'growth_factor': round(fv / present_value, self.config.precision)
            }
        except Exception as e:
            return {'error': str(e)}


class BondPricing:
    """
    Bond pricing, duration, and convexity calculations.
    """

    def __init__(self, config: PricingConfig = None):
        self.config = config or PricingConfig()

    def bond_price(self, face_value: float, coupon_rate: float, yield_rate: float,
                  time_to_maturity: float, coupon_frequency: int = 2) -> Dict[str, Any]:
        """
        Calculate bond price using present value of cash flows.

        Parameters:
        -----------
        face_value : float
            Face value of the bond
        coupon_rate : float
            Annual coupon rate (as decimal)
        yield_rate : float
            Yield to maturity (as decimal)
        time_to_maturity : float
            Time to maturity in years
        coupon_frequency : int
            Number of coupon payments per year

        Returns:
        --------
        Dict with bond pricing details
        """
        try:
            # Calculate periodic values
            periods = int(time_to_maturity * coupon_frequency)
            periodic_yield = yield_rate / coupon_frequency
            periodic_coupon = (coupon_rate * face_value) / coupon_frequency

            # Present value of coupon payments
            if periodic_yield == 0:
                pv_coupons = periodic_coupon * periods
            else:
                pv_coupons = periodic_coupon * (1 - (1 + periodic_yield) ** (-periods)) / periodic_yield

            # Present value of face value
            pv_face_value = face_value / (1 + periodic_yield) ** periods

            # Total bond price
            bond_price = pv_coupons + pv_face_value

            # Additional metrics
            current_yield = (coupon_rate * face_value) / bond_price if bond_price > 0 else 0

            return {
                'face_value': round(face_value, self.config.precision),
                'coupon_rate': round(coupon_rate, self.config.precision),
                'yield_to_maturity': round(yield_rate, self.config.precision),
                'time_to_maturity': round(time_to_maturity, self.config.precision),
                'coupon_frequency': coupon_frequency,
                'periods': periods,
                'periodic_yield': round(periodic_yield, self.config.precision),
                'periodic_coupon': round(periodic_coupon, self.config.precision),
                'pv_coupons': round(pv_coupons, self.config.precision),
                'pv_face_value': round(pv_face_value, self.config.precision),
                'bond_price': round(bond_price, self.config.precision),
                'current_yield': round(current_yield, self.config.precision),
                'price_per_100': round((bond_price / face_value) * 100, self.config.precision)
            }
        except Exception as e:
            return {'error': str(e)}

    def bond_duration(self, face_value: float, coupon_rate: float, yield_rate: float,
                     time_to_maturity: float, coupon_frequency: int = 2) -> Dict[str, Any]:
        """
        Calculate Macaulay and Modified duration.

        Parameters:
        -----------
        face_value : float
            Face value of the bond
        coupon_rate : float
            Annual coupon rate (as decimal)
        yield_rate : float
            Yield to maturity (as decimal)
        time_to_maturity : float
            Time to maturity in years
        coupon_frequency : int
            Number of coupon payments per year

        Returns:
        --------
        Dict with duration calculations
        """
        try:
            # Get bond price first
            bond_info = self.bond_price(face_value, coupon_rate, yield_rate, time_to_maturity, coupon_frequency)
            if 'error' in bond_info:
                return bond_info

            bond_price = bond_info['bond_price']
            periods = bond_info['periods']
            periodic_yield = bond_info['periodic_yield']
            periodic_coupon = bond_info['periodic_coupon']

            # Calculate weighted average time to cash flows
            weighted_time = 0
            for t in range(1, periods + 1):
                if t < periods:
                    cash_flow = periodic_coupon
                else:
                    cash_flow = periodic_coupon + face_value

                present_value = cash_flow / (1 + periodic_yield) ** t
                time_weight = (t / coupon_frequency) * present_value
                weighted_time += time_weight

            # Macaulay duration
            macaulay_duration = weighted_time / bond_price

            # Modified duration
            modified_duration = macaulay_duration / (1 + periodic_yield)

            return {
                'bond_price': round(bond_price, self.config.precision),
                'macaulay_duration': round(macaulay_duration, self.config.precision),
                'modified_duration': round(modified_duration, self.config.precision),
                'duration_years': round(macaulay_duration, self.config.precision),
                'price_sensitivity': round(-modified_duration, self.config.precision)
            }
        except Exception as e:
            return {'error': str(e)}

    def bond_convexity(self, face_value: float, coupon_rate: float, yield_rate: float,
                      time_to_maturity: float, coupon_frequency: int = 2) -> Dict[str, Any]:
        """
        Calculate bond convexity.

        Parameters:
        -----------
        face_value : float
            Face value of the bond
        coupon_rate : float
            Annual coupon rate (as decimal)
        yield_rate : float
            Yield to maturity (as decimal)
        time_to_maturity : float
            Time to maturity in years
        coupon_frequency : int
            Number of coupon payments per year

        Returns:
        --------
        Dict with convexity calculations
        """
        try:
            # Get bond price
            bond_info = self.bond_price(face_value, coupon_rate, yield_rate, time_to_maturity, coupon_frequency)
            if 'error' in bond_info:
                return bond_info

            bond_price = bond_info['bond_price']
            periods = bond_info['periods']
            periodic_yield = bond_info['periodic_yield']
            periodic_coupon = bond_info['periodic_coupon']

            # Calculate convexity
            convexity_sum = 0
            for t in range(1, periods + 1):
                if t < periods:
                    cash_flow = periodic_coupon
                else:
                    cash_flow = periodic_coupon + face_value

                present_value = cash_flow / (1 + periodic_yield) ** t
                convexity_term = (t * (t + 1)) * present_value
                convexity_sum += convexity_term

            convexity = convexity_sum / (bond_price * (1 + periodic_yield) ** 2 * coupon_frequency ** 2)

            return {
                'bond_price': round(bond_price, self.config.precision),
                'convexity': round(convexity, self.config.precision),
                'convexity_adjustment': round(0.5 * convexity, self.config.precision)
            }
        except Exception as e:
            return {'error': str(e)}

    def bond_yield_to_maturity(self, bond_price: float, face_value: float, coupon_rate: float,
                              time_to_maturity: float, coupon_frequency: int = 2) -> Dict[str, Any]:
        """
        Calculate yield to maturity given bond price (using numerical methods).

        Parameters:
        -----------
        bond_price : float
            Current market price of bond
        face_value : float
            Face value of the bond
        coupon_rate : float
            Annual coupon rate (as decimal)
        time_to_maturity : float
            Time to maturity in years
        coupon_frequency : int
            Number of coupon payments per year

        Returns:
        --------
        Dict with YTM calculation
        """
        try:
            def price_difference(ytm):
                """Function to minimize: difference between calculated and market price."""
                calc_price = self.bond_price(face_value, coupon_rate, ytm, time_to_maturity, coupon_frequency)
                if 'error' in calc_price:
                    return float('inf')
                return abs(calc_price['bond_price'] - bond_price)

            # Use optimization to find YTM
            result = minimize_scalar(price_difference, bounds=(0, 1), method='bounded')

            if result.success:
                ytm = result.x

                # Verify the result
                verification = self.bond_price(face_value, coupon_rate, ytm, time_to_maturity, coupon_frequency)

                return {
                    'market_price': round(bond_price, self.config.precision),
                    'yield_to_maturity': round(ytm, self.config.precision),
                    'ytm_percent': round(ytm * 100, self.config.precision),
                    'calculated_price': round(verification['bond_price'], self.config.precision),
                    'price_difference': round(abs(verification['bond_price'] - bond_price), self.config.precision),
                    'optimization_success': True
                }
            else:
                return {'error': 'YTM optimization failed to converge'}

        except Exception as e:
            return {'error': str(e)}

    def plot_duration_convexity(self, face_value: float, coupon_rate: float, yield_rate: float,
                               time_to_maturity: float, yield_range: float = 0.05) -> Dict[str, Any]:
        """
        Plot bond price sensitivity (duration and convexity effects).

        Parameters:
        -----------
        face_value : float
            Face value of the bond
        coupon_rate : float
            Annual coupon rate (as decimal)
        yield_rate : float
            Current yield to maturity (as decimal)
        time_to_maturity : float
            Time to maturity in years
        yield_range : float
            Range of yields to plot around current yield

        Returns:
        --------
        Dict with plot information and data
        """
        try:
            # Get baseline metrics
            base_price_info = self.bond_price(face_value, coupon_rate, yield_rate, time_to_maturity)
            duration_info = self.bond_duration(face_value, coupon_rate, yield_rate, time_to_maturity)
            convexity_info = self.bond_convexity(face_value, coupon_rate, yield_rate, time_to_maturity)

            if any('error' in info for info in [base_price_info, duration_info, convexity_info]):
                return {'error': 'Failed to calculate baseline metrics'}

            base_price = base_price_info['bond_price']
            modified_duration = duration_info['modified_duration']
            convexity = convexity_info['convexity']

            # Create yield range
            yields = np.linspace(yield_rate - yield_range, yield_rate + yield_range, 100)
            actual_prices = []
            duration_prices = []
            duration_convexity_prices = []

            for y in yields:
                # Actual price
                price_info = self.bond_price(face_value, coupon_rate, y, time_to_maturity)
                actual_prices.append(price_info['bond_price'])

                # Duration approximation
                yield_change = y - yield_rate
                duration_price = base_price * (1 - modified_duration * yield_change)
                duration_prices.append(duration_price)

                # Duration + Convexity approximation
                convexity_price = base_price * (1 - modified_duration * yield_change +
                                              0.5 * convexity * yield_change ** 2)
                duration_convexity_prices.append(convexity_price)

            # Create plot
            plt.style.use(self.config.plot_style)
            plt.figure(figsize=self.config.figure_size)

            plt.plot(yields * 100, actual_prices, 'b-', linewidth=2, label='Actual Price')
            plt.plot(yields * 100, duration_prices, 'r--', linewidth=2, label='Duration Approximation')
            plt.plot(yields * 100, duration_convexity_prices, 'g:', linewidth=2, label='Duration + Convexity')

            plt.axvline(yield_rate * 100, color='black', linestyle=':', alpha=0.7, label='Current Yield')
            plt.axhline(base_price, color='black', linestyle=':', alpha=0.7)

            plt.xlabel('Yield to Maturity (%)')
            plt.ylabel('Bond Price')
            plt.title(f'Bond Price Sensitivity Analysis\nDuration: {modified_duration:.2f}, Convexity: {convexity:.2f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            return {
                'base_price': round(base_price, self.config.precision),
                'modified_duration': round(modified_duration, self.config.precision),
                'convexity': round(convexity, self.config.precision),
                'yield_range': yields.tolist(),
                'actual_prices': [round(p, self.config.precision) for p in actual_prices],
                'duration_prices': [round(p, self.config.precision) for p in duration_prices],
                'duration_convexity_prices': [round(p, self.config.precision) for p in duration_convexity_prices],
                'plot_created': True
            }
        except Exception as e:
            return {'error': str(e)}


class OptionPayoffs:
    """
    Option payoffs and P&L calculations for various option types.
    """

    def __init__(self, config: PricingConfig = None):
        self.config = config or PricingConfig()

    def european_call_payoff(self, spot_prices: Union[float, np.ndarray], strike: float) -> Union[float, np.ndarray]:
        """Calculate European call option payoff."""
        return np.maximum(spot_prices - strike, 0)

    def european_put_payoff(self, spot_prices: Union[float, np.ndarray], strike: float) -> Union[float, np.ndarray]:
        """Calculate European put option payoff."""
        return np.maximum(strike - spot_prices, 0)

    def digital_call_payoff(self, spot_prices: Union[float, np.ndarray], strike: float,
                           payout: float = 1.0) -> Union[float, np.ndarray]:
        """Calculate digital (binary) call option payoff."""
        return payout * (spot_prices >= strike).astype(float)

    def digital_put_payoff(self, spot_prices: Union[float, np.ndarray], strike: float,
                          payout: float = 1.0) -> Union[float, np.ndarray]:
        """Calculate digital (binary) put option payoff."""
        return payout * (spot_prices <= strike).astype(float)

    def option_pnl(self, payoffs: Union[float, np.ndarray], premium: float,
                   position: str = 'long') -> Dict[str, Any]:
        """
        Calculate option P&L given payoffs and premium.

        Parameters:
        -----------
        payoffs : float or array
            Option payoffs at expiration
        premium : float
            Option premium paid/received
        position : str
            'long' or 'short' position

        Returns:
        --------
        Dict with P&L analysis
        """
        try:
            payoffs = np.asarray(payoffs)

            if position.lower() == 'long':
                pnl = payoffs - premium
                max_loss = -premium
                breakeven = None
            elif position.lower() == 'short':
                pnl = premium - payoffs
                max_loss = None  # Potentially unlimited for short calls
                breakeven = None
            else:
                return {'error': 'Position must be "long" or "short"'}

            if np.isscalar(pnl):
                return {
                    'payoff': round(float(payoffs), self.config.precision),
                    'premium': round(premium, self.config.precision),
                    'pnl': round(float(pnl), self.config.precision),
                    'position': position,
                    'max_loss': max_loss
                }
            else:
                return {
                    'payoffs': np.round(payoffs, self.config.precision).tolist(),
                    'premium': round(premium, self.config.precision),
                    'pnl': np.round(pnl, self.config.precision).tolist(),
                    'position': position,
                    'max_loss': max_loss,
                    'max_profit': round(float(np.max(pnl)), self.config.precision),
                    'min_profit': round(float(np.min(pnl)), self.config.precision)
                }
        except Exception as e:
            return {'error': str(e)}

    def plot_option_payoff(self, option_type: str, strike: float, premium: float = 0,
                          spot_range: Tuple[float, float] = None, position: str = 'long') -> Dict[str, Any]:
        """
        Plot option payoff diagram.

        Parameters:
        -----------
        option_type : str
            'call', 'put', 'digital_call', 'digital_put'
        strike : float
            Strike price
        premium : float
            Option premium (for P&L calculation)
        spot_range : tuple
            (min_spot, max_spot) for plotting
        position : str
            'long' or 'short'

        Returns:
        --------
        Dict with plot data and analysis
        """
        try:
            # Set default spot range
            if spot_range is None:
                spot_range = (strike * 0.7, strike * 1.3)

            spot_prices = np.linspace(spot_range[0], spot_range[1], 100)

            # Calculate payoffs
            if option_type.lower() == 'call':
                payoffs = self.european_call_payoff(spot_prices, strike)
            elif option_type.lower() == 'put':
                payoffs = self.european_put_payoff(spot_prices, strike)
            elif option_type.lower() == 'digital_call':
                payoffs = self.digital_call_payoff(spot_prices, strike)
            elif option_type.lower() == 'digital_put':
                payoffs = self.digital_put_payoff(spot_prices, strike)
            else:
                return {'error': 'Invalid option type'}

            # Calculate P&L
            pnl_result = self.option_pnl(payoffs, premium, position)
            if 'error' in pnl_result:
                return pnl_result

            pnl = np.array(pnl_result['pnl'])

            # Create plot
            plt.style.use(self.config.plot_style)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figure_size)

            # Payoff plot
            ax1.plot(spot_prices, payoffs, 'b-', linewidth=2, label='Payoff')
            ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax1.axvline(strike, color='red', linestyle='--', alpha=0.7, label=f'Strike: {strike}')
            ax1.set_xlabel('Spot Price')
            ax1.set_ylabel('Payoff')
            ax1.set_title(f'{option_type.title()} Option Payoff ({position.title()} Position)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # P&L plot
            ax2.plot(spot_prices, pnl, 'g-', linewidth=2, label='P&L')
            ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax2.axvline(strike, color='red', linestyle='--', alpha=0.7, label=f'Strike: {strike}')
            if premium > 0:
                ax2.axhline(-premium if position == 'long' else premium, color='orange',
                           linestyle=':', alpha=0.7, label=f'Max Loss: {-premium if position == "long" else premium}')
            ax2.set_xlabel('Spot Price')
            ax2.set_ylabel('Profit/Loss')
            ax2.set_title(f'{option_type.title()} Option P&L (Premium: {premium})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            return {
                'option_type': option_type,
                'strike': strike,
                'premium': premium,
                'position': position,
                'spot_prices': spot_prices.tolist(),
                'payoffs': payoffs.tolist(),
                'pnl': pnl.tolist(),
                'max_profit': float(np.max(pnl)),
                'max_loss': float(np.min(pnl)),
                'breakeven_points': self._find_breakeven_points(spot_prices, pnl),
                'plot_created': True
            }
        except Exception as e:
            return {'error': str(e)}

    def _find_breakeven_points(self, spot_prices: np.ndarray, pnl: np.ndarray) -> List[float]:
        """Find breakeven points where P&L crosses zero."""
        breakeven_points = []
        for i in range(len(pnl) - 1):
            if pnl[i] * pnl[i + 1] < 0:  # Sign change indicates zero crossing
                # Linear interpolation to find exact breakeven
                breakeven = spot_prices[i] + (spot_prices[i + 1] - spot_prices[i]) * (-pnl[i] / (pnl[i + 1] - pnl[i]))
                breakeven_points.append(round(breakeven, self.config.precision))
        return breakeven_points


class OptionPricingModels:
    """
    Option pricing models: Black-Scholes, Binomial Tree, Monte Carlo, etc.
    """

    def __init__(self, config: PricingConfig = None):
        self.config = config or PricingConfig()

    def black_scholes(self, spot: float, strike: float, time_to_expiry: float,
                     risk_free_rate: float, volatility: float, option_type: str = 'call',
                     dividend_yield: float = 0.0) -> Dict[str, Any]:
        """
        Black-Scholes option pricing model.

        Parameters:
        -----------
        spot : float
            Current stock price
        strike : float
            Strike price
        time_to_expiry : float
            Time to expiration in years
        risk_free_rate : float
            Risk-free interest rate (annual)
        volatility : float
            Volatility (annual)
        option_type : str
            'call' or 'put'
        dividend_yield : float
            Continuous dividend yield

        Returns:
        --------
        Dict with option price and parameters
        """
        try:
            if time_to_expiry <= 0:
                # At expiration
                if option_type.lower() == 'call':
                    return {'option_price': round(max(spot - strike, 0), self.config.precision)}
                else:
                    return {'option_price': round(max(strike - spot, 0), self.config.precision)}

            # Calculate d1 and d2
            d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
            d2 = d1 - volatility * np.sqrt(time_to_expiry)

            # Standard normal CDF
            N_d1 = stats.norm.cdf(d1)
            N_d2 = stats.norm.cdf(d2)
            N_minus_d1 = stats.norm.cdf(-d1)
            N_minus_d2 = stats.norm.cdf(-d2)

            # Discount factors
            discount_factor = np.exp(-risk_free_rate * time_to_expiry)
            dividend_discount = np.exp(-dividend_yield * time_to_expiry)

            if option_type.lower() == 'call':
                option_price = spot * dividend_discount * N_d1 - strike * discount_factor * N_d2
            elif option_type.lower() == 'put':
                option_price = strike * discount_factor * N_minus_d2 - spot * dividend_discount * N_minus_d1
            else:
                return {'error': 'Option type must be "call" or "put"'}

            return {
                'option_price': round(option_price, self.config.precision),
                'spot': round(spot, self.config.precision),
                'strike': round(strike, self.config.precision),
                'time_to_expiry': round(time_to_expiry, self.config.precision),
                'risk_free_rate': round(risk_free_rate, self.config.precision),
                'volatility': round(volatility, self.config.precision),
                'dividend_yield': round(dividend_yield, self.config.precision),
                'option_type': option_type,
                'd1': round(d1, self.config.precision),
                'd2': round(d2, self.config.precision),
                'N_d1': round(N_d1, self.config.precision),
                'N_d2': round(N_d2, self.config.precision),
                'model': 'black_scholes'
            }
        except Exception as e:
            return {'error': str(e)}

    def black_model(self, forward_price: float, strike: float, time_to_expiry: float,
                   risk_free_rate: float, volatility: float, option_type: str = 'call') -> Dict[str, Any]:
        """
        Black model for options on forwards/futures.

        Parameters:
        -----------
        forward_price : float
            Forward price of underlying
        strike : float
            Strike price
        time_to_expiry : float
            Time to expiration in years
        risk_free_rate : float
            Risk-free interest rate
        volatility : float
            Volatility of forward price
        option_type : str
            'call' or 'put'

        Returns:
        --------
        Dict with option price and parameters
        """
        try:
            if time_to_expiry <= 0:
                if option_type.lower() == 'call':
                    return {'option_price': round(max(forward_price - strike, 0), self.config.precision)}
                else:
                    return {'option_price': round(max(strike - forward_price, 0), self.config.precision)}

            # Calculate d1 and d2
            d1 = (np.log(forward_price / strike) + 0.5 * volatility ** 2 * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
            d2 = d1 - volatility * np.sqrt(time_to_expiry)

            # Standard normal CDF
            N_d1 = stats.norm.cdf(d1)
            N_d2 = stats.norm.cdf(d2)
            N_minus_d1 = stats.norm.cdf(-d1)
            N_minus_d2 = stats.norm.cdf(-d2)

            # Discount factor
            discount_factor = np.exp(-risk_free_rate * time_to_expiry)

            if option_type.lower() == 'call':
                option_price = discount_factor * (forward_price * N_d1 - strike * N_d2)
            elif option_type.lower() == 'put':
                option_price = discount_factor * (strike * N_minus_d2 - forward_price * N_minus_d1)
            else:
                return {'error': 'Option type must be "call" or "put"'}

            return {
                'option_price': round(option_price, self.config.precision),
                'forward_price': round(forward_price, self.config.precision),
                'strike': round(strike, self.config.precision),
                'time_to_expiry': round(time_to_expiry, self.config.precision),
                'risk_free_rate': round(risk_free_rate, self.config.precision),
                'volatility': round(volatility, self.config.precision),
                'option_type': option_type,
                'd1': round(d1, self.config.precision),
                'd2': round(d2, self.config.precision),
                'model': 'black_model'
            }
        except Exception as e:
            return {'error': str(e)}

    def binomial_tree(self, spot: float, strike: float, time_to_expiry: float,
                     risk_free_rate: float, volatility: float, option_type: str = 'call',
                     option_style: str = 'european', dividend_yield: float = 0.0,
                     steps: int = None) -> Dict[str, Any]:
        """
        Binomial tree option pricing model.

        Parameters:
        -----------
        spot : float
            Current stock price
        strike : float
            Strike price
        time_to_expiry : float
            Time to expiration in years
        risk_free_rate : float
            Risk-free interest rate
        volatility : float
            Volatility
        option_type : str
            'call' or 'put'
        option_style : str
            'european' or 'american'
        dividend_yield : float
            Continuous dividend yield
        steps : int
            Number of time steps

        Returns:
        --------
        Dict with option price and tree information
        """
        try:
            if steps is None:
                steps = self.config.tree_steps

            if time_to_expiry <= 0:
                if option_type.lower() == 'call':
                    return {'option_price': round(max(spot - strike, 0), self.config.precision)}
                else:
                    return {'option_price': round(max(strike - spot, 0), self.config.precision)}

            # Tree parameters
            dt = time_to_expiry / steps
            u = np.exp(volatility * np.sqrt(dt))  # Up factor
            d = 1 / u  # Down factor
            p = (np.exp((risk_free_rate - dividend_yield) * dt) - d) / (u - d)  # Risk-neutral probability
            discount = np.exp(-risk_free_rate * dt)

            # Initialize asset price tree
            asset_prices = np.zeros((steps + 1, steps + 1))
            for i in range(steps + 1):
                for j in range(i + 1):
                    asset_prices[j, i] = spot * (u ** (i - j)) * (d ** j)

            # Initialize option values at expiration
            option_values = np.zeros((steps + 1, steps + 1))
            for j in range(steps + 1):
                if option_type.lower() == 'call':
                    option_values[j, steps] = max(asset_prices[j, steps] - strike, 0)
                else:
                    option_values[j, steps] = max(strike - asset_prices[j, steps], 0)

            # Backward induction
            for i in range(steps - 1, -1, -1):
                for j in range(i + 1):
                    # European option value
                    option_values[j, i] = discount * (p * option_values[j, i + 1] + (1 - p) * option_values[j + 1, i + 1])

                    # American option: check early exercise
                    if option_style.lower() == 'american':
                        if option_type.lower() == 'call':
                            intrinsic_value = max(asset_prices[j, i] - strike, 0)
                        else:
                            intrinsic_value = max(strike - asset_prices[j, i], 0)
                        option_values[j, i] = max(option_values[j, i], intrinsic_value)

            return {
                'option_price': round(option_values[0, 0], self.config.precision),
                'spot': round(spot, self.config.precision),
                'strike': round(strike, self.config.precision),
                'time_to_expiry': round(time_to_expiry, self.config.precision),
                'risk_free_rate': round(risk_free_rate, self.config.precision),
                'volatility': round(volatility, self.config.precision),
                'dividend_yield': round(dividend_yield, self.config.precision),
                'option_type': option_type,
                'option_style': option_style,
                'steps': steps,
                'up_factor': round(u, self.config.precision),
                'down_factor': round(d, self.config.precision),
                'risk_neutral_prob': round(p, self.config.precision),
                'dt': round(dt, self.config.precision),
                'model': 'binomial_tree'
            }
        except Exception as e:
            return {'error': str(e)}

    def monte_carlo(self, spot: float, strike: float, time_to_expiry: float,
                   risk_free_rate: float, volatility: float, option_type: str = 'call',
                   dividend_yield: float = 0.0, simulations: int = None,
                   option_style: str = 'european', monitoring_points: int = 50) -> Dict[str, Any]:
        """
        Monte Carlo option pricing.

        Parameters:
        -----------
        spot : float
            Current stock price
        strike : float
            Strike price
        time_to_expiry : float
            Time to expiration in years
        risk_free_rate : float
            Risk-free interest rate
        volatility : float
            Volatility
        option_type : str
            'call' or 'put'
        dividend_yield : float
            Continuous dividend yield
        simulations : int
            Number of Monte Carlo simulations
        option_style : str
            'european' or 'american' (approximation)
        monitoring_points : int
            Number of monitoring points for American options

        Returns:
        --------
        Dict with option price and simulation statistics
        """
        try:
            if simulations is None:
                simulations = self.config.mc_simulations

            np.random.seed(self.config.random_seed)

            if time_to_expiry <= 0:
                if option_type.lower() == 'call':
                    return {'option_price': round(max(spot - strike, 0), self.config.precision)}
                else:
                    return {'option_price': round(max(strike - spot, 0), self.config.precision)}

            # Generate random paths
            dt = time_to_expiry / monitoring_points
            drift = (risk_free_rate - dividend_yield - 0.5 * volatility ** 2) * dt
            diffusion = volatility * np.sqrt(dt)

            # Generate all random numbers at once for efficiency
            random_shocks = np.random.normal(0, 1, (simulations, monitoring_points))

            payoffs = np.zeros(simulations)

            for i in range(simulations):
                # Generate price path
                log_returns = drift + diffusion * random_shocks[i]
                price_path = spot * np.exp(np.cumsum(log_returns))
                price_path = np.concatenate(([spot], price_path))

                if option_style.lower() == 'european':
                    # European option: only check at expiration
                    final_price = price_path[-1]
                    if option_type.lower() == 'call':
                        payoffs[i] = max(final_price - strike, 0)
                    else:
                        payoffs[i] = max(strike - final_price, 0)

                else:  # American option approximation
                    # Check all monitoring points for early exercise
                    max_payoff = 0
                    for j, price in enumerate(price_path):
                        if option_type.lower() == 'call':
                            intrinsic = max(price - strike, 0)
                        else:
                            intrinsic = max(strike - price, 0)

                        # Approximate continuation value (simplified)
                        remaining_time = time_to_expiry * (monitoring_points - j) / monitoring_points
                        if remaining_time > 0:
                            # Use simple heuristic for continuation value
                            time_value = intrinsic * 0.1 * remaining_time  # Simplified approximation
                            total_value = intrinsic + time_value
                        else:
                            total_value = intrinsic

                        max_payoff = max(max_payoff, total_value)

                    payoffs[i] = max_payoff

            # Calculate discounted expectation
            discount_factor = np.exp(-risk_free_rate * time_to_expiry)
            option_price = discount_factor * np.mean(payoffs)

            # Calculate statistics
            payoff_std = np.std(payoffs)
            standard_error = payoff_std / np.sqrt(simulations) * discount_factor
            confidence_interval = 1.96 * standard_error  # 95% CI

            return {
                'option_price': round(option_price, self.config.precision),
                'standard_error': round(standard_error, self.config.precision),
                'confidence_interval': round(confidence_interval, self.config.precision),
                'confidence_lower': round(option_price - confidence_interval, self.config.precision),
                'confidence_upper': round(option_price + confidence_interval, self.config.precision),
                'spot': round(spot, self.config.precision),
                'strike': round(strike, self.config.precision),
                'time_to_expiry': round(time_to_expiry, self.config.precision),
                'risk_free_rate': round(risk_free_rate, self.config.precision),
                'volatility': round(volatility, self.config.precision),
                'dividend_yield': round(dividend_yield, self.config.precision),
                'option_type': option_type,
                'option_style': option_style,
                'simulations': simulations,
                'monitoring_points': monitoring_points,
                'mean_payoff': round(np.mean(payoffs), self.config.precision),
                'payoff_std': round(payoff_std, self.config.precision),
                'model': 'monte_carlo'
            }
        except Exception as e:
            return {'error': str(e)}

    def implied_volatility(self, market_price: float, spot: float, strike: float,
                          time_to_expiry: float, risk_free_rate: float,
                          option_type: str = 'call', dividend_yield: float = 0.0) -> Dict[str, Any]:
        """
        Calculate implied volatility using Newton-Raphson method.

        Parameters:
        -----------
        market_price : float
            Observed market price of option
        spot : float
            Current stock price
        strike : float
            Strike price
        time_to_expiry : float
            Time to expiration in years
        risk_free_rate : float
            Risk-free interest rate
        option_type : str
            'call' or 'put'
        dividend_yield : float
            Continuous dividend yield

        Returns:
        --------
        Dict with implied volatility and calculation details
        """
        try:
            def objective_function(vol):
                """Function to find root: market_price - theoretical_price = 0"""
                bs_result = self.black_scholes(spot, strike, time_to_expiry, risk_free_rate, vol, option_type, dividend_yield)
                if 'error' in bs_result:
                    return float('inf')
                return bs_result['option_price'] - market_price

            def vega_function(vol):
                """Calculate vega for Newton-Raphson method"""
                # Calculate vega analytically
                if time_to_expiry <= 0:
                    return 0

                d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * vol ** 2) * time_to_expiry) / (vol * np.sqrt(time_to_expiry))
                vega = spot * np.exp(-dividend_yield * time_to_expiry) * stats.norm.pdf(d1) * np.sqrt(time_to_expiry)
                return vega

            # Initial guess
            vol_guess = 0.2

            # Newton-Raphson iteration
            for i in range(self.config.max_iterations):
                price_diff = objective_function(vol_guess)

                if abs(price_diff) < self.config.tolerance:
                    break

                vega = vega_function(vol_guess)
                if abs(vega) < 1e-10:
                    return {'error': 'Vega too small, cannot compute implied volatility'}

                vol_guess = vol_guess - price_diff / vega

                if vol_guess < 0:
                    vol_guess = 0.001

            # Verify convergence
            final_price = self.black_scholes(spot, strike, time_to_expiry, risk_free_rate, vol_guess, option_type, dividend_yield)
            price_error = abs(final_price['option_price'] - market_price)

            return {
                'implied_volatility': round(vol_guess, self.config.precision),
                'implied_volatility_percent': round(vol_guess * 100, self.config.precision),
                'market_price': round(market_price, self.config.precision),
                'theoretical_price': round(final_price['option_price'], self.config.precision),
                'price_error': round(price_error, self.config.precision),
                'iterations': i + 1,
                'converged': price_error < self.config.tolerance,
                'spot': round(spot, self.config.precision),
                'strike': round(strike, self.config.precision),
                'time_to_expiry': round(time_to_expiry, self.config.precision),
                'risk_free_rate': round(risk_free_rate, self.config.precision),
                'option_type': option_type
            }
        except Exception as e:
            return {'error': str(e)}


class OptionGreeks:
    """
    Option Greeks calculations: Delta, Gamma, Theta, Vega, Rho.
    """

    def __init__(self, config: PricingConfig = None):
        self.config = config or PricingConfig()
        self.pricing_models = OptionPricingModels(config)

    def delta(self, spot: float, strike: float, time_to_expiry: float,
             risk_free_rate: float, volatility: float, option_type: str = 'call',
             dividend_yield: float = 0.0, method: str = 'analytical') -> Dict[str, Any]:
        """
        Calculate option delta.

        Parameters:
        -----------
        spot : float
            Current stock price
        strike : float
            Strike price
        time_to_expiry : float
            Time to expiration in years
        risk_free_rate : float
            Risk-free interest rate
        volatility : float
            Volatility
        option_type : str
            'call' or 'put'
        dividend_yield : float
            Continuous dividend yield
        method : str
            'analytical' or 'numerical'

        Returns:
        --------
        Dict with delta calculation
        """
        try:
            if method == 'analytical':
                if time_to_expiry <= 0:
                    if option_type.lower() == 'call':
                        delta = 1.0 if spot > strike else 0.0
                    else:
                        delta = -1.0 if spot < strike else 0.0
                else:
                    d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))

                    if option_type.lower() == 'call':
                        delta = np.exp(-dividend_yield * time_to_expiry) * stats.norm.cdf(d1)
                    else:
                        delta = -np.exp(-dividend_yield * time_to_expiry) * stats.norm.cdf(-d1)

            else:  # numerical method
                h = self.config.finite_difference_h
                price_up = self.pricing_models.black_scholes(spot + h, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield)
                price_down = self.pricing_models.black_scholes(spot - h, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield)

                if 'error' in price_up or 'error' in price_down:
                    return {'error': 'Failed to calculate numerical delta'}

                delta = (price_up['option_price'] - price_down['option_price']) / (2 * h)

            return {
                'delta': round(delta, self.config.precision),
                'spot': round(spot, self.config.precision),
                'strike': round(strike, self.config.precision),
                'time_to_expiry': round(time_to_expiry, self.config.precision),
                'option_type': option_type,
                'method': method,
                'greek': 'delta'
            }
        except Exception as e:
            return {'error': str(e)}

    def gamma(self, spot: float, strike: float, time_to_expiry: float,
             risk_free_rate: float, volatility: float, option_type: str = 'call',
             dividend_yield: float = 0.0, method: str = 'analytical') -> Dict[str, Any]:
        """
        Calculate option gamma.

        Parameters:
        -----------
        Similar to delta method

        Returns:
        --------
        Dict with gamma calculation
        """
        try:
            if method == 'analytical':
                if time_to_expiry <= 0:
                    gamma = 0.0
                else:
                    d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
                    gamma = (np.exp(-dividend_yield * time_to_expiry) * stats.norm.pdf(d1)) / (spot * volatility * np.sqrt(time_to_expiry))

            else:  # numerical method
                h = self.config.finite_difference_h
                delta_up = self.delta(spot + h, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield, 'analytical')
                delta_down = self.delta(spot - h, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield, 'analytical')

                if 'error' in delta_up or 'error' in delta_down:
                    return {'error': 'Failed to calculate numerical gamma'}

                gamma = (delta_up['delta'] - delta_down['delta']) / (2 * h)

            return {
                'gamma': round(gamma, self.config.precision),
                'spot': round(spot, self.config.precision),
                'strike': round(strike, self.config.precision),
                'time_to_expiry': round(time_to_expiry, self.config.precision),
                'option_type': option_type,
                'method': method,
                'greek': 'gamma'
            }
        except Exception as e:
            return {'error': str(e)}

    def theta(self, spot: float, strike: float, time_to_expiry: float,
             risk_free_rate: float, volatility: float, option_type: str = 'call',
             dividend_yield: float = 0.0, method: str = 'analytical') -> Dict[str, Any]:
        """
        Calculate option theta (time decay).

        Parameters:
        -----------
        Similar to delta method

        Returns:
        --------
        Dict with theta calculation
        """
        try:
            if method == 'analytical':
                if time_to_expiry <= 0:
                    theta = 0.0
                else:
                    d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
                    d2 = d1 - volatility * np.sqrt(time_to_expiry)

                    term1 = -(spot * np.exp(-dividend_yield * time_to_expiry) * stats.norm.pdf(d1) * volatility) / (2 * np.sqrt(time_to_expiry))

                    if option_type.lower() == 'call':
                        term2 = risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * stats.norm.cdf(d2)
                        term3 = dividend_yield * spot * np.exp(-dividend_yield * time_to_expiry) * stats.norm.cdf(d1)
                        theta = term1 - term2 + term3
                    else:
                        term2 = risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * stats.norm.cdf(-d2)
                        term3 = dividend_yield * spot * np.exp(-dividend_yield * time_to_expiry) * stats.norm.cdf(-d1)
                        theta = term1 + term2 - term3

                    # Convert to per-day theta
                    theta = theta / 365

            else:  # numerical method
                h = 1/365  # One day
                if time_to_expiry <= h:
                    theta = 0.0
                else:
                    price_current = self.pricing_models.black_scholes(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield)
                    price_tomorrow = self.pricing_models.black_scholes(spot, strike, time_to_expiry - h, risk_free_rate, volatility, option_type, dividend_yield)

                    if 'error' in price_current or 'error' in price_tomorrow:
                        return {'error': 'Failed to calculate numerical theta'}

                    theta = price_tomorrow['option_price'] - price_current['option_price']

            return {
                'theta': round(theta, self.config.precision),
                'theta_per_day': round(theta, self.config.precision),
                'theta_per_year': round(theta * 365, self.config.precision),
                'spot': round(spot, self.config.precision),
                'strike': round(strike, self.config.precision),
                'time_to_expiry': round(time_to_expiry, self.config.precision),
                'option_type': option_type,
                'method': method,
                'greek': 'theta'
            }
        except Exception as e:
            return {'error': str(e)}

    def vega(self, spot: float, strike: float, time_to_expiry: float,
            risk_free_rate: float, volatility: float, option_type: str = 'call',
            dividend_yield: float = 0.0, method: str = 'analytical') -> Dict[str, Any]:
        """
        Calculate option vega.

        Parameters:
        -----------
        Similar to delta method

        Returns:
        --------
        Dict with vega calculation
        """
        try:
            if method == 'analytical':
                if time_to_expiry <= 0:
                    vega = 0.0
                else:
                    d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
                    vega = spot * np.exp(-dividend_yield * time_to_expiry) * stats.norm.pdf(d1) * np.sqrt(time_to_expiry)

                    # Convert to percentage point change (vega per 1% volatility change)
                    vega = vega / 100

            else:  # numerical method
                h = 0.01  # 1% volatility change
                price_up = self.pricing_models.black_scholes(spot, strike, time_to_expiry, risk_free_rate, volatility + h, option_type, dividend_yield)
                price_down = self.pricing_models.black_scholes(spot, strike, time_to_expiry, risk_free_rate, volatility - h, option_type, dividend_yield)

                if 'error' in price_up or 'error' in price_down:
                    return {'error': 'Failed to calculate numerical vega'}

                vega = (price_up['option_price'] - price_down['option_price']) / (2 * h * 100)

            return {
                'vega': round(vega, self.config.precision),
                'vega_per_percent': round(vega, self.config.precision),
                'vega_per_decimal': round(vega * 100, self.config.precision),
                'spot': round(spot, self.config.precision),
                'strike': round(strike, self.config.precision),
                'time_to_expiry': round(time_to_expiry, self.config.precision),
                'volatility': round(volatility, self.config.precision),
                'option_type': option_type,
                'method': method,
                'greek': 'vega'
            }
        except Exception as e:
            return {'error': str(e)}

    def rho(self, spot: float, strike: float, time_to_expiry: float,
           risk_free_rate: float, volatility: float, option_type: str = 'call',
           dividend_yield: float = 0.0, method: str = 'analytical') -> Dict[str, Any]:
        """
        Calculate option rho.

        Parameters:
        -----------
        Similar to delta method

        Returns:
        --------
        Dict with rho calculation
        """
        try:
            if method == 'analytical':
                if time_to_expiry <= 0:
                    rho = 0.0
                else:
                    d2 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry)) - volatility * np.sqrt(time_to_expiry)

                    if option_type.lower() == 'call':
                        rho = strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * stats.norm.cdf(d2)
                    else:
                        rho = -strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * stats.norm.cdf(-d2)

                    # Convert to percentage point change (rho per 1% rate change)
                    rho = rho / 100

            else:  # numerical method
                h = 0.01  # 1% rate change
                price_up = self.pricing_models.black_scholes(spot, strike, time_to_expiry, risk_free_rate + h, volatility, option_type, dividend_yield)
                price_down = self.pricing_models.black_scholes(spot, strike, time_to_expiry, risk_free_rate - h, volatility, option_type, dividend_yield)

                if 'error' in price_up or 'error' in price_down:
                    return {'error': 'Failed to calculate numerical rho'}

                rho = (price_up['option_price'] - price_down['option_price']) / (2 * h * 100)

            return {
                'rho': round(rho, self.config.precision),
                'rho_per_percent': round(rho, self.config.precision),
                'rho_per_decimal': round(rho * 100, self.config.precision),
                'spot': round(spot, self.config.precision),
                'strike': round(strike, self.config.precision),
                'time_to_expiry': round(time_to_expiry, self.config.precision),
                'risk_free_rate': round(risk_free_rate, self.config.precision),
                'option_type': option_type,
                'method': method,
                'greek': 'rho'
            }
        except Exception as e:
            return {'error': str(e)}

    def all_greeks(self, spot: float, strike: float, time_to_expiry: float,
                  risk_free_rate: float, volatility: float, option_type: str = 'call',
                  dividend_yield: float = 0.0, method: str = 'analytical') -> Dict[str, Any]:
        """
        Calculate all Greeks at once.

        Parameters:
        -----------
        Similar to individual Greek methods

        Returns:
        --------
        Dict with all Greeks
        """
        try:
            delta_result = self.delta(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield, method)
            gamma_result = self.gamma(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield, method)
            theta_result = self.theta(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield, method)
            vega_result = self.vega(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield, method)
            rho_result = self.rho(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield, method)

            # Get option price
            price_result = self.pricing_models.black_scholes(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield)

            return {
                'option_price': price_result.get('option_price', None),
                'delta': delta_result.get('delta', None),
                'gamma': gamma_result.get('gamma', None),
                'theta': theta_result.get('theta', None),
                'vega': vega_result.get('vega', None),
                'rho': rho_result.get('rho', None),
                'spot': round(spot, self.config.precision),
                'strike': round(strike, self.config.precision),
                'time_to_expiry': round(time_to_expiry, self.config.precision),
                'risk_free_rate': round(risk_free_rate, self.config.precision),
                'volatility': round(volatility, self.config.precision),
                'dividend_yield': round(dividend_yield, self.config.precision),
                'option_type': option_type,
                'method': method,
                'calculation_date': datetime.now()
            }
        except Exception as e:
            return {'error': str(e)}


class MortgageFinance:
    """
    Comprehensive mortgage and real estate finance calculations.
    """

    def __init__(self, config: PricingConfig = None):
        self.config = config or PricingConfig()

    def mortgage_payment(self, principal: float, annual_rate: float, years: int,
                        payment_frequency: int = 12) -> Dict[str, Any]:
        """
        Calculate mortgage payment and amortization details.

        Parameters:
        -----------
        principal : float
            Loan principal amount
        annual_rate : float
            Annual interest rate (as decimal)
        years : int
            Loan term in years
        payment_frequency : int
            Payments per year (12 for monthly)

        Returns:
        --------
        Dict with mortgage payment details
        """
        try:
            periodic_rate = annual_rate / payment_frequency
            total_payments = years * payment_frequency

            if periodic_rate == 0:
                monthly_payment = principal / total_payments
            else:
                monthly_payment = principal * (periodic_rate * (1 + periodic_rate) ** total_payments) / \
                                 ((1 + periodic_rate) ** total_payments - 1)

            total_interest = monthly_payment * total_payments - principal

            return {
                'principal': round(principal, self.config.precision),
                'annual_rate': round(annual_rate, self.config.precision),
                'periodic_rate': round(periodic_rate, self.config.precision),
                'years': years,
                'payment_frequency': payment_frequency,
                'total_payments': total_payments,
                'monthly_payment': round(monthly_payment, self.config.precision),
                'total_payments_amount': round(monthly_payment * total_payments, self.config.precision),
                'total_interest': round(total_interest, self.config.precision),
                'interest_percentage': round((total_interest / principal) * 100, self.config.precision)
            }
        except Exception as e:
            return {'error': str(e)}

    def ltv_dti_analysis(self, property_values: List[float], loan_amounts: List[float],
                        incomes: List[float], debt_payments: List[float]) -> Dict[str, Any]:
        """
        Analyze Loan-to-Value (LTV) and Debt-to-Income (DTI) ratios for market structure study.

        Parameters:
        -----------
        property_values : List[float]
            Property values
        loan_amounts : List[float]
            Loan amounts
        incomes : List[float]
            Borrower incomes
        debt_payments : List[float]
            Monthly debt payments

        Returns:
        --------
        Dict with LTV/DTI analysis and market structure insights
        """
        try:
            if not all(len(lst) == len(property_values) for lst in [loan_amounts, incomes, debt_payments]):
                return {'error': 'All input lists must have the same length'}

            # Calculate ratios
            ltv_ratios = [loan / value for loan, value in zip(loan_amounts, property_values)]
            dti_ratios = [debt / (income / 12) for debt, income in zip(debt_payments, incomes)]  # Monthly DTI

            ltv_array = np.array(ltv_ratios)
            dti_array = np.array(dti_ratios)

            # Basic statistics
            ltv_stats = {
                'mean': round(np.mean(ltv_array), self.config.precision),
                'median': round(np.median(ltv_array), self.config.precision),
                'std': round(np.std(ltv_array), self.config.precision),
                'min': round(np.min(ltv_array), self.config.precision),
                'max': round(np.max(ltv_array), self.config.precision),
                'q25': round(np.percentile(ltv_array, 25), self.config.precision),
                'q75': round(np.percentile(ltv_array, 75), self.config.precision)
            }

            dti_stats = {
                'mean': round(np.mean(dti_array), self.config.precision),
                'median': round(np.median(dti_array), self.config.precision),
                'std': round(np.std(dti_array), self.config.precision),
                'min': round(np.min(dti_array), self.config.precision),
                'max': round(np.max(dti_array), self.config.precision),
                'q25': round(np.percentile(dti_array, 25), self.config.precision),
                'q75': round(np.percentile(dti_array, 75), self.config.precision)
            }

            # Risk categorization
            high_ltv_threshold = 0.80
            high_dti_threshold = 0.43  # 43% DTI threshold

            high_ltv_count = np.sum(ltv_array > high_ltv_threshold)
            high_dti_count = np.sum(dti_array > high_dti_threshold)
            high_risk_count = np.sum((ltv_array > high_ltv_threshold) & (dti_array > high_dti_threshold))

            risk_analysis = {
                'high_ltv_count': int(high_ltv_count),
                'high_ltv_percentage': round((high_ltv_count / len(ltv_array)) * 100, self.config.precision),
                'high_dti_count': int(high_dti_count),
                'high_dti_percentage': round((high_dti_count / len(dti_array)) * 100, self.config.precision),
                'high_risk_count': int(high_risk_count),
                'high_risk_percentage': round((high_risk_count / len(ltv_array)) * 100, self.config.precision)
            }

            # Correlation analysis
            correlation = np.corrcoef(ltv_array, dti_array)[0, 1]

            return {
                'n_loans': len(property_values),
                'ltv_statistics': ltv_stats,
                'dti_statistics': dti_stats,
                'risk_analysis': risk_analysis,
                'ltv_dti_correlation': round(correlation, self.config.precision),
                'ltv_ratios': [round(ratio, self.config.precision) for ratio in ltv_ratios],
                'dti_ratios': [round(ratio, self.config.precision) for ratio in dti_ratios],
                'analysis_date': datetime.now()
            }
        except Exception as e:
            return {'error': str(e)}

    def plot_ltv_dti_market_structure(self, property_values: List[float], loan_amounts: List[float],
                                    incomes: List[float], debt_payments: List[float]) -> Dict[str, Any]:
        """
        Create visualizations for LTV and DTI market structure analysis.

        Parameters:
        -----------
        Same as ltv_dti_analysis

        Returns:
        --------
        Dict with plot information
        """
        try:
            # Get analysis data
            analysis = self.ltv_dti_analysis(property_values, loan_amounts, incomes, debt_payments)
            if 'error' in analysis:
                return analysis

            ltv_ratios = analysis['ltv_ratios']
            dti_ratios = analysis['dti_ratios']

            plt.style.use(self.config.plot_style)
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

            # LTV vs DTI Scatter Plot
            ax1.scatter(ltv_ratios, dti_ratios, alpha=0.6, c='blue', s=50)
            ax1.axvline(0.80, color='red', linestyle='--', alpha=0.7, label='80% LTV Threshold')
            ax1.axhline(0.43, color='red', linestyle='--', alpha=0.7, label='43% DTI Threshold')
            ax1.set_xlabel('Loan-to-Value Ratio')
            ax1.set_ylabel('Debt-to-Income Ratio')
            ax1.set_title('LTV vs DTI Market Structure')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # LTV Distribution
            ax2.hist(ltv_ratios, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax2.axvline(np.mean(ltv_ratios), color='red', linestyle='-', label=f'Mean: {np.mean(ltv_ratios):.3f}')
            ax2.axvline(0.80, color='orange', linestyle='--', label='80% Threshold')
            ax2.set_xlabel('Loan-to-Value Ratio')
            ax2.set_ylabel('Frequency')
            ax2.set_title('LTV Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # DTI Distribution
            ax3.hist(dti_ratios, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax3.axvline(np.mean(dti_ratios), color='red', linestyle='-', label=f'Mean: {np.mean(dti_ratios):.3f}')
            ax3.axvline(0.43, color='orange', linestyle='--', label='43% Threshold')
            ax3.set_xlabel('Debt-to-Income Ratio')
            ax3.set_ylabel('Frequency')
            ax3.set_title('DTI Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Risk Heat Map
            ltv_bins = np.linspace(0, 1, 11)
            dti_bins = np.linspace(0, 1, 11)

            hist, ltv_edges, dti_edges = np.histogram2d(ltv_ratios, dti_ratios, bins=[ltv_bins, dti_bins])

            im = ax4.imshow(hist.T, origin='lower', extent=[0, 1, 0, 1], cmap='YlOrRd', aspect='auto')
            ax4.set_xlabel('Loan-to-Value Ratio')
            ax4.set_ylabel('Debt-to-Income Ratio')
            ax4.set_title('Risk Concentration Heat Map')
            plt.colorbar(im, ax=ax4, label='Count')

            plt.tight_layout()

            return {
                'analysis_results': analysis,
                'plot_created': True,
                'plot_components': ['ltv_vs_dti_scatter', 'ltv_distribution', 'dti_distribution', 'risk_heatmap']
            }
        except Exception as e:
            return {'error': str(e)}

    def roe_debt_ratio_analysis(self, equity_values: List[float], net_incomes: List[float],
                              total_assets: List[float], total_debt: List[float]) -> Dict[str, Any]:
        """
        Analyze Return on Equity (ROE) vs Debt Ratio relationships.

        Parameters:
        -----------
        equity_values : List[float]
            Equity values
        net_incomes : List[float]
            Net income values
        total_assets : List[float]
            Total asset values
        total_debt : List[float]
            Total debt values

        Returns:
        --------
        Dict with ROE vs Debt Ratio analysis
        """
        try:
            if not all(len(lst) == len(equity_values) for lst in [net_incomes, total_assets, total_debt]):
                return {'error': 'All input lists must have the same length'}

            # Calculate ratios
            roe_ratios = [income / equity for income, equity in zip(net_incomes, equity_values) if equity > 0]
            debt_ratios = [debt / assets for debt, assets in zip(total_debt, total_assets) if assets > 0]

            # Ensure same length (remove entries where division by zero occurred)
            min_length = min(len(roe_ratios), len(debt_ratios))
            roe_ratios = roe_ratios[:min_length]
            debt_ratios = debt_ratios[:min_length]

            roe_array = np.array(roe_ratios)
            debt_array = np.array(debt_ratios)

            # Statistics
            roe_stats = {
                'mean': round(np.mean(roe_array) * 100, self.config.precision),  # Convert to percentage
                'median': round(np.median(roe_array) * 100, self.config.precision),
                'std': round(np.std(roe_array) * 100, self.config.precision),
                'min': round(np.min(roe_array) * 100, self.config.precision),
                'max': round(np.max(roe_array) * 100, self.config.precision)
            }

            debt_stats = {
                'mean': round(np.mean(debt_array) * 100, self.config.precision),
                'median': round(np.median(debt_array) * 100, self.config.precision),
                'std': round(np.std(debt_array) * 100, self.config.precision),
                'min': round(np.min(debt_array) * 100, self.config.precision),
                'max': round(np.max(debt_array) * 100, self.config.precision)
            }

            # Correlation and regression
            correlation = np.corrcoef(roe_array, debt_array)[0, 1]

            # Simple linear regression
            coefficients = np.polyfit(debt_array, roe_array, 1)
            slope, intercept = coefficients

            return {
                'n_observations': len(roe_ratios),
                'roe_statistics': roe_stats,
                'debt_ratio_statistics': debt_stats,
                'correlation': round(correlation, self.config.precision),
                'regression_slope': round(slope, self.config.precision),
                'regression_intercept': round(intercept, self.config.precision),
                'roe_percentages': [round(ratio * 100, self.config.precision) for ratio in roe_ratios],
                'debt_percentages': [round(ratio * 100, self.config.precision) for ratio in debt_ratios],
                'analysis_date': datetime.now()
            }
        except Exception as e:
            return {'error': str(e)}

    def plot_roe_debt_ratio(self, equity_values: List[float], net_incomes: List[float],
                           total_assets: List[float], total_debt: List[float]) -> Dict[str, Any]:
        """
        Plot ROE vs Debt Ratio analysis.

        Parameters:
        -----------
        Same as roe_debt_ratio_analysis

        Returns:
        --------
        Dict with plot information
        """
        try:
            analysis = self.roe_debt_ratio_analysis(equity_values, net_incomes, total_assets, total_debt)
            if 'error' in analysis:
                return analysis

            roe_percentages = analysis['roe_percentages']
            debt_percentages = analysis['debt_percentages']

            plt.style.use(self.config.plot_style)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Scatter plot with regression line
            ax1.scatter(debt_percentages, roe_percentages, alpha=0.6, c='blue', s=50)

            # Add regression line
            slope = analysis['regression_slope']
            intercept = analysis['regression_intercept']
            debt_range = np.linspace(min(debt_percentages), max(debt_percentages), 100)
            roe_line = slope * (np.array(debt_range) / 100) * 100 + intercept * 100  # Convert back to percentage
            ax1.plot(debt_range, roe_line, 'r-', label=f'Regression Line (slope={slope:.3f})')

            ax1.set_xlabel('Debt Ratio (%)')
            ax1.set_ylabel('Return on Equity (%)')
            ax1.set_title(f'ROE vs Debt Ratio (Correlation: {analysis["correlation"]:.3f})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Distribution plots
            ax2.hist(roe_percentages, bins=20, alpha=0.5, color='blue', label='ROE (%)', density=True)
            ax2_twin = ax2.twinx()
            ax2_twin.hist(debt_percentages, bins=20, alpha=0.5, color='red', label='Debt Ratio (%)', density=True)

            ax2.set_xlabel('Percentage')
            ax2.set_ylabel('ROE Density', color='blue')
            ax2_twin.set_ylabel('Debt Ratio Density', color='red')
            ax2.set_title('Distribution of ROE and Debt Ratios')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            return {
                'analysis_results': analysis,
                'plot_created': True
            }
        except Exception as e:
            return {'error': str(e)}

    def housing_microfinance_model(self, loan_amount: float, interest_rate: float,
                                 term_years: int, grace_period_months: int = 0,
                                 step_up_rate: float = 0.0) -> Dict[str, Any]:
        """
        Housing Microfinance Model (HMF) with progressive payment structure.

        Parameters:
        -----------
        loan_amount : float
            Initial loan amount
        interest_rate : float
            Annual interest rate
        term_years : int
            Loan term in years
        grace_period_months : int
            Grace period in months (interest-only)
        step_up_rate : float
            Annual step-up rate for payments

        Returns:
        --------
        Dict with HMF loan structure
        """
        try:
            total_months = term_years * 12
            monthly_rate = interest_rate / 12

            # Grace period calculations
            if grace_period_months > 0:
                grace_interest = loan_amount * monthly_rate * grace_period_months
                remaining_months = total_months - grace_period_months
            else:
                grace_interest = 0
                remaining_months = total_months

            # Base payment calculation for remaining period
            if monthly_rate == 0:
                base_payment = loan_amount / remaining_months
            else:
                base_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** remaining_months) / \
                              ((1 + monthly_rate) ** remaining_months - 1)

            # Generate payment schedule with step-ups
            payment_schedule = []
            outstanding_balance = loan_amount
            cumulative_interest = 0
            cumulative_principal = 0

            for month in range(1, total_months + 1):
                if month <= grace_period_months:
                    # Grace period: interest only
                    interest_payment = outstanding_balance * monthly_rate
                    principal_payment = 0
                    total_payment = interest_payment
                else:
                    # Regular payments with step-up
                    years_into_loan = (month - grace_period_months - 1) // 12
                    step_up_factor = (1 + step_up_rate) ** years_into_loan
                    adjusted_payment = base_payment * step_up_factor

                    interest_payment = outstanding_balance * monthly_rate
                    principal_payment = max(0, adjusted_payment - interest_payment)
                    total_payment = interest_payment + principal_payment

                outstanding_balance = max(0, outstanding_balance - principal_payment)
                cumulative_interest += interest_payment
                cumulative_principal += principal_payment

                payment_schedule.append({
                    'month': month,
                    'payment': round(total_payment, self.config.precision),
                    'principal': round(principal_payment, self.config.precision),
                    'interest': round(interest_payment, self.config.precision),
                    'balance': round(outstanding_balance, self.config.precision)
                })

            return {
                'loan_amount': round(loan_amount, self.config.precision),
                'interest_rate': round(interest_rate, self.config.precision),
                'term_years': term_years,
                'grace_period_months': grace_period_months,
                'step_up_rate': round(step_up_rate, self.config.precision),
                'base_payment': round(base_payment, self.config.precision),
                'total_interest': round(cumulative_interest, self.config.precision),
                'total_payments': round(cumulative_interest + cumulative_principal, self.config.precision),
                'payment_schedule': payment_schedule,
                'model_type': 'housing_microfinance'
            }
        except Exception as e:
            return {'error': str(e)}

    def contractual_savings_housing(self, target_amount: float, savings_rate: float,
                                  term_years: int, bonus_rate: float = 0.0) -> Dict[str, Any]:
        """
        Contractual Savings for Housing (CSH) model.

        Parameters:
        -----------
        target_amount : float
            Target savings amount for housing
        savings_rate : float
            Annual interest rate on savings
        term_years : int
            Savings period in years
        bonus_rate : float
            Additional bonus rate for completing the contract

        Returns:
        --------
        Dict with CSH calculations
        """
        try:
            monthly_rate = savings_rate / 12
            total_months = term_years * 12

            # Calculate required monthly savings to reach target
            if monthly_rate == 0:
                monthly_savings = target_amount / total_months
            else:
                # Future value of annuity formula solved for payment
                monthly_savings = target_amount * monthly_rate / ((1 + monthly_rate) ** total_months - 1)

            # Calculate savings schedule
            savings_schedule = []
            accumulated_savings = 0
            total_deposits = 0

            for month in range(1, total_months + 1):
                interest_earned = accumulated_savings * monthly_rate
                accumulated_savings += monthly_savings + interest_earned
                total_deposits += monthly_savings

                savings_schedule.append({
                    'month': month,
                    'deposit': round(monthly_savings, self.config.precision),
                    'interest': round(interest_earned, self.config.precision),
                    'balance': round(accumulated_savings, self.config.precision)
                })

            # Apply bonus at the end
            final_bonus = accumulated_savings * bonus_rate
            final_amount = accumulated_savings + final_bonus

            return {
                'target_amount': round(target_amount, self.config.precision),
                'monthly_savings_required': round(monthly_savings, self.config.precision),
                'total_deposits': round(total_deposits, self.config.precision),
                'interest_earned': round(accumulated_savings - total_deposits, self.config.precision),
                'bonus_amount': round(final_bonus, self.config.precision),
                'final_amount': round(final_amount, self.config.precision),
                'effective_rate': round(((final_amount / total_deposits) ** (1/term_years) - 1), self.config.precision),
                'savings_schedule': savings_schedule,
                'model_type': 'contractual_savings_housing'
            }
        except Exception as e:
            return {'error': str(e)}

    def housing_bond_pricing(self, face_value: float, coupon_rate: float, yield_rate: float,
                           years_to_maturity: float, tax_exemption: bool = True,
                           tax_rate: float = 0.0) -> Dict[str, Any]:
        """
        Housing bond pricing with tax considerations.

        Parameters:
        -----------
        face_value : float
            Face value of the bond
        coupon_rate : float
            Annual coupon rate
        yield_rate : float
            Required yield rate
        years_to_maturity : float
            Years to maturity
        tax_exemption : bool
            Whether the bond is tax-exempt
        tax_rate : float
            Marginal tax rate (if applicable)

        Returns:
        --------
        Dict with housing bond pricing
        """
        try:
            # Adjust yield for tax considerations
            if tax_exemption:
                effective_yield = yield_rate
                tax_equivalent_yield = yield_rate / (1 - tax_rate) if tax_rate > 0 else yield_rate
            else:
                effective_yield = yield_rate * (1 - tax_rate)
                tax_equivalent_yield = yield_rate

            # Semi-annual payment structure (typical for housing bonds)
            periods = int(years_to_maturity * 2)
            periodic_yield = effective_yield / 2
            periodic_coupon = (coupon_rate * face_value) / 2

            # Present value calculations
            if periodic_yield == 0:
                pv_coupons = periodic_coupon * periods
            else:
                pv_coupons = periodic_coupon * (1 - (1 + periodic_yield) ** (-periods)) / periodic_yield

            pv_face_value = face_value / (1 + periodic_yield) ** periods
            bond_price = pv_coupons + pv_face_value

            # Additional metrics
            current_yield = (coupon_rate * face_value) / bond_price if bond_price > 0 else 0
            yield_to_maturity = effective_yield

            return {
                'bond_price': round(bond_price, self.config.precision),
                'face_value': round(face_value, self.config.precision),
                'coupon_rate': round(coupon_rate, self.config.precision),
                'effective_yield': round(effective_yield, self.config.precision),
                'tax_equivalent_yield': round(tax_equivalent_yield, self.config.precision),
                'current_yield': round(current_yield, self.config.precision),
                'yield_to_maturity': round(yield_to_maturity, self.config.precision),
                'periods': periods,
                'periodic_coupon': round(periodic_coupon, self.config.precision),
                'pv_coupons': round(pv_coupons, self.config.precision),
                'pv_face_value': round(pv_face_value, self.config.precision),
                'tax_exempt': tax_exemption,
                'bond_type': 'housing_bond'
            }
        except Exception as e:
            return {'error': str(e)}

    def mortgage_liquidity_facility(self, mortgage_pool_value: float, advance_rate: float,
                                  facility_rate: float, term_days: int,
                                  haircut_rate: float = 0.05) -> Dict[str, Any]:
        """
        Mortgage Liquidity Facility pricing and terms.

        Parameters:
        -----------
        mortgage_pool_value : float
            Value of mortgage pool as collateral
        advance_rate : float
            Percentage of pool value advanced
        facility_rate : float
            Annual interest rate on facility
        term_days : int
            Term of facility in days
        haircut_rate : float
            Haircut applied to collateral value

        Returns:
        --------
        Dict with facility terms and pricing
        """
        try:
            # Calculate facility terms
            adjusted_collateral_value = mortgage_pool_value * (1 - haircut_rate)
            max_advance = adjusted_collateral_value * advance_rate

            # Daily rate calculation
            daily_rate = facility_rate / 365

            # Interest calculation
            total_interest = max_advance * daily_rate * term_days
            total_repayment = max_advance + total_interest

            # Risk metrics
            ltv_facility = max_advance / mortgage_pool_value
            coverage_ratio = mortgage_pool_value / max_advance

            return {
                'mortgage_pool_value': round(mortgage_pool_value, self.config.precision),
                'haircut_rate': round(haircut_rate, self.config.precision),
                'adjusted_collateral_value': round(adjusted_collateral_value, self.config.precision),
                'advance_rate': round(advance_rate, self.config.precision),
                'max_advance': round(max_advance, self.config.precision),
                'facility_rate': round(facility_rate, self.config.precision),
                'daily_rate': round(daily_rate, self.config.precision),
                'term_days': term_days,
                'total_interest': round(total_interest, self.config.precision),
                'total_repayment': round(total_repayment, self.config.precision),
                'ltv_facility': round(ltv_facility, self.config.precision),
                'coverage_ratio': round(coverage_ratio, self.config.precision),
                'facility_type': 'mortgage_liquidity'
            }
        except Exception as e:
            return {'error': str(e)}

    def reit_pricing(self, annual_dividends: float, dividend_growth_rate: float,
                    required_return: float, nav_per_share: float = None,
                    price_to_nav_ratio: float = 1.0) -> Dict[str, Any]:
        """
        Real Estate Investment Trust (REIT) pricing using dividend discount model.

        Parameters:
        -----------
        annual_dividends : float
            Current annual dividends per share
        dividend_growth_rate : float
            Expected annual dividend growth rate
        required_return : float
            Required rate of return
        nav_per_share : float
            Net Asset Value per share
        price_to_nav_ratio : float
            Price-to-NAV ratio for market pricing

        Returns:
        --------
        Dict with REIT valuation
        """
        try:
            if required_return <= dividend_growth_rate:
                return {'error': 'Required return must be greater than dividend growth rate'}

            # Gordon Growth Model
            intrinsic_value = annual_dividends * (1 + dividend_growth_rate) / (required_return - dividend_growth_rate)

            # Market-based pricing (if NAV available)
            market_price = nav_per_share * price_to_nav_ratio if nav_per_share else None

            # Yield calculations
            current_yield = annual_dividends / intrinsic_value if intrinsic_value > 0 else 0

            # Future dividend projections (5 years)
            dividend_projections = []
            for year in range(1, 6):
                future_dividend = annual_dividends * (1 + dividend_growth_rate) ** year
                dividend_projections.append({
                    'year': year,
                    'dividend': round(future_dividend, self.config.precision)
                })

            result = {
                'intrinsic_value': round(intrinsic_value, self.config.precision),
                'current_dividends': round(annual_dividends, self.config.precision),
                'dividend_growth_rate': round(dividend_growth_rate, self.config.precision),
                'required_return': round(required_return, self.config.precision),
                'current_yield': round(current_yield, self.config.precision),
                'dividend_projections': dividend_projections,
                'investment_type': 'REIT'
            }

            if nav_per_share:
                result.update({
                    'nav_per_share': round(nav_per_share, self.config.precision),
                    'price_to_nav_ratio': round(price_to_nav_ratio, self.config.precision),
                    'market_price': round(market_price, self.config.precision),
                    'premium_discount_to_nav': round(((market_price / nav_per_share) - 1) * 100, self.config.precision),
                    'premium_discount_to_intrinsic': round(((market_price / intrinsic_value) - 1) * 100, self.config.precision)
                })

            return result
        except Exception as e:
            return {'error': str(e)}

    def reoc_pricing(self, noi: float, cap_rate: float, debt_amount: float,
                    debt_rate: float, shares_outstanding: float,
                    development_pipeline_value: float = 0.0) -> Dict[str, Any]:
        """
        Real Estate Operating Company (REOC) pricing.

        Parameters:
        -----------
        noi : float
            Net Operating Income
        cap_rate : float
            Capitalization rate
        debt_amount : float
            Total debt amount
        debt_rate : float
            Cost of debt
        shares_outstanding : float
            Number of shares outstanding
        development_pipeline_value : float
            Value of development pipeline

        Returns:
        --------
        Dict with REOC valuation
        """
        try:
            # Asset valuation
            property_value = noi / cap_rate if cap_rate > 0 else 0
            total_asset_value = property_value + development_pipeline_value

            # Debt service
            annual_debt_service = debt_amount * debt_rate

            # Equity calculations
            equity_value = total_asset_value - debt_amount
            net_income = noi - annual_debt_service

            # Per share metrics
            nav_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0
            earnings_per_share = net_income / shares_outstanding if shares_outstanding > 0 else 0

            # Financial ratios
            debt_to_assets = debt_amount / total_asset_value if total_asset_value > 0 else 0
            debt_to_equity = debt_amount / equity_value if equity_value > 0 else float('inf')
            interest_coverage = noi / annual_debt_service if annual_debt_service > 0 else float('inf')

            # Return metrics
            roe = net_income / equity_value if equity_value > 0 else 0
            roa = net_income / total_asset_value if total_asset_value > 0 else 0

            return {
                'property_value': round(property_value, self.config.precision),
                'development_pipeline_value': round(development_pipeline_value, self.config.precision),
                'total_asset_value': round(total_asset_value, self.config.precision),
                'debt_amount': round(debt_amount, self.config.precision),
                'equity_value': round(equity_value, self.config.precision),
                'nav_per_share': round(nav_per_share, self.config.precision),
                'net_income': round(net_income, self.config.precision),
                'earnings_per_share': round(earnings_per_share, self.config.precision),
                'debt_to_assets': round(debt_to_assets, self.config.precision),
                'debt_to_equity': round(debt_to_equity, self.config.precision),
                'interest_coverage': round(interest_coverage, self.config.precision),
                'roe': round(roe, self.config.precision),
                'roa': round(roa, self.config.precision),
                'cap_rate': round(cap_rate, self.config.precision),
                'debt_rate': round(debt_rate, self.config.precision),
                'investment_type': 'REOC'
            }
        except Exception as e:
            return {'error': str(e)}

    def comprehensive_mortgage_analysis(self, loan_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Comprehensive mortgage market analysis combining multiple metrics.

        Parameters:
        -----------
        loan_data : Dict
            Dictionary containing lists of loan data:
            - property_values, loan_amounts, incomes, debt_payments
            - equity_values, net_incomes, total_assets, total_debt

        Returns:
        --------
        Dict with comprehensive mortgage market analysis
        """
        try:
            results = {
                'analysis_date': datetime.now(),
                'comprehensive_analysis': True
            }

            # LTV/DTI Analysis
            if all(key in loan_data for key in ['property_values', 'loan_amounts', 'incomes', 'debt_payments']):
                ltv_dti = self.ltv_dti_analysis(
                    loan_data['property_values'],
                    loan_data['loan_amounts'],
                    loan_data['incomes'],
                    loan_data['debt_payments']
                )
                results['ltv_dti_analysis'] = ltv_dti

            # ROE/Debt Ratio Analysis
            if all(key in loan_data for key in ['equity_values', 'net_incomes', 'total_assets', 'total_debt']):
                roe_debt = self.roe_debt_ratio_analysis(
                    loan_data['equity_values'],
                    loan_data['net_incomes'],
                    loan_data['total_assets'],
                    loan_data['total_debt']
                )
                results['roe_debt_analysis'] = roe_debt

            # Market structure insights
            if 'ltv_dti_analysis' in results:
                ltv_stats = results['ltv_dti_analysis']['ltv_statistics']
                dti_stats = results['ltv_dti_analysis']['dti_statistics']

                market_insights = {
                    'average_ltv': ltv_stats['mean'],
                    'average_dti': dti_stats['mean'],
                    'risk_concentration': results['ltv_dti_analysis']['risk_analysis']['high_risk_percentage'],
                    'market_stability_score': self._calculate_stability_score(ltv_stats, dti_stats)
                }
                results['market_insights'] = market_insights

            return results
        except Exception as e:
            return {'error': str(e)}

    def _calculate_stability_score(self, ltv_stats: Dict, dti_stats: Dict) -> float:
        """Calculate a market stability score based on LTV and DTI statistics."""
        try:
            # Lower LTV and DTI means are better for stability
            ltv_score = max(0, 100 - (ltv_stats['mean'] * 100))
            dti_score = max(0, 100 - (dti_stats['mean'] * 100))

            # Lower standard deviations indicate more stability
            ltv_stability = max(0, 100 - (ltv_stats['std'] * 200))  # Penalize high volatility
            dti_stability = max(0, 100 - (dti_stats['std'] * 200))

            # Weighted average
            stability_score = (ltv_score * 0.3 + dti_score * 0.3 + ltv_stability * 0.2 + dti_stability * 0.2)

            return round(min(100, max(0, stability_score)), self.config.precision)
        except Exception:
            return 50.0  # Default neutral score