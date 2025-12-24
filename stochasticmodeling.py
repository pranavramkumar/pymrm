"""
Stochastic Modeling Module
==========================

This module provides comprehensive tools for stochastic modeling including:
- Estimators for unknown stochastic model outcomes
- Expected Value and Variance calculations
- Monte Carlo Simulation framework
- Schelling's Segregation model
- Agent-Based Modeling framework with validation

Author: Claude Code Assistant
Date: 2025-09-29
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import random
import warnings
from scipy import stats
from collections import defaultdict
import networkx as nx
from datetime import datetime
import pickle


class StochasticEstimators:
    """
    Class for estimating parameters of unknown stochastic models.
    """

    def __init__(self):
        self.fitted_models = {}
        self.estimation_history = []

    def method_of_moments(self, data: np.ndarray, distribution: str = 'normal',
                         moments: int = 2) -> Dict[str, Any]:
        """
        Estimate parameters using Method of Moments.

        Parameters:
        -----------
        data : np.ndarray
            Observed data
        distribution : str
            Target distribution ('normal', 'exponential', 'gamma', 'beta')
        moments : int
            Number of moments to match

        Returns:
        --------
        Dict with estimated parameters and diagnostics
        """
        try:
            data = np.asarray(data)
            n = len(data)

            # Calculate sample moments
            sample_moments = {}
            for k in range(1, moments + 1):
                sample_moments[f'moment_{k}'] = np.mean(data**k)

            # Calculate central moments
            mean = np.mean(data)
            variance = np.var(data, ddof=1)
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)

            sample_moments.update({
                'mean': mean,
                'variance': variance,
                'skewness': skewness,
                'kurtosis': kurtosis
            })

            # Estimate parameters based on distribution
            if distribution == 'normal':
                mu_hat = mean
                sigma_hat = np.sqrt(variance)
                parameters = {'mu': mu_hat, 'sigma': sigma_hat}

            elif distribution == 'exponential':
                lambda_hat = 1 / mean
                parameters = {'lambda': lambda_hat}

            elif distribution == 'gamma':
                # Method of moments for Gamma distribution
                alpha_hat = mean**2 / variance
                beta_hat = variance / mean
                parameters = {'alpha': alpha_hat, 'beta': beta_hat}

            elif distribution == 'beta':
                # Method of moments for Beta distribution
                mean_sq = sample_moments['moment_2']
                var_est = mean_sq - mean**2

                alpha_hat = mean * (mean * (1 - mean) / var_est - 1)
                beta_hat = (1 - mean) * (mean * (1 - mean) / var_est - 1)
                parameters = {'alpha': alpha_hat, 'beta': beta_hat}

            else:
                raise ValueError(f"Distribution '{distribution}' not supported")

            # Calculate standard errors (asymptotic)
            standard_errors = {}
            if distribution == 'normal':
                standard_errors = {
                    'mu_se': np.sqrt(variance / n),
                    'sigma_se': np.sqrt(variance / (2 * n))
                }

            # Goodness of fit test
            if distribution == 'normal':
                ks_stat, ks_pvalue = stats.kstest(data, 'norm', args=(mu_hat, sigma_hat))
            elif distribution == 'exponential':
                ks_stat, ks_pvalue = stats.kstest(data, 'expon', args=(0, 1/lambda_hat))
            else:
                ks_stat, ks_pvalue = None, None

            result = {
                'method': 'Method of Moments',
                'distribution': distribution,
                'parameters': parameters,
                'sample_moments': sample_moments,
                'standard_errors': standard_errors,
                'sample_size': n,
                'goodness_of_fit': {
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pvalue
                },
                'estimation_date': datetime.now()
            }

            self.fitted_models[f'mom_{distribution}_{len(self.fitted_models)}'] = result
            self.estimation_history.append(result)

            return result

        except Exception as e:
            return {'error': str(e), 'method': 'Method of Moments'}

    def maximum_likelihood(self, data: np.ndarray, distribution: str = 'normal',
                          initial_params: Dict = None) -> Dict[str, Any]:
        """
        Estimate parameters using Maximum Likelihood Estimation.

        Parameters:
        -----------
        data : np.ndarray
            Observed data
        distribution : str
            Target distribution
        initial_params : Dict
            Initial parameter values for optimization

        Returns:
        --------
        Dict with MLE estimates and diagnostics
        """
        try:
            data = np.asarray(data)
            n = len(data)

            # Use scipy.stats for MLE
            if distribution == 'normal':
                mu_mle, sigma_mle = stats.norm.fit(data)
                parameters = {'mu': mu_mle, 'sigma': sigma_mle}
                log_likelihood = np.sum(stats.norm.logpdf(data, mu_mle, sigma_mle))

            elif distribution == 'exponential':
                _, lambda_inv_mle = stats.expon.fit(data, floc=0)
                lambda_mle = 1 / lambda_inv_mle
                parameters = {'lambda': lambda_mle}
                log_likelihood = np.sum(stats.expon.logpdf(data, 0, lambda_inv_mle))

            elif distribution == 'gamma':
                alpha_mle, _, beta_mle = stats.gamma.fit(data, floc=0)
                parameters = {'alpha': alpha_mle, 'beta': beta_mle}
                log_likelihood = np.sum(stats.gamma.logpdf(data, alpha_mle, 0, beta_mle))

            elif distribution == 'beta':
                alpha_mle, beta_mle, _, _ = stats.beta.fit(data)
                parameters = {'alpha': alpha_mle, 'beta': beta_mle}
                log_likelihood = np.sum(stats.beta.logpdf(data, alpha_mle, beta_mle))

            else:
                raise ValueError(f"Distribution '{distribution}' not supported")

            # Calculate AIC and BIC
            k = len(parameters)  # number of parameters
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood

            # Calculate Hessian for standard errors (numerical approximation)
            def neg_log_likelihood(params):
                if distribution == 'normal':
                    return -np.sum(stats.norm.logpdf(data, params[0], params[1]))
                elif distribution == 'exponential':
                    return -np.sum(stats.expon.logpdf(data, 0, 1/params[0]))
                # Add other distributions as needed
                return 0

            result = {
                'method': 'Maximum Likelihood',
                'distribution': distribution,
                'parameters': parameters,
                'log_likelihood': log_likelihood,
                'aic': aic,
                'bic': bic,
                'sample_size': n,
                'estimation_date': datetime.now()
            }

            self.fitted_models[f'mle_{distribution}_{len(self.fitted_models)}'] = result
            self.estimation_history.append(result)

            return result

        except Exception as e:
            return {'error': str(e), 'method': 'Maximum Likelihood'}

    def bayesian_estimation(self, data: np.ndarray, distribution: str = 'normal',
                           prior_params: Dict = None, n_samples: int = 10000) -> Dict[str, Any]:
        """
        Bayesian parameter estimation using conjugate priors where possible.

        Parameters:
        -----------
        data : np.ndarray
            Observed data
        distribution : str
            Target distribution
        prior_params : Dict
            Prior hyperparameters
        n_samples : int
            Number of posterior samples

        Returns:
        --------
        Dict with Bayesian estimates and credible intervals
        """
        try:
            data = np.asarray(data)
            n = len(data)

            if distribution == 'normal' and prior_params:
                # Normal-Normal conjugate prior for mean (known variance)
                # or Normal-Inverse-Gamma for unknown mean and variance

                if 'sigma' in prior_params and 'mu_prior' in prior_params:
                    # Known variance case
                    sigma = prior_params['sigma']
                    mu_prior = prior_params['mu_prior']
                    tau_prior = prior_params.get('tau_prior', 1.0)  # prior precision

                    # Posterior parameters
                    tau_post = tau_prior + n / (sigma**2)
                    mu_post = (tau_prior * mu_prior + n * np.mean(data) / (sigma**2)) / tau_post

                    # Sample from posterior
                    posterior_samples = np.random.normal(mu_post, 1/np.sqrt(tau_post), n_samples)

                    parameters = {
                        'mu_posterior_mean': mu_post,
                        'mu_posterior_var': 1/tau_post,
                        'sigma': sigma
                    }

                else:
                    # Unknown mean and variance - use Normal-Inverse-Gamma
                    mu0 = prior_params.get('mu0', 0)
                    lambda0 = prior_params.get('lambda0', 1)
                    alpha0 = prior_params.get('alpha0', 1)
                    beta0 = prior_params.get('beta0', 1)

                    # Posterior parameters
                    lambda_n = lambda0 + n
                    mu_n = (lambda0 * mu0 + n * np.mean(data)) / lambda_n
                    alpha_n = alpha0 + n/2
                    beta_n = beta0 + 0.5 * np.sum((data - np.mean(data))**2) + \
                            (lambda0 * n * (np.mean(data) - mu0)**2) / (2 * lambda_n)

                    # Sample from posterior
                    sigma2_samples = stats.invgamma.rvs(alpha_n, scale=beta_n, size=n_samples)
                    mu_samples = np.array([np.random.normal(mu_n, np.sqrt(sigma2/lambda_n))
                                         for sigma2 in sigma2_samples])

                    posterior_samples = {'mu': mu_samples, 'sigma2': sigma2_samples}

                    parameters = {
                        'mu_posterior_mean': np.mean(mu_samples),
                        'mu_posterior_var': np.var(mu_samples),
                        'sigma2_posterior_mean': np.mean(sigma2_samples),
                        'sigma2_posterior_var': np.var(sigma2_samples)
                    }

            elif distribution == 'exponential':
                # Gamma prior for exponential rate parameter
                alpha_prior = prior_params.get('alpha', 1) if prior_params else 1
                beta_prior = prior_params.get('beta', 1) if prior_params else 1

                # Posterior parameters (Gamma is conjugate to Exponential)
                alpha_post = alpha_prior + n
                beta_post = beta_prior + np.sum(data)

                # Sample from posterior
                posterior_samples = stats.gamma.rvs(alpha_post, scale=1/beta_post, size=n_samples)

                parameters = {
                    'lambda_posterior_mean': alpha_post / beta_post,
                    'lambda_posterior_var': alpha_post / (beta_post**2)
                }

            else:
                raise ValueError(f"Bayesian estimation for '{distribution}' not implemented")

            # Calculate credible intervals
            if isinstance(posterior_samples, dict):
                credible_intervals = {}
                for param, samples in posterior_samples.items():
                    credible_intervals[param] = {
                        '95%_ci': np.percentile(samples, [2.5, 97.5]),
                        '90%_ci': np.percentile(samples, [5, 95]),
                        '68%_ci': np.percentile(samples, [16, 84])
                    }
            else:
                credible_intervals = {
                    '95%_ci': np.percentile(posterior_samples, [2.5, 97.5]),
                    '90%_ci': np.percentile(posterior_samples, [5, 95]),
                    '68%_ci': np.percentile(posterior_samples, [16, 84])
                }

            result = {
                'method': 'Bayesian Estimation',
                'distribution': distribution,
                'parameters': parameters,
                'posterior_samples': posterior_samples,
                'credible_intervals': credible_intervals,
                'prior_params': prior_params,
                'sample_size': n,
                'n_posterior_samples': n_samples,
                'estimation_date': datetime.now()
            }

            self.fitted_models[f'bayes_{distribution}_{len(self.fitted_models)}'] = result
            self.estimation_history.append(result)

            return result

        except Exception as e:
            return {'error': str(e), 'method': 'Bayesian Estimation'}

    def bootstrap_estimation(self, data: np.ndarray, estimator_func: Callable,
                           n_bootstrap: int = 1000, confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Bootstrap estimation for any parameter estimator.

        Parameters:
        -----------
        data : np.ndarray
            Original data
        estimator_func : Callable
            Function that takes data and returns parameter estimate
        n_bootstrap : int
            Number of bootstrap samples
        confidence_level : float
            Confidence level for intervals

        Returns:
        --------
        Dict with bootstrap estimates and confidence intervals
        """
        try:
            data = np.asarray(data)
            n = len(data)

            # Original estimate
            original_estimate = estimator_func(data)

            # Bootstrap samples
            bootstrap_estimates = []
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(data, size=n, replace=True)
                bootstrap_estimate = estimator_func(bootstrap_sample)
                bootstrap_estimates.append(bootstrap_estimate)

            bootstrap_estimates = np.array(bootstrap_estimates)

            # Calculate statistics
            bootstrap_mean = np.mean(bootstrap_estimates)
            bootstrap_std = np.std(bootstrap_estimates)
            bootstrap_bias = bootstrap_mean - original_estimate

            # Confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha/2) * 100
            upper_percentile = (1 - alpha/2) * 100

            ci_percentile = np.percentile(bootstrap_estimates, [lower_percentile, upper_percentile])

            # Bias-corrected and accelerated (BCa) interval
            try:
                # Acceleration parameter
                n_jackknife = n
                jackknife_estimates = []
                for i in range(n):
                    jackknife_sample = np.delete(data, i)
                    jackknife_estimate = estimator_func(jackknife_sample)
                    jackknife_estimates.append(jackknife_estimate)

                jackknife_estimates = np.array(jackknife_estimates)
                jackknife_mean = np.mean(jackknife_estimates)

                acceleration = np.sum((jackknife_mean - jackknife_estimates)**3) / \
                              (6 * (np.sum((jackknife_mean - jackknife_estimates)**2))**(3/2))

                # Bias correction
                bias_correction = stats.norm.ppf(np.mean(bootstrap_estimates < original_estimate))

                # BCa percentiles
                z_alpha_2 = stats.norm.ppf(alpha/2)
                z_1_alpha_2 = stats.norm.ppf(1 - alpha/2)

                alpha_1 = stats.norm.cdf(bias_correction +
                                       (bias_correction + z_alpha_2)/(1 - acceleration*(bias_correction + z_alpha_2)))
                alpha_2 = stats.norm.cdf(bias_correction +
                                       (bias_correction + z_1_alpha_2)/(1 - acceleration*(bias_correction + z_1_alpha_2)))

                ci_bca = np.percentile(bootstrap_estimates, [alpha_1*100, alpha_2*100])

            except:
                ci_bca = ci_percentile  # Fallback to percentile method

            result = {
                'method': 'Bootstrap Estimation',
                'original_estimate': original_estimate,
                'bootstrap_mean': bootstrap_mean,
                'bootstrap_std': bootstrap_std,
                'bootstrap_bias': bootstrap_bias,
                'bootstrap_estimates': bootstrap_estimates,
                'confidence_intervals': {
                    'percentile': ci_percentile,
                    'bca': ci_bca
                },
                'confidence_level': confidence_level,
                'n_bootstrap': n_bootstrap,
                'sample_size': n,
                'estimation_date': datetime.now()
            }

            return result

        except Exception as e:
            return {'error': str(e), 'method': 'Bootstrap Estimation'}


class ExpectedValueVariance:
    """
    Class for calculating expected values and variances of stochastic processes.
    """

    def __init__(self):
        self.calculation_history = []

    def analytical_moments(self, distribution: str, parameters: Dict[str, float],
                          moments: List[str] = ['mean', 'variance']) -> Dict[str, Any]:
        """
        Calculate analytical moments for known distributions.

        Parameters:
        -----------
        distribution : str
            Distribution name
        parameters : Dict
            Distribution parameters
        moments : List[str]
            Moments to calculate

        Returns:
        --------
        Dict with calculated moments
        """
        try:
            result = {'distribution': distribution, 'parameters': parameters}

            if distribution == 'normal':
                mu = parameters['mu']
                sigma = parameters['sigma']

                if 'mean' in moments:
                    result['mean'] = mu
                if 'variance' in moments:
                    result['variance'] = sigma**2
                if 'skewness' in moments:
                    result['skewness'] = 0
                if 'kurtosis' in moments:
                    result['kurtosis'] = 0
                if 'std' in moments:
                    result['std'] = sigma

            elif distribution == 'exponential':
                lambda_param = parameters['lambda']

                if 'mean' in moments:
                    result['mean'] = 1 / lambda_param
                if 'variance' in moments:
                    result['variance'] = 1 / (lambda_param**2)
                if 'skewness' in moments:
                    result['skewness'] = 2
                if 'kurtosis' in moments:
                    result['kurtosis'] = 6
                if 'std' in moments:
                    result['std'] = 1 / lambda_param

            elif distribution == 'gamma':
                alpha = parameters['alpha']
                beta = parameters['beta']

                if 'mean' in moments:
                    result['mean'] = alpha * beta
                if 'variance' in moments:
                    result['variance'] = alpha * beta**2
                if 'skewness' in moments:
                    result['skewness'] = 2 / np.sqrt(alpha)
                if 'kurtosis' in moments:
                    result['kurtosis'] = 6 / alpha
                if 'std' in moments:
                    result['std'] = beta * np.sqrt(alpha)

            elif distribution == 'beta':
                alpha = parameters['alpha']
                beta = parameters['beta']

                if 'mean' in moments:
                    result['mean'] = alpha / (alpha + beta)
                if 'variance' in moments:
                    result['variance'] = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
                if 'skewness' in moments:
                    result['skewness'] = 2 * (beta - alpha) * np.sqrt(alpha + beta + 1) / \
                                       ((alpha + beta + 2) * np.sqrt(alpha * beta))
                if 'std' in moments:
                    result['std'] = np.sqrt(result.get('variance', 0))

            elif distribution == 'poisson':
                lambda_param = parameters['lambda']

                if 'mean' in moments:
                    result['mean'] = lambda_param
                if 'variance' in moments:
                    result['variance'] = lambda_param
                if 'skewness' in moments:
                    result['skewness'] = 1 / np.sqrt(lambda_param)
                if 'kurtosis' in moments:
                    result['kurtosis'] = 1 / lambda_param
                if 'std' in moments:
                    result['std'] = np.sqrt(lambda_param)

            elif distribution == 'uniform':
                a = parameters['a']
                b = parameters['b']

                if 'mean' in moments:
                    result['mean'] = (a + b) / 2
                if 'variance' in moments:
                    result['variance'] = (b - a)**2 / 12
                if 'skewness' in moments:
                    result['skewness'] = 0
                if 'kurtosis' in moments:
                    result['kurtosis'] = -1.2  # Excess kurtosis
                if 'std' in moments:
                    result['std'] = (b - a) / np.sqrt(12)

            else:
                raise ValueError(f"Distribution '{distribution}' not supported")

            result['calculation_date'] = datetime.now()
            self.calculation_history.append(result)

            return result

        except Exception as e:
            return {'error': str(e), 'distribution': distribution}

    def empirical_moments(self, data: np.ndarray,
                         moments: List[str] = ['mean', 'variance']) -> Dict[str, Any]:
        """
        Calculate empirical moments from data.

        Parameters:
        -----------
        data : np.ndarray
            Sample data
        moments : List[str]
            Moments to calculate

        Returns:
        --------
        Dict with empirical moments
        """
        try:
            data = np.asarray(data)
            result = {'sample_size': len(data)}

            if 'mean' in moments:
                result['mean'] = np.mean(data)
            if 'variance' in moments:
                result['variance'] = np.var(data, ddof=1)
            if 'std' in moments:
                result['std'] = np.std(data, ddof=1)
            if 'skewness' in moments:
                result['skewness'] = stats.skew(data)
            if 'kurtosis' in moments:
                result['kurtosis'] = stats.kurtosis(data)
            if 'median' in moments:
                result['median'] = np.median(data)
            if 'mode' in moments:
                try:
                    mode_result = stats.mode(data)
                    result['mode'] = mode_result.mode[0] if hasattr(mode_result, 'mode') else mode_result[0]
                except:
                    result['mode'] = None
            if 'range' in moments:
                result['range'] = np.max(data) - np.min(data)
            if 'iqr' in moments:
                result['iqr'] = np.percentile(data, 75) - np.percentile(data, 25)

            # Higher moments
            if 'moment_3' in moments:
                result['moment_3'] = np.mean((data - np.mean(data))**3)
            if 'moment_4' in moments:
                result['moment_4'] = np.mean((data - np.mean(data))**4)

            # Confidence intervals for mean
            if 'mean_ci' in moments:
                se_mean = result.get('std', np.std(data, ddof=1)) / np.sqrt(len(data))
                t_critical = stats.t.ppf(0.975, len(data) - 1)
                margin_error = t_critical * se_mean
                result['mean_ci_95'] = [result['mean'] - margin_error, result['mean'] + margin_error]

            result['calculation_date'] = datetime.now()
            self.calculation_history.append(result)

            return result

        except Exception as e:
            return {'error': str(e)}

    def moment_generating_function(self, distribution: str, parameters: Dict[str, float],
                                  t_values: np.ndarray) -> Dict[str, Any]:
        """
        Calculate moment generating function values.

        Parameters:
        -----------
        distribution : str
            Distribution name
        parameters : Dict
            Distribution parameters
        t_values : np.ndarray
            Values at which to evaluate MGF

        Returns:
        --------
        Dict with MGF values and derivatives
        """
        try:
            t_values = np.asarray(t_values)
            result = {'distribution': distribution, 'parameters': parameters, 't_values': t_values}

            if distribution == 'normal':
                mu = parameters['mu']
                sigma = parameters['sigma']

                mgf_values = np.exp(mu * t_values + 0.5 * sigma**2 * t_values**2)
                result['mgf_values'] = mgf_values

                # First derivative (first moment)
                result['mgf_first_derivative'] = mgf_values * (mu + sigma**2 * t_values)

                # Second derivative (second moment)
                result['mgf_second_derivative'] = mgf_values * \
                    ((mu + sigma**2 * t_values)**2 + sigma**2)

            elif distribution == 'exponential':
                lambda_param = parameters['lambda']

                # MGF exists for t < lambda
                valid_t = t_values[t_values < lambda_param]
                mgf_values = lambda_param / (lambda_param - valid_t)

                result['mgf_values'] = mgf_values
                result['valid_t_values'] = valid_t
                result['mgf_domain'] = f't < {lambda_param}'

            elif distribution == 'gamma':
                alpha = parameters['alpha']
                beta = parameters['beta']

                # MGF exists for t < 1/beta
                valid_t = t_values[t_values < 1/beta]
                mgf_values = (1 - beta * valid_t)**(-alpha)

                result['mgf_values'] = mgf_values
                result['valid_t_values'] = valid_t
                result['mgf_domain'] = f't < {1/beta}'

            else:
                raise ValueError(f"MGF for '{distribution}' not implemented")

            result['calculation_date'] = datetime.now()

            return result

        except Exception as e:
            return {'error': str(e), 'distribution': distribution}

    def characteristic_function(self, distribution: str, parameters: Dict[str, float],
                               t_values: np.ndarray) -> Dict[str, Any]:
        """
        Calculate characteristic function values.

        Parameters:
        -----------
        distribution : str
            Distribution name
        parameters : Dict
            Distribution parameters
        t_values : np.ndarray
            Values at which to evaluate CF

        Returns:
        --------
        Dict with characteristic function values
        """
        try:
            t_values = np.asarray(t_values)
            result = {'distribution': distribution, 'parameters': parameters, 't_values': t_values}

            if distribution == 'normal':
                mu = parameters['mu']
                sigma = parameters['sigma']

                cf_values = np.exp(1j * mu * t_values - 0.5 * sigma**2 * t_values**2)
                result['cf_values'] = cf_values
                result['cf_real'] = np.real(cf_values)
                result['cf_imag'] = np.imag(cf_values)
                result['cf_magnitude'] = np.abs(cf_values)
                result['cf_phase'] = np.angle(cf_values)

            elif distribution == 'exponential':
                lambda_param = parameters['lambda']

                cf_values = lambda_param / (lambda_param - 1j * t_values)
                result['cf_values'] = cf_values
                result['cf_real'] = np.real(cf_values)
                result['cf_imag'] = np.imag(cf_values)
                result['cf_magnitude'] = np.abs(cf_values)
                result['cf_phase'] = np.angle(cf_values)

            else:
                raise ValueError(f"Characteristic function for '{distribution}' not implemented")

            result['calculation_date'] = datetime.now()

            return result

        except Exception as e:
            return {'error': str(e), 'distribution': distribution}


class MonteCarloSimulation:
    """
    Comprehensive Monte Carlo simulation framework.
    """

    def __init__(self, random_seed: Optional[int] = None):
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        self.simulation_history = []
        self.random_seed = random_seed

    def simulate_distribution(self, distribution: str, parameters: Dict[str, float],
                            n_trials: int = 10000, n_runs: int = 1) -> Dict[str, Any]:
        """
        Monte Carlo simulation for various probability distributions.

        Parameters:
        -----------
        distribution : str
            Distribution to simulate
        parameters : Dict
            Distribution parameters
        n_trials : int
            Number of trials per run
        n_runs : int
            Number of simulation runs

        Returns:
        --------
        Dict with simulation results and statistics
        """
        try:
            all_samples = []
            run_statistics = []

            for run in range(n_runs):
                if distribution == 'normal':
                    mu = parameters['mu']
                    sigma = parameters['sigma']
                    samples = np.random.normal(mu, sigma, n_trials)

                elif distribution == 'exponential':
                    lambda_param = parameters['lambda']
                    samples = np.random.exponential(1/lambda_param, n_trials)

                elif distribution == 'gamma':
                    alpha = parameters['alpha']
                    beta = parameters['beta']
                    samples = np.random.gamma(alpha, beta, n_trials)

                elif distribution == 'beta':
                    alpha = parameters['alpha']
                    beta = parameters['beta']
                    samples = np.random.beta(alpha, beta, n_trials)

                elif distribution == 'poisson':
                    lambda_param = parameters['lambda']
                    samples = np.random.poisson(lambda_param, n_trials)

                elif distribution == 'uniform':
                    a = parameters['a']
                    b = parameters['b']
                    samples = np.random.uniform(a, b, n_trials)

                elif distribution == 'binomial':
                    n = int(parameters['n'])
                    p = parameters['p']
                    samples = np.random.binomial(n, p, n_trials)

                elif distribution == 'geometric':
                    p = parameters['p']
                    samples = np.random.geometric(p, n_trials)

                else:
                    raise ValueError(f"Distribution '{distribution}' not supported")

                all_samples.extend(samples)

                # Calculate statistics for this run
                run_stats = {
                    'run': run + 1,
                    'mean': np.mean(samples),
                    'variance': np.var(samples, ddof=1),
                    'std': np.std(samples, ddof=1),
                    'min': np.min(samples),
                    'max': np.max(samples),
                    'median': np.median(samples),
                    'q25': np.percentile(samples, 25),
                    'q75': np.percentile(samples, 75)
                }
                run_statistics.append(run_stats)

            all_samples = np.array(all_samples)

            # Overall statistics
            overall_stats = {
                'mean': np.mean(all_samples),
                'variance': np.var(all_samples, ddof=1),
                'std': np.std(all_samples, ddof=1),
                'skewness': stats.skew(all_samples),
                'kurtosis': stats.kurtosis(all_samples),
                'min': np.min(all_samples),
                'max': np.max(all_samples),
                'median': np.median(all_samples),
                'percentiles': {
                    '5%': np.percentile(all_samples, 5),
                    '25%': np.percentile(all_samples, 25),
                    '75%': np.percentile(all_samples, 75),
                    '95%': np.percentile(all_samples, 95)
                }
            }

            # Convergence analysis
            cumulative_means = np.cumsum(all_samples) / np.arange(1, len(all_samples) + 1)

            # Monte Carlo standard error
            mc_std_error = overall_stats['std'] / np.sqrt(len(all_samples))

            result = {
                'distribution': distribution,
                'parameters': parameters,
                'n_trials': n_trials,
                'n_runs': n_runs,
                'total_samples': len(all_samples),
                'samples': all_samples,
                'run_statistics': run_statistics,
                'overall_statistics': overall_stats,
                'cumulative_means': cumulative_means,
                'mc_standard_error': mc_std_error,
                'random_seed': self.random_seed,
                'simulation_date': datetime.now()
            }

            self.simulation_history.append(result)

            return result

        except Exception as e:
            return {'error': str(e), 'distribution': distribution}

    def estimate_integral(self, func: Callable, domain: Tuple[float, float],
                         n_trials: int = 100000, method: str = 'uniform') -> Dict[str, Any]:
        """
        Monte Carlo integration.

        Parameters:
        -----------
        func : Callable
            Function to integrate
        domain : Tuple
            Integration domain (a, b)
        n_trials : int
            Number of random samples
        method : str
            Sampling method ('uniform', 'importance')

        Returns:
        --------
        Dict with integral estimate and statistics
        """
        try:
            a, b = domain
            domain_length = b - a

            if method == 'uniform':
                # Uniform sampling
                x_samples = np.random.uniform(a, b, n_trials)
                y_samples = np.array([func(x) for x in x_samples])

                # Estimate integral
                integral_estimate = domain_length * np.mean(y_samples)

                # Standard error
                y_variance = np.var(y_samples, ddof=1)
                standard_error = domain_length * np.sqrt(y_variance / n_trials)

            elif method == 'importance':
                # Importance sampling (simple example with exponential distribution)
                # This is a basic implementation - in practice, you'd choose the importance
                # distribution based on the function characteristics

                # Use exponential importance distribution
                lambda_importance = 1.0
                importance_samples = np.random.exponential(1/lambda_importance, n_trials)

                # Keep only samples in domain
                valid_samples = importance_samples[(importance_samples >= a) & (importance_samples <= b)]

                if len(valid_samples) == 0:
                    raise ValueError("No valid samples in domain for importance sampling")

                # Calculate weights
                y_samples = np.array([func(x) for x in valid_samples])
                weights = 1 / (lambda_importance * np.exp(-lambda_importance * valid_samples))

                # Weighted estimate
                integral_estimate = np.mean(y_samples * weights) * (len(valid_samples) / n_trials)
                standard_error = None  # More complex calculation needed

            else:
                raise ValueError(f"Method '{method}' not supported")

            # Confidence interval
            confidence_interval = [
                integral_estimate - 1.96 * standard_error,
                integral_estimate + 1.96 * standard_error
            ] if standard_error is not None else None

            result = {
                'function_name': func.__name__ if hasattr(func, '__name__') else 'anonymous',
                'domain': domain,
                'method': method,
                'n_trials': n_trials,
                'integral_estimate': integral_estimate,
                'standard_error': standard_error,
                'confidence_interval_95': confidence_interval,
                'samples_used': len(y_samples) if method == 'uniform' else len(valid_samples),
                'simulation_date': datetime.now()
            }

            return result

        except Exception as e:
            return {'error': str(e), 'function': str(func)}

    def estimate_probability(self, event_func: Callable, simulation_func: Callable,
                           n_trials: int = 100000) -> Dict[str, Any]:
        """
        Estimate probability of an event using Monte Carlo.

        Parameters:
        -----------
        event_func : Callable
            Function that returns True if event occurs
        simulation_func : Callable
            Function that generates random samples
        n_trials : int
            Number of trials

        Returns:
        --------
        Dict with probability estimate and statistics
        """
        try:
            event_count = 0
            all_samples = []

            for _ in range(n_trials):
                sample = simulation_func()
                all_samples.append(sample)
                if event_func(sample):
                    event_count += 1

            # Probability estimate
            probability_estimate = event_count / n_trials

            # Standard error (binomial proportion)
            standard_error = np.sqrt(probability_estimate * (1 - probability_estimate) / n_trials)

            # Confidence interval
            z_critical = 1.96  # 95% confidence
            margin_error = z_critical * standard_error
            confidence_interval = [
                max(0, probability_estimate - margin_error),
                min(1, probability_estimate + margin_error)
            ]

            result = {
                'event_function': event_func.__name__ if hasattr(event_func, '__name__') else 'anonymous',
                'simulation_function': simulation_func.__name__ if hasattr(simulation_func, '__name__') else 'anonymous',
                'n_trials': n_trials,
                'event_count': event_count,
                'probability_estimate': probability_estimate,
                'standard_error': standard_error,
                'confidence_interval_95': confidence_interval,
                'samples': all_samples[:1000],  # Store first 1000 samples for analysis
                'simulation_date': datetime.now()
            }

            return result

        except Exception as e:
            return {'error': str(e)}

    def variance_reduction(self, func: Callable, n_trials: int = 10000,
                          methods: List[str] = ['antithetic', 'control_variate']) -> Dict[str, Any]:
        """
        Variance reduction techniques for Monte Carlo simulation.

        Parameters:
        -----------
        func : Callable
            Function to evaluate
        n_trials : int
            Number of trials
        methods : List[str]
            Variance reduction methods to apply

        Returns:
        --------
        Dict with results from different variance reduction methods
        """
        try:
            results = {}

            # Standard Monte Carlo
            standard_samples = np.random.uniform(0, 1, n_trials)
            standard_values = np.array([func(x) for x in standard_samples])
            standard_estimate = np.mean(standard_values)
            standard_variance = np.var(standard_values, ddof=1)

            results['standard'] = {
                'estimate': standard_estimate,
                'variance': standard_variance,
                'standard_error': np.sqrt(standard_variance / n_trials)
            }

            # Antithetic variates
            if 'antithetic' in methods:
                n_pairs = n_trials // 2
                u_samples = np.random.uniform(0, 1, n_pairs)

                # Original samples and their antithetic pairs
                values_1 = np.array([func(u) for u in u_samples])
                values_2 = np.array([func(1 - u) for u in u_samples])

                # Average of pairs
                antithetic_values = (values_1 + values_2) / 2
                antithetic_estimate = np.mean(antithetic_values)
                antithetic_variance = np.var(antithetic_values, ddof=1)

                results['antithetic'] = {
                    'estimate': antithetic_estimate,
                    'variance': antithetic_variance,
                    'standard_error': np.sqrt(antithetic_variance / n_pairs),
                    'variance_reduction_ratio': standard_variance / antithetic_variance
                }

            # Control variates (using simple linear control)
            if 'control_variate' in methods:
                # Use u as control variate (E[U] = 0.5 for uniform[0,1])
                control_samples = np.random.uniform(0, 1, n_trials)
                control_values = np.array([func(x) for x in control_samples])

                # Control variate is simply the sample itself minus its mean
                control_variate = control_samples - 0.5

                # Optimal coefficient
                covariance = np.cov(control_values, control_variate)[0, 1]
                control_variance = np.var(control_variate, ddof=1)
                optimal_c = covariance / control_variance if control_variance > 0 else 0

                # Control variate estimator
                cv_values = control_values - optimal_c * control_variate
                cv_estimate = np.mean(cv_values)
                cv_variance = np.var(cv_values, ddof=1)

                results['control_variate'] = {
                    'estimate': cv_estimate,
                    'variance': cv_variance,
                    'standard_error': np.sqrt(cv_variance / n_trials),
                    'optimal_coefficient': optimal_c,
                    'variance_reduction_ratio': standard_variance / cv_variance if cv_variance > 0 else np.inf
                }

            result = {
                'function_name': func.__name__ if hasattr(func, '__name__') else 'anonymous',
                'n_trials': n_trials,
                'methods': methods,
                'results': results,
                'simulation_date': datetime.now()
            }

            return result

        except Exception as e:
            return {'error': str(e)}


class SchellingSegregationModel:
    """
    Thomas Schelling's segregation model with satisfaction scoring.
    """

    def __init__(self, grid_size: Tuple[int, int] = (50, 50), empty_ratio: float = 0.1):
        self.grid_size = grid_size
        self.empty_ratio = empty_ratio
        self.grid = None
        self.agent_positions = {}
        self.simulation_history = []
        self.satisfaction_history = []

    def initialize_grid(self, group_ratios: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Initialize the grid with agents.

        Parameters:
        -----------
        group_ratios : Dict[str, float]
            Ratios of different groups (e.g., {'A': 0.45, 'B': 0.45})

        Returns:
        --------
        Dict with initialization results
        """
        try:
            if group_ratios is None:
                group_ratios = {'A': 0.45, 'B': 0.45}

            rows, cols = self.grid_size
            total_cells = rows * cols
            empty_cells = int(total_cells * self.empty_ratio)
            occupied_cells = total_cells - empty_cells

            # Initialize grid with empty spaces (0)
            self.grid = np.zeros(self.grid_size, dtype=object)

            # Calculate number of agents per group
            group_counts = {}
            remaining_cells = occupied_cells

            group_names = list(group_ratios.keys())
            for i, (group, ratio) in enumerate(group_ratios.items()):
                if i == len(group_names) - 1:  # Last group gets remaining cells
                    group_counts[group] = remaining_cells
                else:
                    count = int(occupied_cells * ratio)
                    group_counts[group] = count
                    remaining_cells -= count

            # Create list of all positions
            all_positions = [(i, j) for i in range(rows) for j in range(cols)]
            np.random.shuffle(all_positions)

            # Place agents
            position_idx = 0
            self.agent_positions = {}

            for group, count in group_counts.items():
                self.agent_positions[group] = []
                for _ in range(count):
                    if position_idx < len(all_positions):
                        pos = all_positions[position_idx]
                        self.grid[pos] = group
                        self.agent_positions[group].append(pos)
                        position_idx += 1

            result = {
                'grid_size': self.grid_size,
                'empty_ratio': self.empty_ratio,
                'group_ratios': group_ratios,
                'group_counts': group_counts,
                'total_agents': sum(group_counts.values()),
                'empty_cells': empty_cells,
                'initialization_date': datetime.now()
            }

            return result

        except Exception as e:
            return {'error': str(e)}

    def calculate_satisfaction(self, similarity_threshold: float = 0.3,
                             neighborhood_type: str = 'moore') -> Dict[str, Any]:
        """
        Calculate satisfaction scores for all agents.

        Parameters:
        -----------
        similarity_threshold : float
            Minimum fraction of similar neighbors required for satisfaction
        neighborhood_type : str
            Type of neighborhood ('moore' or 'von_neumann')

        Returns:
        --------
        Dict with satisfaction statistics
        """
        try:
            if self.grid is None:
                raise ValueError("Grid not initialized. Call initialize_grid() first.")

            rows, cols = self.grid_size
            satisfaction_scores = {}
            group_satisfaction = defaultdict(list)
            unsatisfied_agents = []

            def get_neighbors(pos, neighborhood_type):
                """Get neighbor positions based on neighborhood type."""
                i, j = pos
                neighbors = []

                if neighborhood_type == 'moore':
                    # Moore neighborhood (8 neighbors)
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < rows and 0 <= nj < cols:
                                neighbors.append((ni, nj))

                elif neighborhood_type == 'von_neumann':
                    # Von Neumann neighborhood (4 neighbors)
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            neighbors.append((ni, nj))

                return neighbors

            # Calculate satisfaction for each agent
            for group, positions in self.agent_positions.items():
                for pos in positions:
                    neighbors = get_neighbors(pos, neighborhood_type)

                    # Count similar and total neighbors
                    similar_neighbors = 0
                    total_neighbors = 0

                    for neighbor_pos in neighbors:
                        neighbor_group = self.grid[neighbor_pos]
                        if neighbor_group != 0:  # Not empty
                            total_neighbors += 1
                            if neighbor_group == group:
                                similar_neighbors += 1

                    # Calculate satisfaction score
                    if total_neighbors > 0:
                        satisfaction_score = similar_neighbors / total_neighbors
                    else:
                        satisfaction_score = 1.0  # Satisfied if no neighbors

                    satisfaction_scores[pos] = satisfaction_score
                    group_satisfaction[group].append(satisfaction_score)

                    # Check if agent is satisfied
                    if satisfaction_score < similarity_threshold:
                        unsatisfied_agents.append((pos, group, satisfaction_score))

            # Calculate overall statistics
            all_scores = list(satisfaction_scores.values())
            overall_stats = {
                'mean_satisfaction': np.mean(all_scores),
                'median_satisfaction': np.median(all_scores),
                'std_satisfaction': np.std(all_scores),
                'min_satisfaction': np.min(all_scores),
                'max_satisfaction': np.max(all_scores),
                'satisfaction_threshold': similarity_threshold,
                'total_agents': len(all_scores),
                'unsatisfied_count': len(unsatisfied_agents),
                'satisfaction_rate': 1 - len(unsatisfied_agents) / len(all_scores)
            }

            # Group-specific statistics
            group_stats = {}
            for group, scores in group_satisfaction.items():
                group_stats[group] = {
                    'mean_satisfaction': np.mean(scores),
                    'median_satisfaction': np.median(scores),
                    'std_satisfaction': np.std(scores),
                    'count': len(scores),
                    'unsatisfied_count': sum(1 for pos, g, _ in unsatisfied_agents if g == group),
                    'satisfaction_rate': 1 - sum(1 for pos, g, _ in unsatisfied_agents if g == group) / len(scores)
                }

            result = {
                'similarity_threshold': similarity_threshold,
                'neighborhood_type': neighborhood_type,
                'satisfaction_scores': satisfaction_scores,
                'overall_statistics': overall_stats,
                'group_statistics': group_stats,
                'unsatisfied_agents': unsatisfied_agents,
                'calculation_date': datetime.now()
            }

            self.satisfaction_history.append(result)

            return result

        except Exception as e:
            return {'error': str(e)}

    def simulate_step(self, similarity_threshold: float = 0.3,
                     neighborhood_type: str = 'moore', max_moves: int = None) -> Dict[str, Any]:
        """
        Simulate one step of the Schelling model.

        Parameters:
        -----------
        similarity_threshold : float
            Satisfaction threshold
        neighborhood_type : str
            Neighborhood type
        max_moves : int
            Maximum number of moves per step

        Returns:
        --------
        Dict with step results
        """
        try:
            # Calculate current satisfaction
            satisfaction_result = self.calculate_satisfaction(similarity_threshold, neighborhood_type)
            unsatisfied_agents = satisfaction_result['unsatisfied_agents']

            if not unsatisfied_agents:
                return {
                    'step_type': 'simulation_step',
                    'moves_made': 0,
                    'unsatisfied_count': 0,
                    'satisfaction_result': satisfaction_result,
                    'converged': True
                }

            # Shuffle unsatisfied agents for random order of moves
            np.random.shuffle(unsatisfied_agents)

            # Limit moves if specified
            if max_moves is not None:
                unsatisfied_agents = unsatisfied_agents[:max_moves]

            # Find empty positions
            empty_positions = []
            rows, cols = self.grid_size
            for i in range(rows):
                for j in range(cols):
                    if self.grid[i, j] == 0:
                        empty_positions.append((i, j))

            moves_made = 0
            moved_agents = []

            # Move unsatisfied agents
            for pos, group, current_satisfaction in unsatisfied_agents:
                if not empty_positions:
                    break

                # Choose random empty position
                new_pos_idx = np.random.randint(len(empty_positions))
                new_pos = empty_positions[new_pos_idx]

                # Move agent
                self.grid[pos] = 0  # Empty old position
                self.grid[new_pos] = group  # Place agent in new position

                # Update agent positions
                self.agent_positions[group].remove(pos)
                self.agent_positions[group].append(new_pos)

                # Update empty positions
                empty_positions.remove(new_pos)
                empty_positions.append(pos)

                moved_agents.append({
                    'group': group,
                    'from': pos,
                    'to': new_pos,
                    'old_satisfaction': current_satisfaction
                })

                moves_made += 1

            # Recalculate satisfaction after moves
            new_satisfaction_result = self.calculate_satisfaction(similarity_threshold, neighborhood_type)

            result = {
                'step_type': 'simulation_step',
                'moves_made': moves_made,
                'moved_agents': moved_agents,
                'unsatisfied_count_before': len(unsatisfied_agents),
                'unsatisfied_count_after': new_satisfaction_result['overall_statistics']['unsatisfied_count'],
                'satisfaction_before': satisfaction_result['overall_statistics']['satisfaction_rate'],
                'satisfaction_after': new_satisfaction_result['overall_statistics']['satisfaction_rate'],
                'satisfaction_result': new_satisfaction_result,
                'converged': new_satisfaction_result['overall_statistics']['unsatisfied_count'] == 0,
                'step_date': datetime.now()
            }

            return result

        except Exception as e:
            return {'error': str(e)}

    def run_simulation(self, similarity_threshold: float = 0.3,
                      neighborhood_type: str = 'moore', max_steps: int = 100,
                      max_moves_per_step: int = None) -> Dict[str, Any]:
        """
        Run complete Schelling segregation simulation.

        Parameters:
        -----------
        similarity_threshold : float
            Satisfaction threshold
        neighborhood_type : str
            Neighborhood type
        max_steps : int
            Maximum simulation steps
        max_moves_per_step : int
            Maximum moves per step

        Returns:
        --------
        Dict with complete simulation results
        """
        try:
            if self.grid is None:
                raise ValueError("Grid not initialized. Call initialize_grid() first.")

            simulation_steps = []
            satisfaction_over_time = []
            moves_over_time = []

            for step in range(max_steps):
                step_result = self.simulate_step(similarity_threshold, neighborhood_type, max_moves_per_step)
                simulation_steps.append(step_result)

                satisfaction_over_time.append(step_result['satisfaction_result']['overall_statistics']['satisfaction_rate'])
                moves_over_time.append(step_result['moves_made'])

                # Check convergence
                if step_result['converged']:
                    break

            # Calculate segregation metrics
            final_satisfaction = self.calculate_satisfaction(similarity_threshold, neighborhood_type)
            segregation_metrics = self._calculate_segregation_metrics()

            result = {
                'simulation_parameters': {
                    'similarity_threshold': similarity_threshold,
                    'neighborhood_type': neighborhood_type,
                    'max_steps': max_steps,
                    'max_moves_per_step': max_moves_per_step
                },
                'steps_completed': len(simulation_steps),
                'converged': simulation_steps[-1]['converged'] if simulation_steps else False,
                'simulation_steps': simulation_steps,
                'satisfaction_over_time': satisfaction_over_time,
                'moves_over_time': moves_over_time,
                'total_moves': sum(moves_over_time),
                'final_satisfaction': final_satisfaction,
                'segregation_metrics': segregation_metrics,
                'simulation_date': datetime.now()
            }

            self.simulation_history.append(result)

            return result

        except Exception as e:
            return {'error': str(e)}

    def _calculate_segregation_metrics(self) -> Dict[str, Any]:
        """Calculate various segregation metrics."""
        try:
            if self.grid is None:
                return {'error': 'Grid not initialized'}

            rows, cols = self.grid_size
            groups = list(self.agent_positions.keys())

            # Calculate exposure indices
            exposure_indices = {}
            for group1 in groups:
                exposure_indices[group1] = {}
                for group2 in groups:
                    total_exposure = 0
                    total_agents = 0

                    for pos in self.agent_positions[group1]:
                        neighbors = self._get_moore_neighbors(pos)
                        group2_neighbors = sum(1 for n_pos in neighbors if self.grid[n_pos] == group2)
                        total_neighbors = sum(1 for n_pos in neighbors if self.grid[n_pos] != 0)

                        if total_neighbors > 0:
                            total_exposure += group2_neighbors / total_neighbors
                        total_agents += 1

                    if total_agents > 0:
                        exposure_indices[group1][group2] = total_exposure / total_agents
                    else:
                        exposure_indices[group1][group2] = 0

            # Calculate isolation index (exposure to own group)
            isolation_indices = {}
            for group in groups:
                isolation_indices[group] = exposure_indices[group].get(group, 0)

            # Calculate dissimilarity index (simplified version)
            if len(groups) == 2:
                group1, group2 = groups
                count1 = len(self.agent_positions[group1])
                count2 = len(self.agent_positions[group2])
                total = count1 + count2

                if total > 0:
                    # Simplified dissimilarity based on spatial distribution
                    # This is a basic implementation - full calculation would require
                    # proper spatial unit definitions
                    proportion1 = count1 / total
                    proportion2 = count2 / total

                    # Use grid quadrants as spatial units
                    mid_row, mid_col = rows // 2, cols // 2
                    quadrants = [
                        (0, mid_row, 0, mid_col),
                        (0, mid_row, mid_col, cols),
                        (mid_row, rows, 0, mid_col),
                        (mid_row, rows, mid_col, cols)
                    ]

                    dissimilarity = 0
                    for r1, r2, c1, c2 in quadrants:
                        quad_count1 = sum(1 for pos in self.agent_positions[group1]
                                        if r1 <= pos[0] < r2 and c1 <= pos[1] < c2)
                        quad_count2 = sum(1 for pos in self.agent_positions[group2]
                                        if r1 <= pos[0] < r2 and c1 <= pos[1] < c2)
                        quad_total = quad_count1 + quad_count2

                        if quad_total > 0:
                            quad_prop1 = quad_count1 / quad_total
                            dissimilarity += abs(quad_prop1 - proportion1)

                    dissimilarity = dissimilarity / (2 * len(quadrants))
                else:
                    dissimilarity = 0
            else:
                dissimilarity = None

            return {
                'exposure_indices': exposure_indices,
                'isolation_indices': isolation_indices,
                'dissimilarity_index': dissimilarity,
                'groups': groups,
                'group_counts': {group: len(positions) for group, positions in self.agent_positions.items()}
            }

        except Exception as e:
            return {'error': str(e)}

    def _get_moore_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get Moore neighborhood positions."""
        i, j = pos
        rows, cols = self.grid_size
        neighbors = []

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbors.append((ni, nj))

        return neighbors

    def visualize_grid(self, save_path: str = None, show_satisfaction: bool = False) -> Dict[str, Any]:
        """
        Visualize the current grid state.

        Parameters:
        -----------
        save_path : str
            Path to save the visualization
        show_satisfaction : bool
            Whether to show satisfaction scores

        Returns:
        --------
        Dict with visualization info
        """
        try:
            if self.grid is None:
                raise ValueError("Grid not initialized")

            plt.figure(figsize=(10, 10))

            # Create color map for groups
            groups = list(self.agent_positions.keys())
            colors = plt.cm.Set1(np.linspace(0, 1, len(groups)))
            color_map = {group: color for group, color in zip(groups, colors)}
            color_map[0] = [1, 1, 1, 1]  # White for empty spaces

            # Create colored grid
            colored_grid = np.zeros((*self.grid_size, 4))
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    cell_value = self.grid[i, j]
                    colored_grid[i, j] = color_map[cell_value]

            plt.imshow(colored_grid)
            plt.title('Schelling Segregation Model')

            # Add legend
            legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color_map[group],
                                           label=f'Group {group}') for group in groups]
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='white',
                                               edgecolor='black', label='Empty'))
            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

            plt.axis('off')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            result = {
                'visualization_type': 'grid_state',
                'grid_size': self.grid_size,
                'groups': groups,
                'save_path': save_path,
                'visualization_date': datetime.now()
            }

            return result

        except Exception as e:
            return {'error': str(e)}


# Agent-Based Modeling Framework

class AgentType(Enum):
    """Enumeration of different agent types."""
    BASIC = "basic"
    LEARNING = "learning"
    SOCIAL = "social"
    ECONOMIC = "economic"
    SPATIAL = "spatial"


@dataclass
class AgentCharacteristics:
    """Data class for agent characteristics."""
    agent_id: str
    agent_type: AgentType
    position: Optional[Tuple[float, float]] = None
    wealth: float = 0.0
    health: float = 1.0
    age: int = 0
    memory: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, float] = field(default_factory=dict)
    social_connections: List[str] = field(default_factory=list)
    learning_rate: float = 0.1
    custom_attributes: Dict[str, Any] = field(default_factory=dict)


class NetworkTopology(Enum):
    """Network topology types."""
    RANDOM = "random"
    SMALL_WORLD = "small_world"
    SCALE_FREE = "scale_free"
    LATTICE = "lattice"
    COMPLETE = "complete"
    RING = "ring"


class Environment:
    """Environment class for agent-based models."""

    def __init__(self, size: Tuple[float, float] = (100, 100),
                 resources: Dict[str, float] = None,
                 boundaries: str = "periodic"):
        self.size = size
        self.resources = resources or {}
        self.boundaries = boundaries
        self.time_step = 0
        self.history = []
        self.spatial_grid = {}

    def update_resources(self, resource_updates: Dict[str, float]):
        """Update environment resources."""
        for resource, change in resource_updates.items():
            if resource in self.resources:
                self.resources[resource] += change
            else:
                self.resources[resource] = change

    def get_neighbors(self, position: Tuple[float, float],
                     radius: float) -> List[Tuple[float, float]]:
        """Get neighboring positions within radius."""
        x, y = position
        neighbors = []

        for pos in self.spatial_grid.keys():
            px, py = pos
            distance = np.sqrt((x - px)**2 + (y - py)**2)
            if 0 < distance <= radius:
                neighbors.append(pos)

        return neighbors

    def add_agent(self, agent_id: str, position: Tuple[float, float]):
        """Add agent to spatial grid."""
        self.spatial_grid[position] = agent_id

    def remove_agent(self, position: Tuple[float, float]):
        """Remove agent from spatial grid."""
        if position in self.spatial_grid:
            del self.spatial_grid[position]

    def move_agent(self, old_position: Tuple[float, float],
                  new_position: Tuple[float, float]):
        """Move agent to new position."""
        if old_position in self.spatial_grid:
            agent_id = self.spatial_grid[old_position]
            del self.spatial_grid[old_position]
            self.spatial_grid[new_position] = agent_id

    def step(self):
        """Environment step - update time and resources."""
        self.time_step += 1

        # Resource regeneration (example)
        for resource in self.resources:
            if resource.endswith('_renewable'):
                base_resource = resource.replace('_renewable', '')
                if base_resource in self.resources:
                    self.resources[base_resource] = min(
                        self.resources[base_resource] * 1.01,  # 1% growth
                        1000  # Maximum capacity
                    )


class Agent:
    """Base agent class for agent-based modeling."""

    def __init__(self, characteristics: AgentCharacteristics, environment: Environment):
        self.characteristics = characteristics
        self.environment = environment
        self.state_history = []
        self.decision_history = []

    def perceive(self) -> Dict[str, Any]:
        """Perceive environment and other agents."""
        perception = {
            'environment_resources': self.environment.resources.copy(),
            'time_step': self.environment.time_step,
            'local_agents': [],
            'position': self.characteristics.position
        }

        # Perceive nearby agents if position is set
        if self.characteristics.position:
            nearby_positions = self.environment.get_neighbors(
                self.characteristics.position, radius=10.0)
            perception['local_agents'] = [
                self.environment.spatial_grid.get(pos) for pos in nearby_positions
                if self.environment.spatial_grid.get(pos) != self.characteristics.agent_id
            ]

        return perception

    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Make decisions based on perception."""
        decision = {
            'action_type': 'stay',
            'parameters': {},
            'reasoning': 'Default behavior - no action'
        }

        # Basic decision-making can be overridden in subclasses
        if self.characteristics.agent_type == AgentType.BASIC:
            # Random movement
            if np.random.random() < 0.1:  # 10% chance to move
                if self.characteristics.position:
                    x, y = self.characteristics.position
                    new_x = x + np.random.normal(0, 1)
                    new_y = y + np.random.normal(0, 1)

                    # Apply boundaries
                    if self.environment.boundaries == "periodic":
                        new_x = new_x % self.environment.size[0]
                        new_y = new_y % self.environment.size[1]
                    else:  # Reflective boundaries
                        new_x = max(0, min(new_x, self.environment.size[0]))
                        new_y = max(0, min(new_y, self.environment.size[1]))

                    decision = {
                        'action_type': 'move',
                        'parameters': {'new_position': (new_x, new_y)},
                        'reasoning': 'Random movement'
                    }

        self.decision_history.append(decision)
        return decision

    def act(self, decision: Dict[str, Any]):
        """Execute decision."""
        if decision['action_type'] == 'move' and 'new_position' in decision['parameters']:
            old_position = self.characteristics.position
            new_position = decision['parameters']['new_position']

            self.characteristics.position = new_position

            if old_position:
                self.environment.move_agent(old_position, new_position)
            else:
                self.environment.add_agent(self.characteristics.agent_id, new_position)

        elif decision['action_type'] == 'consume_resource':
            resource = decision['parameters'].get('resource')
            amount = decision['parameters'].get('amount', 0)

            if resource in self.environment.resources:
                available = self.environment.resources[resource]
                consumed = min(amount, available)
                self.environment.resources[resource] -= consumed

                # Update agent based on consumption
                if resource == 'food':
                    self.characteristics.health = min(1.0, self.characteristics.health + consumed * 0.1)

    def update_state(self, perception: Dict[str, Any], decision: Dict[str, Any]):
        """Update internal state."""
        # Age the agent
        self.characteristics.age += 1

        # Natural health decay
        self.characteristics.health = max(0, self.characteristics.health - 0.001)

        # Update memory with recent experiences
        if len(self.characteristics.memory) > 100:  # Limit memory size
            # Remove oldest memory entries
            oldest_key = min(self.characteristics.memory.keys())
            del self.characteristics.memory[oldest_key]

        self.characteristics.memory[str(self.environment.time_step)] = {
            'perception': perception,
            'decision': decision,
            'health': self.characteristics.health,
            'wealth': self.characteristics.wealth
        }

        # Save state history
        state = {
            'time_step': self.environment.time_step,
            'position': self.characteristics.position,
            'health': self.characteristics.health,
            'wealth': self.characteristics.wealth,
            'age': self.characteristics.age
        }
        self.state_history.append(state)

    def step(self):
        """Execute one agent step."""
        perception = self.perceive()
        decision = self.decide(perception)
        self.act(decision)
        self.update_state(perception, decision)


class LearningAgent(Agent):
    """Agent with learning capabilities."""

    def __init__(self, characteristics: AgentCharacteristics, environment: Environment):
        super().__init__(characteristics, environment)
        self.q_table = defaultdict(lambda: defaultdict(float))  # Simple Q-learning
        self.action_counts = defaultdict(lambda: defaultdict(int))
        self.epsilon = 0.1  # Exploration rate

    def get_state_key(self, perception: Dict[str, Any]) -> str:
        """Convert perception to state key for learning."""
        # Simplified state representation
        nearby_agents = len(perception.get('local_agents', []))
        resource_level = sum(perception.get('environment_resources', {}).values())

        # Discretize continuous values
        nearby_bucket = min(nearby_agents // 2, 5)  # 0-5
        resource_bucket = min(int(resource_level // 100), 10)  # 0-10

        return f"nearby_{nearby_bucket}_resources_{resource_bucket}"

    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision using Q-learning."""
        state_key = self.get_state_key(perception)

        available_actions = ['stay', 'move', 'consume_food']

        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.choice(available_actions)
        else:
            # Exploit: best known action
            q_values = [self.q_table[state_key][action] for action in available_actions]
            best_action_idx = np.argmax(q_values)
            action = available_actions[best_action_idx]

        # Convert action to decision format
        if action == 'move':
            if self.characteristics.position:
                x, y = self.characteristics.position
                new_x = x + np.random.normal(0, 2)
                new_y = y + np.random.normal(0, 2)

                # Apply boundaries
                if self.environment.boundaries == "periodic":
                    new_x = new_x % self.environment.size[0]
                    new_y = new_y % self.environment.size[1]
                else:
                    new_x = max(0, min(new_x, self.environment.size[0]))
                    new_y = max(0, min(new_y, self.environment.size[1]))

                decision = {
                    'action_type': 'move',
                    'parameters': {'new_position': (new_x, new_y)},
                    'reasoning': f'Q-learning action: {action}',
                    'q_action': action,
                    'state_key': state_key
                }
            else:
                decision = {
                    'action_type': 'stay',
                    'parameters': {},
                    'reasoning': 'Cannot move without position',
                    'q_action': 'stay',
                    'state_key': state_key
                }

        elif action == 'consume_food':
            decision = {
                'action_type': 'consume_resource',
                'parameters': {'resource': 'food', 'amount': 1.0},
                'reasoning': f'Q-learning action: {action}',
                'q_action': action,
                'state_key': state_key
            }

        else:  # stay
            decision = {
                'action_type': 'stay',
                'parameters': {},
                'reasoning': f'Q-learning action: {action}',
                'q_action': action,
                'state_key': state_key
            }

        self.action_counts[state_key][action] += 1
        self.decision_history.append(decision)

        return decision

    def update_q_value(self, reward: float):
        """Update Q-value based on received reward."""
        if len(self.decision_history) < 2:
            return

        # Get previous decision
        prev_decision = self.decision_history[-2]
        current_decision = self.decision_history[-1]

        if 'state_key' in prev_decision and 'q_action' in prev_decision:
            prev_state = prev_decision['state_key']
            prev_action = prev_decision['q_action']

            # Current state
            current_state = current_decision.get('state_key')

            # Q-learning update
            alpha = self.characteristics.learning_rate
            gamma = 0.9  # Discount factor

            old_q = self.q_table[prev_state][prev_action]

            if current_state:
                # Find maximum Q-value for current state
                current_q_values = list(self.q_table[current_state].values())
                max_future_q = max(current_q_values) if current_q_values else 0
            else:
                max_future_q = 0

            # Q-learning update rule
            new_q = old_q + alpha * (reward + gamma * max_future_q - old_q)
            self.q_table[prev_state][prev_action] = new_q

    def calculate_reward(self) -> float:
        """Calculate reward based on agent's current state."""
        reward = 0

        # Health-based reward
        reward += self.characteristics.health * 10

        # Wealth-based reward
        reward += self.characteristics.wealth * 0.1

        # Survival reward
        if self.characteristics.health > 0:
            reward += 1

        # Social reward (if has social connections)
        reward += len(self.characteristics.social_connections) * 0.5

        return reward

    def update_state(self, perception: Dict[str, Any], decision: Dict[str, Any]):
        """Update state and perform learning."""
        super().update_state(perception, decision)

        # Calculate reward and update Q-values
        reward = self.calculate_reward()
        self.update_q_value(reward)

        # Decay exploration rate
        self.epsilon = max(0.01, self.epsilon * 0.999)


class SocialAgent(Agent):
    """Agent with social interaction capabilities."""

    def __init__(self, characteristics: AgentCharacteristics, environment: Environment):
        super().__init__(characteristics, environment)
        self.influence_history = []
        self.trust_network = {}
        self.social_memory = {}

    def interact_with_agent(self, other_agent_id: str,
                           interaction_type: str) -> Dict[str, Any]:
        """Interact with another agent."""
        interaction_result = {
            'other_agent': other_agent_id,
            'interaction_type': interaction_type,
            'time_step': self.environment.time_step,
            'success': False,
            'outcome': {}
        }

        if interaction_type == 'information_exchange':
            # Simple information exchange
            if other_agent_id not in self.social_memory:
                self.social_memory[other_agent_id] = {}

            self.social_memory[other_agent_id]['last_contact'] = self.environment.time_step
            interaction_result['success'] = True
            interaction_result['outcome'] = {'information_gained': True}

        elif interaction_type == 'cooperation':
            # Cooperation interaction - both agents benefit
            cooperation_benefit = 0.1
            self.characteristics.wealth += cooperation_benefit

            # Update trust
            if other_agent_id not in self.trust_network:
                self.trust_network[other_agent_id] = 0.5  # Neutral trust

            self.trust_network[other_agent_id] = min(1.0,
                                                   self.trust_network[other_agent_id] + 0.1)

            interaction_result['success'] = True
            interaction_result['outcome'] = {
                'wealth_gain': cooperation_benefit,
                'trust_change': 0.1
            }

        elif interaction_type == 'competition':
            # Competition - winner takes resources
            win_probability = 0.5  # Simplified
            if np.random.random() < win_probability:
                resource_gain = 0.2
                self.characteristics.wealth += resource_gain
                interaction_result['outcome'] = {'won': True, 'resource_gain': resource_gain}
            else:
                resource_loss = 0.1
                self.characteristics.wealth = max(0, self.characteristics.wealth - resource_loss)
                interaction_result['outcome'] = {'won': False, 'resource_loss': resource_loss}

            interaction_result['success'] = True

        self.influence_history.append(interaction_result)
        return interaction_result

    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Make social decisions."""
        local_agents = perception.get('local_agents', [])

        if local_agents and np.random.random() < 0.3:  # 30% chance to interact
            # Choose agent to interact with
            target_agent = np.random.choice(local_agents)

            # Choose interaction type based on trust
            trust_level = self.trust_network.get(target_agent, 0.5)

            if trust_level > 0.7:
                interaction_type = 'cooperation'
            elif trust_level < 0.3:
                interaction_type = 'competition'
            else:
                interaction_type = 'information_exchange'

            decision = {
                'action_type': 'social_interaction',
                'parameters': {
                    'target_agent': target_agent,
                    'interaction_type': interaction_type
                },
                'reasoning': f'Social interaction with trust level {trust_level:.2f}'
            }
        else:
            # Fall back to parent class decision
            decision = super().decide(perception)

        return decision

    def act(self, decision: Dict[str, Any]):
        """Execute social actions."""
        if decision['action_type'] == 'social_interaction':
            target_agent = decision['parameters']['target_agent']
            interaction_type = decision['parameters']['interaction_type']
            self.interact_with_agent(target_agent, interaction_type)
        else:
            super().act(decision)


class AgentBasedModel:
    """
    Comprehensive Agent-Based Modeling framework.
    """

    def __init__(self, environment: Environment):
        self.environment = environment
        self.agents = {}
        self.simulation_data = []
        self.network = None
        self.model_parameters = {}

    def create_network(self, topology: NetworkTopology, network_params: Dict[str, Any] = None) -> nx.Graph:
        """
        Create interaction network between agents.

        Parameters:
        -----------
        topology : NetworkTopology
            Type of network topology
        network_params : Dict
            Parameters for network generation

        Returns:
        --------
        NetworkX graph object
        """
        try:
            if network_params is None:
                network_params = {}

            agent_ids = list(self.agents.keys())
            n_agents = len(agent_ids)

            if n_agents == 0:
                return nx.Graph()

            if topology == NetworkTopology.RANDOM:
                p = network_params.get('p', 0.1)  # Connection probability
                G = nx.erdos_renyi_graph(n_agents, p)

            elif topology == NetworkTopology.SMALL_WORLD:
                k = network_params.get('k', 4)  # Each node connected to k nearest neighbors
                p = network_params.get('p', 0.1)  # Rewiring probability
                G = nx.watts_strogatz_graph(n_agents, k, p)

            elif topology == NetworkTopology.SCALE_FREE:
                m = network_params.get('m', 2)  # Number of edges for new nodes
                G = nx.barabasi_albert_graph(n_agents, m)

            elif topology == NetworkTopology.LATTICE:
                rows = int(np.sqrt(n_agents))
                cols = n_agents // rows
                G = nx.grid_2d_graph(rows, cols)
                # Relabel nodes to use agent indices
                mapping = {node: i for i, node in enumerate(G.nodes())}
                G = nx.relabel_nodes(G, mapping)

            elif topology == NetworkTopology.COMPLETE:
                G = nx.complete_graph(n_agents)

            elif topology == NetworkTopology.RING:
                G = nx.cycle_graph(n_agents)

            else:
                raise ValueError(f"Topology {topology} not supported")

            # Relabel nodes with agent IDs
            node_mapping = {i: agent_id for i, agent_id in enumerate(agent_ids)}
            G = nx.relabel_nodes(G, node_mapping)

            self.network = G
            return G

        except Exception as e:
            print(f"Error creating network: {e}")
            return nx.Graph()

    def add_agent(self, agent: Agent) -> str:
        """Add agent to the model."""
        agent_id = agent.characteristics.agent_id
        self.agents[agent_id] = agent

        # Add to spatial environment if position is set
        if agent.characteristics.position:
            self.environment.add_agent(agent_id, agent.characteristics.position)

        return agent_id

    def remove_agent(self, agent_id: str) -> bool:
        """Remove agent from the model."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]

            # Remove from spatial environment
            if agent.characteristics.position:
                self.environment.remove_agent(agent.characteristics.position)

            # Remove from network
            if self.network and agent_id in self.network:
                self.network.remove_node(agent_id)

            del self.agents[agent_id]
            return True

        return False

    def step(self) -> Dict[str, Any]:
        """Execute one simulation step."""
        try:
            step_data = {
                'time_step': self.environment.time_step,
                'agent_states': {},
                'environment_state': {
                    'resources': self.environment.resources.copy(),
                    'spatial_occupancy': len(self.environment.spatial_grid)
                },
                'interactions': [],
                'step_summary': {}
            }

            # Step all agents
            for agent_id, agent in self.agents.items():
                agent.step()

                # Record agent state
                step_data['agent_states'][agent_id] = {
                    'position': agent.characteristics.position,
                    'health': agent.characteristics.health,
                    'wealth': agent.characteristics.wealth,
                    'age': agent.characteristics.age,
                    'connections': len(agent.characteristics.social_connections)
                }

                # Record interactions if it's a social agent
                if isinstance(agent, SocialAgent) and agent.influence_history:
                    recent_interactions = [
                        interaction for interaction in agent.influence_history
                        if interaction['time_step'] == self.environment.time_step
                    ]
                    step_data['interactions'].extend(recent_interactions)

            # Step environment
            self.environment.step()

            # Calculate step summary statistics
            if self.agents:
                health_values = [agent.characteristics.health for agent in self.agents.values()]
                wealth_values = [agent.characteristics.wealth for agent in self.agents.values()]
                age_values = [agent.characteristics.age for agent in self.agents.values()]

                step_data['step_summary'] = {
                    'total_agents': len(self.agents),
                    'avg_health': np.mean(health_values),
                    'avg_wealth': np.mean(wealth_values),
                    'avg_age': np.mean(age_values),
                    'alive_agents': sum(1 for h in health_values if h > 0),
                    'total_interactions': len(step_data['interactions'])
                }

            self.simulation_data.append(step_data)
            return step_data

        except Exception as e:
            return {'error': str(e), 'time_step': self.environment.time_step}

    def run_simulation(self, n_steps: int, save_interval: int = 100) -> Dict[str, Any]:
        """
        Run complete simulation.

        Parameters:
        -----------
        n_steps : int
            Number of simulation steps
        save_interval : int
            Interval for saving simulation state

        Returns:
        --------
        Dict with simulation results and analysis
        """
        try:
            simulation_start = datetime.now()

            for step in range(n_steps):
                step_result = self.step()

                if 'error' in step_result:
                    break

                # Periodic cleanup and analysis
                if step % save_interval == 0:
                    print(f"Step {step}/{n_steps} completed")

            # Analyze simulation results
            analysis = self.analyze_simulation()

            result = {
                'simulation_parameters': {
                    'n_steps': n_steps,
                    'save_interval': save_interval,
                    'total_agents': len(self.agents),
                    'environment_size': self.environment.size
                },
                'steps_completed': len(self.simulation_data),
                'simulation_data': self.simulation_data,
                'final_analysis': analysis,
                'simulation_duration': datetime.now() - simulation_start,
                'simulation_date': simulation_start
            }

            return result

        except Exception as e:
            return {'error': str(e)}

    def analyze_simulation(self) -> Dict[str, Any]:
        """Analyze simulation results and compute metrics."""
        try:
            if not self.simulation_data:
                return {'error': 'No simulation data available'}

            analysis = {
                'time_series': {
                    'health': [],
                    'wealth': [],
                    'population': [],
                    'interactions': []
                },
                'final_state': {},
                'network_metrics': {},
                'agent_trajectories': {}
            }

            # Extract time series data
            for step_data in self.simulation_data:
                summary = step_data.get('step_summary', {})

                analysis['time_series']['health'].append(summary.get('avg_health', 0))
                analysis['time_series']['wealth'].append(summary.get('avg_wealth', 0))
                analysis['time_series']['population'].append(summary.get('alive_agents', 0))
                analysis['time_series']['interactions'].append(summary.get('total_interactions', 0))

            # Analyze final state
            if self.simulation_data:
                final_step = self.simulation_data[-1]
                analysis['final_state'] = final_step.get('step_summary', {})

            # Network analysis
            if self.network:
                analysis['network_metrics'] = {
                    'nodes': self.network.number_of_nodes(),
                    'edges': self.network.number_of_edges(),
                    'density': nx.density(self.network),
                    'clustering': nx.average_clustering(self.network),
                    'path_length': nx.average_shortest_path_length(self.network) if nx.is_connected(self.network) else None,
                    'components': nx.number_connected_components(self.network)
                }

                # Centrality measures
                try:
                    analysis['network_metrics']['degree_centrality'] = nx.degree_centrality(self.network)
                    analysis['network_metrics']['betweenness_centrality'] = nx.betweenness_centrality(self.network)
                    analysis['network_metrics']['closeness_centrality'] = nx.closeness_centrality(self.network)
                except:
                    pass  # Skip if network is too small or disconnected

            # Individual agent trajectories
            for agent_id, agent in self.agents.items():
                if agent.state_history:
                    trajectory = {
                        'health_trajectory': [state['health'] for state in agent.state_history],
                        'wealth_trajectory': [state['wealth'] for state in agent.state_history],
                        'position_trajectory': [state['position'] for state in agent.state_history],
                        'final_age': agent.characteristics.age,
                        'total_decisions': len(agent.decision_history)
                    }

                    # Learning agent specific metrics
                    if isinstance(agent, LearningAgent):
                        trajectory['exploration_rate'] = agent.epsilon
                        trajectory['q_table_size'] = len(agent.q_table)

                    # Social agent specific metrics
                    if isinstance(agent, SocialAgent):
                        trajectory['total_interactions'] = len(agent.influence_history)
                        trajectory['trust_network_size'] = len(agent.trust_network)
                        trajectory['avg_trust'] = np.mean(list(agent.trust_network.values())) if agent.trust_network else 0

                    analysis['agent_trajectories'][agent_id] = trajectory

            return analysis

        except Exception as e:
            return {'error': str(e)}


class ModelValidation:
    """
    Model verification and validation framework.
    """

    def __init__(self, model: AgentBasedModel):
        self.model = model
        self.validation_results = {}

    def verify_internal_consistency(self) -> Dict[str, Any]:
        """
        Internal verification checks.

        Returns:
        --------
        Dict with verification results
        """
        try:
            verification = {
                'checks_passed': 0,
                'checks_failed': 0,
                'detailed_results': {},
                'overall_status': 'UNKNOWN'
            }

            # Check 1: Agent conservation
            agent_count_model = len(self.model.agents)
            agent_count_env = len(self.model.environment.spatial_grid)

            agents_with_position = sum(1 for agent in self.model.agents.values()
                                     if agent.characteristics.position is not None)

            if agents_with_position == agent_count_env:
                verification['checks_passed'] += 1
                verification['detailed_results']['agent_conservation'] = 'PASS'
            else:
                verification['checks_failed'] += 1
                verification['detailed_results']['agent_conservation'] = f'FAIL: {agents_with_position} positioned agents vs {agent_count_env} in environment'

            # Check 2: Resource conservation (if applicable)
            total_resources = sum(self.model.environment.resources.values())
            if total_resources >= 0:  # Resources should be non-negative
                verification['checks_passed'] += 1
                verification['detailed_results']['resource_conservation'] = 'PASS'
            else:
                verification['checks_failed'] += 1
                verification['detailed_results']['resource_conservation'] = f'FAIL: Negative total resources: {total_resources}'

            # Check 3: Agent health bounds
            health_values = [agent.characteristics.health for agent in self.model.agents.values()]
            valid_health = all(0 <= h <= 1 for h in health_values)

            if valid_health:
                verification['checks_passed'] += 1
                verification['detailed_results']['health_bounds'] = 'PASS'
            else:
                invalid_count = sum(1 for h in health_values if not (0 <= h <= 1))
                verification['checks_failed'] += 1
                verification['detailed_results']['health_bounds'] = f'FAIL: {invalid_count} agents with invalid health'

            # Check 4: Network consistency
            if self.model.network:
                network_nodes = set(self.model.network.nodes())
                agent_ids = set(self.model.agents.keys())

                if network_nodes == agent_ids:
                    verification['checks_passed'] += 1
                    verification['detailed_results']['network_consistency'] = 'PASS'
                else:
                    verification['checks_failed'] += 1
                    missing_agents = agent_ids - network_nodes
                    extra_nodes = network_nodes - agent_ids
                    verification['detailed_results']['network_consistency'] = f'FAIL: Missing agents: {missing_agents}, Extra nodes: {extra_nodes}'

            # Check 5: Agent decision history consistency
            decision_consistency = True
            for agent_id, agent in self.model.agents.items():
                expected_decisions = self.model.environment.time_step
                actual_decisions = len(agent.decision_history)

                if actual_decisions > expected_decisions:
                    decision_consistency = False
                    break

            if decision_consistency:
                verification['checks_passed'] += 1
                verification['detailed_results']['decision_consistency'] = 'PASS'
            else:
                verification['checks_failed'] += 1
                verification['detailed_results']['decision_consistency'] = 'FAIL: Decision count exceeds time steps'

            # Overall status
            total_checks = verification['checks_passed'] + verification['checks_failed']
            if verification['checks_failed'] == 0:
                verification['overall_status'] = 'PASS'
            elif verification['checks_passed'] / total_checks >= 0.8:
                verification['overall_status'] = 'WARNING'
            else:
                verification['overall_status'] = 'FAIL'

            verification['verification_date'] = datetime.now()
            self.validation_results['internal_verification'] = verification

            return verification

        except Exception as e:
            return {'error': str(e), 'verification_type': 'internal'}

    def validate_external_behavior(self, expected_patterns: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        External validation against expected patterns.

        Parameters:
        -----------
        expected_patterns : Dict
            Expected behavioral patterns to validate against

        Returns:
        --------
        Dict with validation results
        """
        try:
            if not self.model.simulation_data:
                return {'error': 'No simulation data available for validation'}

            validation = {
                'tests_passed': 0,
                'tests_failed': 0,
                'detailed_results': {},
                'overall_status': 'UNKNOWN'
            }

            if expected_patterns is None:
                expected_patterns = {
                    'population_stability': {'min_survival_rate': 0.1},
                    'health_decay': {'should_decrease': True},
                    'wealth_accumulation': {'some_agents_gain': True},
                    'spatial_distribution': {'agents_should_move': True}
                }

            # Test 1: Population stability
            if 'population_stability' in expected_patterns:
                initial_pop = self.model.simulation_data[0]['step_summary'].get('total_agents', 0)
                final_pop = self.model.simulation_data[-1]['step_summary'].get('alive_agents', 0)

                min_survival_rate = expected_patterns['population_stability'].get('min_survival_rate', 0.1)
                survival_rate = final_pop / initial_pop if initial_pop > 0 else 0

                if survival_rate >= min_survival_rate:
                    validation['tests_passed'] += 1
                    validation['detailed_results']['population_stability'] = f'PASS: Survival rate {survival_rate:.2f}'
                else:
                    validation['tests_failed'] += 1
                    validation['detailed_results']['population_stability'] = f'FAIL: Survival rate {survival_rate:.2f} below threshold'

            # Test 2: Health dynamics
            if 'health_decay' in expected_patterns and len(self.model.simulation_data) > 10:
                health_series = [step['step_summary'].get('avg_health', 1) for step in self.model.simulation_data]

                should_decrease = expected_patterns['health_decay'].get('should_decrease', True)

                if should_decrease:
                    # Check if health generally decreases over time
                    trend_negative = health_series[-1] < health_series[0]
                    if trend_negative:
                        validation['tests_passed'] += 1
                        validation['detailed_results']['health_decay'] = 'PASS: Health shows expected decline'
                    else:
                        validation['tests_failed'] += 1
                        validation['detailed_results']['health_decay'] = 'FAIL: Health did not decline as expected'

            # Test 3: Wealth dynamics
            if 'wealth_accumulation' in expected_patterns:
                wealth_series = [step['step_summary'].get('avg_wealth', 0) for step in self.model.simulation_data]

                some_agents_gain = expected_patterns['wealth_accumulation'].get('some_agents_gain', True)

                if some_agents_gain:
                    # Check if average wealth increased
                    if len(wealth_series) > 1 and max(wealth_series) > wealth_series[0]:
                        validation['tests_passed'] += 1
                        validation['detailed_results']['wealth_accumulation'] = 'PASS: Some wealth accumulation observed'
                    else:
                        validation['tests_failed'] += 1
                        validation['detailed_results']['wealth_accumulation'] = 'FAIL: No wealth accumulation observed'

            # Test 4: Spatial behavior
            if 'spatial_distribution' in expected_patterns:
                agents_should_move = expected_patterns['spatial_distribution'].get('agents_should_move', True)

                # Check if agents moved during simulation
                movement_detected = False
                for agent_id, agent in self.model.agents.items():
                    if len(agent.state_history) > 1:
                        positions = [state['position'] for state in agent.state_history if state['position']]
                        if len(set(positions)) > 1:  # Agent changed position
                            movement_detected = True
                            break

                if agents_should_move:
                    if movement_detected:
                        validation['tests_passed'] += 1
                        validation['detailed_results']['spatial_distribution'] = 'PASS: Agent movement detected'
                    else:
                        validation['tests_failed'] += 1
                        validation['detailed_results']['spatial_distribution'] = 'FAIL: No agent movement detected'

            # Test 5: Interaction patterns (for social agents)
            social_agents = [agent for agent in self.model.agents.values() if isinstance(agent, SocialAgent)]
            if social_agents:
                total_interactions = sum(len(agent.influence_history) for agent in social_agents)
                interactions_per_agent = total_interactions / len(social_agents)

                if interactions_per_agent > 0:
                    validation['tests_passed'] += 1
                    validation['detailed_results']['social_interactions'] = f'PASS: {interactions_per_agent:.1f} interactions per social agent'
                else:
                    validation['tests_failed'] += 1
                    validation['detailed_results']['social_interactions'] = 'FAIL: No social interactions detected'

            # Overall status
            total_tests = validation['tests_passed'] + validation['tests_failed']
            if total_tests > 0:
                if validation['tests_failed'] == 0:
                    validation['overall_status'] = 'PASS'
                elif validation['tests_passed'] / total_tests >= 0.7:
                    validation['overall_status'] = 'WARNING'
                else:
                    validation['overall_status'] = 'FAIL'

            validation['validation_date'] = datetime.now()
            validation['expected_patterns'] = expected_patterns
            self.validation_results['external_validation'] = validation

            return validation

        except Exception as e:
            return {'error': str(e), 'validation_type': 'external'}

    def sensitivity_analysis(self, parameter_ranges: Dict[str, List[float]],
                           n_runs: int = 10) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on model parameters.

        Parameters:
        -----------
        parameter_ranges : Dict
            Parameters and their ranges to test
        n_runs : int
            Number of runs per parameter combination

        Returns:
        --------
        Dict with sensitivity analysis results
        """
        try:
            if not parameter_ranges:
                return {'error': 'No parameter ranges provided'}

            results = {
                'parameter_ranges': parameter_ranges,
                'sensitivity_results': {},
                'summary_statistics': {},
                'most_sensitive_parameters': []
            }

            baseline_metrics = None

            # Get baseline results (current parameter values)
            try:
                baseline_simulation = self.model.run_simulation(n_steps=50)
                if 'error' not in baseline_simulation:
                    baseline_analysis = baseline_simulation['final_analysis']
                    baseline_metrics = {
                        'avg_health': np.mean(baseline_analysis['time_series']['health']),
                        'avg_wealth': np.mean(baseline_analysis['time_series']['wealth']),
                        'final_population': baseline_analysis['time_series']['population'][-1],
                        'total_interactions': sum(baseline_analysis['time_series']['interactions'])
                    }
            except:
                pass

            # Test each parameter
            for param_name, param_values in parameter_ranges.items():
                param_results = {
                    'parameter_values': param_values,
                    'run_results': [],
                    'sensitivity_metrics': {}
                }

                for param_value in param_values:
                    run_metrics = []

                    for run in range(n_runs):
                        # Create new model instance with modified parameter
                        # This is a simplified approach - in practice, you'd need
                        # parameter injection mechanisms

                        # For demonstration, we'll vary environment size
                        if param_name == 'environment_size':
                            test_env = Environment(size=(param_value, param_value))
                        else:
                            test_env = Environment(size=self.model.environment.size)

                        test_model = AgentBasedModel(test_env)

                        # Add sample agents (simplified)
                        for i in range(min(10, int(param_value) if param_name == 'agent_count' else 10)):
                            char = AgentCharacteristics(
                                agent_id=f'test_agent_{i}',
                                agent_type=AgentType.BASIC,
                                position=(np.random.uniform(0, test_env.size[0]),
                                        np.random.uniform(0, test_env.size[1]))
                            )
                            agent = Agent(char, test_env)
                            test_model.add_agent(agent)

                        # Run short simulation
                        simulation_result = test_model.run_simulation(n_steps=30)

                        if 'error' not in simulation_result:
                            analysis = simulation_result['final_analysis']
                            run_metric = {
                                'avg_health': np.mean(analysis['time_series']['health']) if analysis['time_series']['health'] else 0,
                                'avg_wealth': np.mean(analysis['time_series']['wealth']) if analysis['time_series']['wealth'] else 0,
                                'final_population': analysis['time_series']['population'][-1] if analysis['time_series']['population'] else 0,
                                'total_interactions': sum(analysis['time_series']['interactions']) if analysis['time_series']['interactions'] else 0
                            }
                            run_metrics.append(run_metric)

                    if run_metrics:
                        # Calculate average metrics for this parameter value
                        avg_metrics = {
                            metric: np.mean([run[metric] for run in run_metrics])
                            for metric in run_metrics[0].keys()
                        }
                        param_results['run_results'].append({
                            'parameter_value': param_value,
                            'avg_metrics': avg_metrics,
                            'std_metrics': {
                                metric: np.std([run[metric] for run in run_metrics])
                                for metric in run_metrics[0].keys()
                            }
                        })

                # Calculate sensitivity for this parameter
                if param_results['run_results']:
                    metric_ranges = {}
                    for metric in param_results['run_results'][0]['avg_metrics'].keys():
                        values = [result['avg_metrics'][metric] for result in param_results['run_results']]
                        metric_ranges[metric] = {
                            'min': min(values),
                            'max': max(values),
                            'range': max(values) - min(values),
                            'coefficient_of_variation': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                        }

                    param_results['sensitivity_metrics'] = metric_ranges

                results['sensitivity_results'][param_name] = param_results

            # Identify most sensitive parameters
            sensitivity_scores = {}
            for param_name, param_data in results['sensitivity_results'].items():
                if 'sensitivity_metrics' in param_data:
                    # Use coefficient of variation as sensitivity measure
                    total_cv = sum(metric_data['coefficient_of_variation']
                                 for metric_data in param_data['sensitivity_metrics'].values())
                    sensitivity_scores[param_name] = total_cv

            # Sort by sensitivity
            results['most_sensitive_parameters'] = sorted(
                sensitivity_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            results['analysis_date'] = datetime.now()
            results['baseline_metrics'] = baseline_metrics
            self.validation_results['sensitivity_analysis'] = results

            return results

        except Exception as e:
            return {'error': str(e), 'analysis_type': 'sensitivity'}

    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        try:
            report = {
                'model_info': {
                    'total_agents': len(self.model.agents),
                    'environment_size': self.model.environment.size,
                    'simulation_steps': len(self.model.simulation_data),
                    'has_network': self.model.network is not None
                },
                'validation_summary': {
                    'internal_verification': 'NOT_RUN',
                    'external_validation': 'NOT_RUN',
                    'sensitivity_analysis': 'NOT_RUN'
                },
                'recommendations': [],
                'overall_assessment': 'INCOMPLETE'
            }

            # Check what validations have been run
            if 'internal_verification' in self.validation_results:
                iv_result = self.validation_results['internal_verification']
                report['validation_summary']['internal_verification'] = iv_result.get('overall_status', 'UNKNOWN')

            if 'external_validation' in self.validation_results:
                ev_result = self.validation_results['external_validation']
                report['validation_summary']['external_validation'] = ev_result.get('overall_status', 'UNKNOWN')

            if 'sensitivity_analysis' in self.validation_results:
                sa_result = self.validation_results['sensitivity_analysis']
                report['validation_summary']['sensitivity_analysis'] = 'COMPLETED' if 'error' not in sa_result else 'FAILED'

            # Generate recommendations
            if report['validation_summary']['internal_verification'] == 'FAIL':
                report['recommendations'].append('Fix internal consistency issues before proceeding')

            if report['validation_summary']['external_validation'] == 'FAIL':
                report['recommendations'].append('Review model logic and expected behaviors')

            if len(self.model.simulation_data) < 50:
                report['recommendations'].append('Run longer simulations for more robust validation')

            if self.model.network is None and len(self.model.agents) > 1:
                report['recommendations'].append('Consider adding network topology for agent interactions')

            # Overall assessment
            statuses = list(report['validation_summary'].values())
            if all(status in ['PASS', 'COMPLETED'] for status in statuses if status != 'NOT_RUN'):
                report['overall_assessment'] = 'VALIDATED'
            elif any(status == 'FAIL' for status in statuses):
                report['overall_assessment'] = 'NEEDS_WORK'
            elif any(status == 'WARNING' for status in statuses):
                report['overall_assessment'] = 'ACCEPTABLE'
            else:
                report['overall_assessment'] = 'INCOMPLETE'

            report['validation_results'] = self.validation_results
            report['report_date'] = datetime.now()

            return report

        except Exception as e:
            return {'error': str(e), 'report_type': 'validation_report'}

