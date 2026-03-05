"""
Markov Chain Monte Carlo (MCMC) Analysis — reused from original codebase.
Works identically for index price series.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


def log_likelihood(log_returns, mu, sigma):
    if sigma <= 0: return -np.inf
    drift = mu - 0.5 * sigma ** 2
    return float(stats.norm.logpdf(log_returns, loc=drift, scale=sigma).sum())


def log_prior(mu, sigma):
    if sigma <= 0: return -np.inf
    return float(stats.norm.logpdf(mu, loc=0.0, scale=0.10) +
                 stats.halfnorm.logpdf(sigma, loc=0.0, scale=0.03))


def log_posterior(log_returns, mu, sigma):
    return log_likelihood(log_returns, mu, sigma) + log_prior(mu, sigma)


class MetropolisHastingsSampler:
    def __init__(self, log_returns, n_samples=5000, n_warmup=2000, n_chains=4,
                 tune_interval=100, target_accept=0.234, seed=42):
        self.log_returns = log_returns
        self.n_samples = n_samples
        self.n_warmup = n_warmup
        self.n_chains = n_chains
        self.tune_interval = tune_interval
        self.target_accept = target_accept
        self.rng = np.random.default_rng(seed)
        self.step_mu = 0.005
        self.step_sigma = 0.002

    def _run_single_chain(self, init_mu, init_sigma, chain_id):
        rng = np.random.default_rng(self.rng.integers(0, 2**31) + chain_id)
        total = self.n_warmup + self.n_samples
        mu_arr, sigma_arr = np.empty(total), np.empty(total)
        mu_cur, sigma_cur = init_mu, init_sigma
        lp_cur = log_posterior(self.log_returns, mu_cur, sigma_cur)
        step_mu, step_sigma = self.step_mu, self.step_sigma
        accept_count = tune_accepts = 0

        for i in range(total):
            mu_prop    = mu_cur    + rng.normal(0, step_mu)
            sigma_prop = sigma_cur + rng.normal(0, step_sigma)
            lp_prop = log_posterior(self.log_returns, mu_prop, sigma_prop)
            if np.log(rng.uniform()) < (lp_prop - lp_cur):
                mu_cur, sigma_cur, lp_cur = mu_prop, sigma_prop, lp_prop
                accept_count += 1
                if i < self.n_warmup: tune_accepts += 1
            mu_arr[i], sigma_arr[i] = mu_cur, sigma_cur
            if i < self.n_warmup and (i + 1) % self.tune_interval == 0:
                rate = tune_accepts / self.tune_interval
                factor = np.clip(np.exp(rate - self.target_accept), 0.5, 2.0)
                step_mu *= factor; step_sigma *= factor
                tune_accepts = 0

        return mu_arr[self.n_warmup:], sigma_arr[self.n_warmup:], accept_count / total

    def sample(self):
        rng = np.random.default_rng(self.rng.integers(0, 2**31))
        lr, em, es = self.log_returns, self.log_returns.mean(), self.log_returns.std()
        init_mus    = rng.normal(em, 0.5 * abs(em) + 1e-4, self.n_chains)
        init_sigmas = np.abs(rng.normal(es, 0.3 * es + 1e-4, self.n_chains)) + 1e-5
        mu_chains    = np.empty((self.n_chains, self.n_samples))
        sigma_chains = np.empty((self.n_chains, self.n_samples))
        accept_rates = []
        for c in range(self.n_chains):
            mc, sc, ar = self._run_single_chain(init_mus[c], init_sigmas[c], chain_id=c)
            mu_chains[c], sigma_chains[c] = mc, sc
            accept_rates.append(ar)
        return {
            'mu_samples': mu_chains.flatten(), 'sigma_samples': sigma_chains.flatten(),
            'mu_chains': mu_chains, 'sigma_chains': sigma_chains,
            'acceptance_rates': accept_rates,
            'r_hat_mu':    self._gelman_rubin(mu_chains),
            'r_hat_sigma': self._gelman_rubin(sigma_chains),
            'ess_mu':      self._effective_sample_size(mu_chains),
            'ess_sigma':   self._effective_sample_size(sigma_chains),
            'n_samples': self.n_samples, 'n_chains': self.n_chains, 'n_warmup': self.n_warmup,
        }

    @staticmethod
    def _gelman_rubin(chains):
        n_chains, n = chains.shape
        if n_chains < 2: return 1.0
        cm, cv = chains.mean(axis=1), chains.var(axis=1, ddof=1)
        B = n * cm.var(ddof=1)
        W = cv.mean()
        var_hat = (1 - 1/n) * W + (1/n) * B
        return float(np.sqrt(var_hat / W)) if W > 0 else np.inf

    @staticmethod
    def _effective_sample_size(chains):
        samples = chains.flatten()
        if samples.std() < 1e-10: return float(len(samples))
        max_lag = min(100, chains.shape[1] // 4)
        acf = []
        for lag in range(1, max_lag + 1):
            c = np.corrcoef(samples[:-lag], samples[lag:])[0, 1]
            if np.isnan(c): break
            acf.append(c)
            if c < 0: break
        return float(len(samples) / max(1 + 2 * sum(acf), 1.0))


def simulate_price_paths(current_price, mu_samples, sigma_samples,
                          forecast_days=30, n_paths=2000, seed=99):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(mu_samples), size=n_paths)
    mu_d, sig_d = mu_samples[idx], sigma_samples[idx]
    drift  = mu_d - 0.5 * sig_d ** 2
    noise  = rng.normal(0, 1, size=(n_paths, forecast_days))
    log_r  = drift[:, None] + sig_d[:, None] * noise
    return current_price * np.exp(np.cumsum(log_r, axis=1))


def compute_risk_metrics(paths, current_price, horizon=30, confidence=0.95):
    tp  = paths[:, horizon - 1]
    tr  = (tp - current_price) / current_price
    alpha = 1 - confidence
    var_l = float(np.percentile(tr, alpha * 100))
    cvar_l= float(tr[tr <= var_l].mean()) if (tr <= var_l).any() else var_l
    pcts = [2.5, 10, 25, 50, 75, 90, 97.5]
    bands = {str(p): np.percentile(paths[:, :horizon], p, axis=0).tolist() for p in pcts}
    return {
        'mean_return': float(np.mean(tr)), 'median_return': float(np.median(tr)),
        'std_return': float(np.std(tr)),
        'mean_price': float(np.mean(tp)), 'median_price': float(np.median(tp)),
        f'var_{int(confidence*100)}': var_l, f'cvar_{int(confidence*100)}': cvar_l,
        'prob_profit': float(np.mean(tr > 0)),
        'prob_gain_5pct': float(np.mean(tr > 0.05)),
        'prob_gain_10pct': float(np.mean(tr > 0.10)),
        'prob_loss_5pct': float(np.mean(tr < -0.05)),
        'ci_50': (float(np.percentile(tp, 25)), float(np.percentile(tp, 75))),
        'ci_80': (float(np.percentile(tp, 10)), float(np.percentile(tp, 90))),
        'ci_95': (float(np.percentile(tp, 2.5)), float(np.percentile(tp, 97.5))),
        'fan_bands': bands, 'sample_paths': paths[:200, :horizon],
    }


def summarise_posterior(mcmc_result, log_returns):
    mu, sigma = mcmc_result['mu_samples'], mcmc_result['sigma_samples']

    def s(arr, name):
        return {f'{name}_mean': float(arr.mean()), f'{name}_median': float(np.median(arr)),
                f'{name}_std': float(arr.std()),
                f'{name}_ci_90_lo': float(np.percentile(arr, 5)),
                f'{name}_ci_90_hi': float(np.percentile(arr, 95)),
                f'{name}_ci_95_lo': float(np.percentile(arr, 2.5)),
                f'{name}_ci_95_hi': float(np.percentile(arr, 97.5))}

    pm, ps = s(mu, 'mu'), s(sigma, 'sigma')
    mle_mu  = float(log_returns.mean())
    mle_sig = float(log_returns.std())
    r_mu    = mcmc_result['r_hat_mu']
    r_sig   = mcmc_result['r_hat_sigma']
    ess_mu  = mcmc_result['ess_mu']
    ess_sig = mcmc_result['ess_sigma']
    converged = (r_mu < 1.05) and (r_sig < 1.05) and (ess_mu > 400) and (ess_sig > 400)
    avg_accept = float(np.mean(mcmc_result['acceptance_rates']))
    return {
        **pm, **ps,
        'mle_mu_daily': mle_mu, 'mle_mu_true': mle_mu + 0.5 * log_returns.var(),
        'mle_sigma_daily': mle_sig,
        'ann_mu_mean': pm['mu_mean'] * 252, 'ann_mu_lo95': pm['mu_ci_95_lo'] * 252,
        'ann_mu_hi95': pm['mu_ci_95_hi'] * 252, 'ann_sigma': ps['sigma_mean'] * np.sqrt(252),
        'r_hat_mu': r_mu, 'r_hat_sigma': r_sig, 'ess_mu': ess_mu, 'ess_sigma': ess_sig,
        'avg_accept_rate': avg_accept, 'converged': converged,
        'n_obs': len(log_returns),
        'n_total_samples': mcmc_result['n_samples'] * mcmc_result['n_chains'],
    }


def run_mcmc_analysis(data, forecast_days=30, n_samples=4000, n_warmup=1500,
                       n_chains=4, n_paths=2000, seed=42):
    close = data['Close'].dropna().values
    log_returns = np.diff(np.log(close))
    current_price = float(close[-1])

    sampler = MetropolisHastingsSampler(log_returns=log_returns, n_samples=n_samples,
                                         n_warmup=n_warmup, n_chains=n_chains, seed=seed)
    mcmc_result = sampler.sample()
    posterior   = summarise_posterior(mcmc_result, log_returns)
    paths       = simulate_price_paths(current_price, mcmc_result['mu_samples'],
                                        mcmc_result['sigma_samples'], forecast_days, n_paths, seed)
    risk        = compute_risk_metrics(paths, current_price, horizon=forecast_days)

    target_price = risk['median_price']
    exp_return   = risk['median_return']
    if exp_return > 0.05:   direction, signal = 'BULLISH', 'BUY'
    elif exp_return < -0.05: direction, signal = 'BEARISH', 'SELL'
    else:                    direction, signal = 'NEUTRAL', 'HOLD'

    try:
        last_date = data.index[-1]
        fdates = [d.to_pydatetime() for d in
                  pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)]
    except Exception:
        last_date = pd.Timestamp.now()
        fdates = [last_date + pd.Timedelta(days=i+1) for i in range(forecast_days)]

    mu_pos = float(np.mean(mcmc_result['mu_samples'] > 0))
    regime = ('Bullish Drift' if mu_pos > 0.70 else
              'Bearish Drift' if mu_pos < 0.30 else 'Uncertain / Mean-Reverting')

    forecast_summary = {
        'current_price': current_price, 'target_price': target_price,
        'mean_price': risk['mean_price'],
        'ci_95_low': risk['ci_95'][0], 'ci_95_high': risk['ci_95'][1],
        'ci_80_low': risk['ci_80'][0], 'ci_80_high': risk['ci_80'][1],
        'ci_50_low': risk['ci_50'][0], 'ci_50_high': risk['ci_50'][1],
        'expected_return': exp_return * 100, 'direction': direction, 'signal': signal,
        'regime': regime, 'mu_pos_fraction': mu_pos, 'forecast_days': forecast_days,
        'forecast_dates': fdates, 'fan_bands': risk['fan_bands'],
        'sample_paths': risk['sample_paths'],
        'ann_drift_mean': posterior['ann_mu_mean'] * 100,
        'ann_drift_lo':   posterior['ann_mu_lo95'] * 100,
        'ann_drift_hi':   posterior['ann_mu_hi95'] * 100,
        'ann_volatility': posterior['ann_sigma']   * 100,
    }
    diagnostics = {
        'converged': posterior['converged'],
        'r_hat_mu': posterior['r_hat_mu'], 'r_hat_sigma': posterior['r_hat_sigma'],
        'ess_mu': posterior['ess_mu'],     'ess_sigma': posterior['ess_sigma'],
        'accept_rate': posterior['avg_accept_rate'],
        'n_obs': posterior['n_obs'], 'n_total_samples': posterior['n_total_samples'],
    }
    return {'mcmc_result': mcmc_result, 'posterior': posterior, 'risk_metrics': risk,
            'forecast_summary': forecast_summary, 'diagnostics': diagnostics,
            'log_returns': log_returns}
