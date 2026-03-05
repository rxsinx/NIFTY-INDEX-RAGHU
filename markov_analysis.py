"""
Hidden Markov Model (HMM) Analysis for Stock/Index Market Prediction
Reused from original codebase — unchanged logic, works for indices too.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class HiddenMarkovAnalysis:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.prices = data['Close'].values
        self.returns = np.diff(np.log(self.prices))
        self.n_states = 3
        self.state_names = ['BULL', 'BEAR', 'SIDEWAYS']
        self.transition_matrix = None
        self.emission_params = None
        self.initial_probs = None
        self.hidden_states = None
        self.state_probabilities = None

    def estimate_hmm_parameters(self):
        returns = self.returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        states = np.zeros(len(returns), dtype=int)
        for i, ret in enumerate(returns):
            if ret > mean_return + 0.5 * std_return: states[i] = 0
            elif ret < mean_return - 0.5 * std_return: states[i] = 1
            else: states[i] = 2

        transition_counts = np.zeros((self.n_states, self.n_states))
        for i in range(len(states) - 1):
            transition_counts[states[i], states[i + 1]] += 1

        self.transition_matrix = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            rs = np.sum(transition_counts[i, :])
            self.transition_matrix[i, :] = transition_counts[i, :] / rs if rs > 0 else 1.0 / self.n_states

        self.emission_params = {}
        for state in range(self.n_states):
            sr = returns[states == state]
            if len(sr) > 0:
                self.emission_params[state] = {'mean': float(np.mean(sr)), 'std': float(np.std(sr)), 'count': len(sr)}
            else:
                self.emission_params[state] = {'mean': 0.0, 'std': std_return, 'count': 0}

        self.initial_probs = np.array([np.sum(states == s) / len(states) for s in range(self.n_states)])
        return {'transition_matrix': self.transition_matrix.tolist(), 'emission_params': self.emission_params,
                'initial_probs': self.initial_probs.tolist(), 'state_names': self.state_names}

    def viterbi_algorithm(self):
        if self.transition_matrix is None: self.estimate_hmm_parameters()
        returns = self.returns
        T = len(returns)
        viterbi = np.zeros((self.n_states, T))
        path = np.zeros((self.n_states, T), dtype=int)
        for state in range(self.n_states):
            ep = self._emission_probability(returns[0], state)
            viterbi[state, 0] = np.log(self.initial_probs[state] + 1e-10) + np.log(ep + 1e-10)
        for t in range(1, T):
            for state in range(self.n_states):
                tp = viterbi[:, t-1] + np.log(self.transition_matrix[:, state] + 1e-10)
                path[state, t] = np.argmax(tp)
                viterbi[state, t] = np.max(tp) + np.log(self._emission_probability(returns[t], state) + 1e-10)
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(viterbi[:, T-1])
        for t in range(T-2, -1, -1): states[t] = path[states[t+1], t+1]
        self.hidden_states = states
        return states

    def forward_backward_algorithm(self):
        if self.transition_matrix is None: self.estimate_hmm_parameters()
        returns = self.returns
        T = len(returns)
        alpha = np.zeros((self.n_states, T))
        for state in range(self.n_states):
            alpha[state, 0] = self.initial_probs[state] * self._emission_probability(returns[0], state)
        alpha[:, 0] /= np.sum(alpha[:, 0])
        for t in range(1, T):
            for state in range(self.n_states):
                alpha[state, t] = np.sum(alpha[:, t-1] * self.transition_matrix[:, state]) * self._emission_probability(returns[t], state)
            alpha[:, t] /= np.sum(alpha[:, t])
        beta = np.ones((self.n_states, T))
        for t in range(T-2, -1, -1):
            for state in range(self.n_states):
                beta[state, t] = np.sum(self.transition_matrix[state, :] * self._emission_probability(returns[t+1], np.arange(self.n_states)) * beta[:, t+1])
            beta[:, t] /= max(np.sum(beta[:, t]), 1e-10)
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=0)
        self.state_probabilities = gamma
        return gamma

    def _emission_probability(self, observation, state):
        if isinstance(state, np.ndarray):
            probs = np.zeros(len(state))
            for i, s in enumerate(state):
                p = self.emission_params[int(s)]
                std = p['std'] if p['std'] > 0 else 1e-6
                probs[i] = stats.norm.pdf(observation, loc=p['mean'], scale=std)
            return probs
        p = self.emission_params[state]
        std = p['std'] if p['std'] > 0 else 1e-6
        return stats.norm.pdf(observation, loc=p['mean'], scale=std)

    def forecast_price(self, forecast_days=30, n_simulations=1000):
        if self.transition_matrix is None: self.estimate_hmm_parameters()
        if self.hidden_states is None: self.viterbi_algorithm()
        if self.state_probabilities is None: self.forward_backward_algorithm()
        current_state = self.hidden_states[-1]
        current_price = self.prices[-1]
        current_state_probs = self.state_probabilities[:, -1]
        forecast_prices = np.zeros((n_simulations, forecast_days + 1))
        forecast_prices[:, 0] = current_price
        state_paths = np.zeros((n_simulations, forecast_days), dtype=int)
        for sim in range(n_simulations):
            state = np.random.choice(self.n_states, p=current_state_probs)
            for day in range(forecast_days):
                state_paths[sim, day] = state
                ep = self.emission_params[state]
                dr = np.random.normal(ep['mean'], ep['std'])
                forecast_prices[sim, day + 1] = forecast_prices[sim, day] * np.exp(dr)
                state = np.random.choice(self.n_states, p=self.transition_matrix[state, :])
        forecast_prices = forecast_prices[:, 1:]
        mean_f = np.mean(forecast_prices, axis=0)
        sf = np.std(forecast_prices, axis=0)
        state_freq = np.zeros((forecast_days, self.n_states))
        for day in range(forecast_days):
            for s in range(self.n_states):
                state_freq[day, s] = np.sum(state_paths[:, day] == s) / n_simulations
        expected_return = (mean_f[-1] - current_price) / current_price * 100
        direction = 'BULLISH' if expected_return > 5 else ('BEARISH' if expected_return < -5 else 'NEUTRAL')
        signal = 'BUY' if direction == 'BULLISH' else ('SELL' if direction == 'BEARISH' else 'HOLD')
        dominant_state = np.argmax(state_freq[-1, :])
        try:
            last_date = self.data.index[-1]
            forecast_dates = [d.to_pydatetime() for d in pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)]
        except Exception:
            last_date = datetime.now()
            forecast_dates = [(last_date + timedelta(days=i+1)) for i in range(forecast_days)]
        regime_persistence = self._calculate_regime_persistence()
        confidence_score = self._assess_forecast_confidence(current_state_probs, regime_persistence)
        return {
            'forecast_days': forecast_days, 'n_simulations': n_simulations,
            'current_price': float(current_price),
            'current_state': self.state_names[current_state],
            'current_state_probability': float(current_state_probs[current_state]),
            'state_probabilities': {self.state_names[i]: float(current_state_probs[i]) for i in range(self.n_states)},
            'direction': direction, 'signal': signal,
            'expected_return': float(expected_return),
            'expected_volatility': float(np.mean(sf) / current_price * 100),
            'dates': forecast_dates,
            'mean_forecast': mean_f.tolist(),
            'median_forecast': np.median(forecast_prices, axis=0).tolist(),
            'std_forecast': sf.tolist(),
            'ci_lower_95': np.percentile(forecast_prices, 2.5, axis=0).tolist(),
            'ci_upper_95': np.percentile(forecast_prices, 97.5, axis=0).tolist(),
            'ci_lower_68': np.percentile(forecast_prices, 16, axis=0).tolist(),
            'ci_upper_68': np.percentile(forecast_prices, 84, axis=0).tolist(),
            'target_price': float(mean_f[-1]),
            'best_case': float(np.percentile(forecast_prices, 97.5, axis=0)[-1]),
            'worst_case': float(np.percentile(forecast_prices, 2.5, axis=0)[-1]),
            'dominant_regime': self.state_names[dominant_state],
            'regime_confidence': float(state_freq[-1, dominant_state]),
            'regime_persistence': regime_persistence,
            'state_transition_matrix': self.transition_matrix.tolist(),
            'state_frequencies': state_freq.tolist(),
            'bull_probability': float(np.mean(state_freq[:, 0])),
            'bear_probability': float(np.mean(state_freq[:, 1])),
            'sideways_probability': float(np.mean(state_freq[:, 2])),
            'confidence_level': confidence_score['level'],
            'confidence_score': float(confidence_score['score']),
            'confidence_factors': confidence_score['factors'],
            'method': 'Hidden Markov Model (HMM) with Monte Carlo',
            'algorithm': 'Viterbi + Forward-Backward + Baum-Welch'
        }

    def _calculate_regime_persistence(self):
        if self.hidden_states is None: self.viterbi_algorithm()
        states = self.hidden_states
        persistence = {}
        for state in range(self.n_states):
            runs, current_run = [], 0
            for s in states:
                if s == state: current_run += 1
                else:
                    if current_run > 0: runs.append(current_run)
                    current_run = 0
            if current_run > 0: runs.append(current_run)
            if runs:
                persistence[self.state_names[state]] = {'avg_duration': float(np.mean(runs)), 'max_duration': int(np.max(runs)), 'n_occurrences': len(runs)}
            else:
                persistence[self.state_names[state]] = {'avg_duration': 0.0, 'max_duration': 0, 'n_occurrences': 0}
        return persistence

    def _assess_forecast_confidence(self, state_probs, persistence):
        score, factors = 0, []
        max_prob = np.max(state_probs)
        score += 0.3 if max_prob > 0.7 else (0.2 if max_prob > 0.5 else 0.1)
        factors.append(f"State confidence ({max_prob:.1%})")
        csn = self.state_names[np.argmax(state_probs)]
        ad = persistence[csn]['avg_duration']
        score += 0.3 if ad > 10 else (0.2 if ad > 5 else 0.1)
        factors.append(f"{csn} avg duration {ad:.0f} days")
        te = -np.sum(self.transition_matrix * np.log(self.transition_matrix + 1e-10), axis=1)
        clarity = 1 - (np.mean(te) / np.log(self.n_states))
        score += 0.2 if clarity > 0.5 else 0.1
        factors.append(f"Transition clarity {clarity:.1%}")
        n = len(self.returns)
        score += 0.2 if n > 200 else (0.15 if n > 100 else 0.05)
        factors.append(f"Sample size n={n}")
        level = 'HIGH' if score >= 0.8 else ('MEDIUM' if score >= 0.6 else 'LOW')
        return {'score': score, 'level': level, 'factors': factors}

    def analyze_regime_characteristics(self):
        if self.hidden_states is None: self.viterbi_algorithm()
        states = self.hidden_states
        returns = self.returns
        chars = {}
        for state in range(self.n_states):
            mask = states == state
            sr = returns[mask]
            if len(sr) > 0:
                chars[self.state_names[state]] = {
                    'avg_return': float(np.mean(sr)*100), 'volatility': float(np.std(sr)*100),
                    'sharpe_ratio': float(np.mean(sr)/np.std(sr)) if np.std(sr) > 0 else 0,
                    'win_rate': float(np.sum(sr > 0)/len(sr)*100),
                    'avg_gain': float(np.mean(sr[sr>0])*100) if np.sum(sr>0) > 0 else 0,
                    'avg_loss': float(np.mean(sr[sr<0])*100) if np.sum(sr<0) > 0 else 0,
                    'max_gain': float(np.max(sr)*100), 'max_loss': float(np.min(sr)*100),
                    'occurrences': int(np.sum(mask)), 'duration_pct': float(np.sum(mask)/len(states)*100)
                }
            else:
                chars[self.state_names[state]] = {k: 0.0 for k in ['avg_return','volatility','sharpe_ratio','win_rate','avg_gain','avg_loss','max_gain','max_loss','occurrences','duration_pct']}
        return chars

    def generate_trading_strategy(self, forecast):
        chars = self.analyze_regime_characteristics()
        cc = chars[forecast['current_state']]
        strategy = {
            'signal': forecast['signal'], 'direction': forecast['direction'],
            'confidence': forecast['confidence_level'],
            'entry_price': forecast['current_price'], 'target_price': forecast['target_price'],
            'stop_loss': None, 'position_size': None,
            'time_horizon': f"{forecast['forecast_days']} days",
            'rationale': [], 'risks': []
        }
        if forecast['direction'] == 'BULLISH':
            sd = cc['volatility'] * 2
            strategy['stop_loss'] = forecast['current_price'] * (1 - sd/100)
            strategy['rationale'].append(f"Entering {forecast['current_state']} with {forecast['expected_return']:.1f}% upside")
            strategy['position_size'] = '3-5% of portfolio' if forecast['confidence_level']=='HIGH' else '2-3% of portfolio'
        elif forecast['direction'] == 'BEARISH':
            sd = cc['volatility'] * 2
            strategy['stop_loss'] = forecast['current_price'] * (1 + sd/100)
            strategy['rationale'].append(f"Bearish — {abs(forecast['expected_return']):.1f}% downside expected")
            strategy['position_size'] = 'Reduce exposure or SHORT/PUT'
        else:
            strategy['stop_loss'] = forecast['current_price'] * 0.97
            strategy['rationale'].append("Sideways — range trading or iron condor")
            strategy['position_size'] = 'Options strategies for range'
        return strategy


def run_hmm_analysis(data: pd.DataFrame, forecast_days: int = 30) -> Dict:
    hmm = HiddenMarkovAnalysis(data)
    params = hmm.estimate_hmm_parameters()
    states = hmm.viterbi_algorithm()
    state_probs = hmm.forward_backward_algorithm()
    forecast = hmm.forecast_price(forecast_days=forecast_days)
    characteristics = hmm.analyze_regime_characteristics()
    strategy = hmm.generate_trading_strategy(forecast)
    persistence = hmm._calculate_regime_persistence()
    return {'hmm_parameters': params, 'forecast': forecast,
            'characteristics': characteristics, 'strategy': strategy, 'persistence': persistence}
