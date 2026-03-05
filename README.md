# 📊 NIFTY & BANKNIFTY Index Analyzer

Institutional-grade analysis suite for NSE indices.

## Indices Supported
- NIFTY 50, BANKNIFTY, NIFTY IT, NIFTY FMCG, NIFTY AUTO, NIFTY PHARMA
- NIFTY MIDCAP, NIFTY REALTY, NIFTY METAL, NIFTY ENERGY

## Features

### 📈 Price Action & Indicators
- Candlestick chart with SMA 21/50/200 + EMA 9/20/50/200
- Bollinger Bands, VWAP, RSI, MACD, Stochastics
- Pivot Points (Standard + Camarilla), ATR%
- 52-Week High/Low tracking, percentage from ATH/ATL

### 🔍 Pattern Detection (20 Patterns)
**Trend Continuation (8):** Ascending/Descending/Symmetrical Triangle,
Bull/Bear Flag, Rising/Falling Wedge, Bullish/Bearish Pennant

**Reversal (6):** Head & Shoulders, Inverse H&S, Double Top/Bottom,
Triple Top/Bottom

**Advanced / Index-Specific (7):** Range Breakout, Cup & Handle,
Flat Base/Shelf, Mean Reversion, Elliott Wave,
Wyckoff Accumulation, Wyckoff Distribution

### 🤖 AI Adaptive Supertrend (K-Means)
- K-Means clusters ATR into volatility regimes (Low / Medium / High)
- Auto-selects optimal Supertrend multiplier (1.5× / 2.5× / 3.5×)
- Background regime shading, Buy/Sell flip signals

### ⛓️ MCMC Bayesian Forecast
- Metropolis-Hastings sampler, 2–4 chains, 1000–5000 samples
- Full posterior fan chart with 50/80/95% credible intervals
- R-hat convergence diagnostics, ESS, trace plots
- Bayesian VaR, CVaR, P(profit)

### 🎲 HMM Price Forecast
- 3-state Hidden Markov Model (Bull / Bear / Sideways)
- Viterbi + Forward-Backward algorithms
- 30-day Monte Carlo simulation from posterior regimes

### 📊 Volume Profile
- Volume-at-price (50 bins), Point of Control (POC)
- Value Area (70%), High/Low Volume Nodes

## Running
```bash
pip install -r requirements.txt
streamlit run index_app.py
```

## Files
| File | Description |
|------|-------------|
| `index_app.py` | Main Streamlit application |
| `index_pattern_detector.py` | 20 index-specific pattern detectors |
| `markov_analysis.py` | HMM regime detection & forecast |
| `mcmc_analysis.py` | MCMC Bayesian price forecast |
| `requirements.txt` | Python dependencies |

## Key Differences from Equity Analyzer
- No Market Cap / PE / PB / Dividend Yield (not applicable to indices)
- Pivot points and Camarilla levels (critical for options writers)
- Index-specific volatility metrics (ATR%, 20D Vol annualised)
- Range Breakout pattern (highly relevant for index options)
- Volume represents combined Futures + Cash market activity
