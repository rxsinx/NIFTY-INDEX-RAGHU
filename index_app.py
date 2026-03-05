"""
NIFTY & BANKNIFTY Index Analyzer
=================================
Master-grade analysis system for Indian index trading.
Covers: Pattern Detection, Price Action & Indicators,
        AI Adaptive Supertrend (K-Means), MCMC Bayesian Forecast,
        HMM Price Forecast, Volume Profile, Breadth & Derivatives context.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional

from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from sklearn.cluster import KMeans

from index_pattern_detector import IndexPatternDetector
from markov_analysis import HiddenMarkovAnalysis, run_hmm_analysis
from mcmc_analysis import run_mcmc_analysis

warnings.filterwarnings("ignore")

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIFTY & BANKNIFTY Index Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    font-size: 40px; font-weight: bold; color: #1f77b4;
    text-align: center; margin-bottom: 6px;
}
.sub-header {
    font-size: 22px; font-weight: bold; color: #2ca02c;
    margin-top: 18px; margin-bottom: 8px;
}
.index-badge-nifty  { background:#1565C0; color:white; padding:4px 14px;
                      border-radius:20px; font-weight:bold; font-size:15px; }
.index-badge-bnifty { background:#AD1457; color:white; padding:4px 14px;
                      border-radius:20px; font-weight:bold; font-size:15px; }
</style>
""", unsafe_allow_html=True)

# ── Index symbol map ──────────────────────────────────────────────────────────
INDEX_MAP = {
    "NIFTY 50":   "^NSEI",
    "BANKNIFTY":  "^NSEBANK",
}

VIX_SYMBOL = "^INDIAVIX"


# ============================================================================
# AI / ADAPTIVE SUPERTREND  (K-Means)
# ============================================================================

def calculate_adaptive_supertrend(df, atr_period=10, n_clusters=3, lookback_clusters=100):
    df = df.copy()
    atr_ind = AverageTrueRange(df["High"], df["Low"], df["Close"], window=atr_period)
    df["ATR"] = atr_ind.average_true_range().bfill()

    atr_vals = df["ATR"].dropna().values
    window   = min(lookback_clusters, len(atr_vals))
    atr_win  = atr_vals[-window:].reshape(-1, 1)

    kmeans     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(atr_win)
    centroids  = kmeans.cluster_centers_.flatten()
    sorted_idx = np.argsort(centroids)
    cluster_map= {old: new for new, old in enumerate(sorted_idx)}

    all_labels        = kmeans.predict(df["ATR"].values.reshape(-1, 1))
    df["ATR_Cluster"] = np.array([cluster_map[l] for l in all_labels])

    base_mult = {0: 1.5, 1: 2.5, 2: 3.5}
    for c in range(3, n_clusters):
        base_mult[c] = 3.5 + (c - 2) * 0.5
    df["AI_Multiplier"] = df["ATR_Cluster"].map(base_mult)

    hl2     = (df["High"] + df["Low"]) / 2.0
    upper_b = hl2 + df["AI_Multiplier"] * df["ATR"]
    lower_b = hl2 - df["AI_Multiplier"] * df["ATR"]

    upper_f = upper_b.copy()
    lower_f = lower_b.copy()
    ai_st   = pd.Series(np.nan, index=df.index)
    ai_dir  = pd.Series(1, index=df.index)
    close   = df["Close"].values

    for i in range(1, len(df)):
        upper_f.iloc[i] = (min(upper_b.iloc[i], upper_f.iloc[i-1])
                           if close[i-1] <= upper_f.iloc[i-1] else upper_b.iloc[i])
        lower_f.iloc[i] = (max(lower_b.iloc[i], lower_f.iloc[i-1])
                           if close[i-1] >= lower_f.iloc[i-1] else lower_b.iloc[i])

        if   close[i] > upper_f.iloc[i-1]: ai_dir.iloc[i] =  1
        elif close[i] < lower_f.iloc[i-1]: ai_dir.iloc[i] = -1
        else:                               ai_dir.iloc[i] = ai_dir.iloc[i-1]

        ai_st.iloc[i] = lower_f.iloc[i] if ai_dir.iloc[i] == 1 else upper_f.iloc[i]

    df["AI_ST_Upper"]     = upper_f
    df["AI_ST_Lower"]     = lower_f
    df["AI_Supertrend"]   = ai_st
    df["AI_ST_Direction"] = ai_dir

    regime_map = {0: "Low Vol", 1: "Medium Vol", 2: "High Vol"}
    for c in range(3, n_clusters):
        regime_map[c] = f"Extreme Vol {c}"
    df["AI_ST_Regime"] = df["ATR_Cluster"].map(regime_map)
    return df


def create_adaptive_supertrend_chart(df_full, atr_period=10, n_clusters=3):
    df = calculate_adaptive_supertrend(df_full.tail(200).copy(),
                                       atr_period=atr_period, n_clusters=n_clusters)
    buy_signals  = df[(df["AI_ST_Direction"] == 1)  & (df["AI_ST_Direction"].shift(1) == -1)]
    sell_signals = df[(df["AI_ST_Direction"] == -1) & (df["AI_ST_Direction"].shift(1) ==  1)]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.50, 0.18, 0.14, 0.18],
                        subplot_titles=("🤖 AI Adaptive Supertrend — K-Means",
                                         "ATR by Volatility Regime",
                                         "Adaptive Multiplier",
                                         "Volume"))

    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                  low=df["Low"],  close=df["Close"], name="OHLC",
                                  increasing_line_color="#26a69a",
                                  decreasing_line_color="#ef5350"), row=1, col=1)

    bull_st = df["AI_Supertrend"].where(df["AI_ST_Direction"] == 1)
    bear_st = df["AI_Supertrend"].where(df["AI_ST_Direction"] == -1)
    fig.add_trace(go.Scatter(x=df.index, y=bull_st, name="AI-ST Bull",
                              line=dict(color="#00e676", width=2.5), mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bear_st, name="AI-ST Bear",
                              line=dict(color="#ff1744", width=2.5), mode="lines"), row=1, col=1)

    cluster_colours = {0: "rgba(0,230,118,0.07)", 1: "rgba(255,214,0,0.07)",
                       2: "rgba(255,23,68,0.07)"}
    prev_c, seg_s = None, df.index[0]
    for idx, row_d in df.iterrows():
        c = int(row_d["ATR_Cluster"])
        if c != prev_c:
            if prev_c is not None:
                fig.add_vrect(x0=seg_s, x1=idx,
                              fillcolor=cluster_colours.get(prev_c, "rgba(200,200,200,0.05)"),
                              layer="below", line_width=0, row=1, col=1)
            seg_s, prev_c = idx, c
    if prev_c is not None:
        fig.add_vrect(x0=seg_s, x1=df.index[-1],
                      fillcolor=cluster_colours.get(prev_c, "rgba(200,200,200,0.05)"),
                      layer="below", line_width=0, row=1, col=1)

    if not buy_signals.empty:
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals["Low"] * 0.993,
                                  mode="markers+text",
                                  marker=dict(symbol="triangle-up", size=14, color="#00e676",
                                              line=dict(color="white", width=1)),
                                  text=["BUY"] * len(buy_signals), textposition="bottom center",
                                  textfont=dict(color="#00e676", size=9),
                                  name="Buy Signal"), row=1, col=1)
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals["High"] * 1.007,
                                  mode="markers+text",
                                  marker=dict(symbol="triangle-down", size=14, color="#ff1744",
                                              line=dict(color="white", width=1)),
                                  text=["SELL"] * len(sell_signals), textposition="top center",
                                  textfont=dict(color="#ff1744", size=9),
                                  name="Sell Signal"), row=1, col=1)

    c_line  = {0: "#00e676", 1: "#ffd600", 2: "#ff1744"}
    c_label = {0: "ATR Low Vol", 1: "ATR Med Vol", 2: "ATR High Vol"}
    for c_id in range(n_clusters):
        fig.add_trace(go.Scatter(x=df.index, y=df["ATR"].where(df["ATR_Cluster"] == c_id),
                                  name=c_label.get(c_id, f"ATR Cluster {c_id}"),
                                  line=dict(color=c_line.get(c_id, "#aaa"), width=2),
                                  mode="lines"), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["AI_Multiplier"], name="Multiplier",
                              line=dict(color="#7c4dff", width=2), mode="lines",
                              fill="tozeroy", fillcolor="rgba(124,77,255,0.15)"), row=3, col=1)

    vol_c = ["#26a69a" if df["Close"].iloc[i] >= df["Open"].iloc[i]
             else "#ef5350" for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                          marker_color=vol_c, opacity=0.7), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Volume"].rolling(20).mean(),
                              name="Vol SMA-20", line=dict(color="orange", width=1.5, dash="dot")),
                  row=4, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, height=1100, showlegend=True,
                      hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
                      plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font=dict(color="#fafafa"))
    fig.update_xaxes(showgrid=False, color="#555")
    fig.update_yaxes(showgrid=True, gridcolor="#1e1e2e", color="#999")
    fig.update_yaxes(title_text="Index Value", row=1, col=1)
    fig.update_yaxes(title_text="ATR",         row=2, col=1)
    fig.update_yaxes(title_text="Multiplier",  row=3, col=1)
    fig.update_yaxes(title_text="Volume",      row=4, col=1)
    return fig, df, buy_signals, sell_signals


def get_ai_st_dashboard(df_ai):
    latest, prev = df_ai.iloc[-1], df_ai.iloc[-2]
    direction = int(latest["AI_ST_Direction"])
    st_level  = float(latest["AI_Supertrend"])
    close     = float(latest["Close"])
    dist_pct  = abs(close - st_level) / close * 100
    is_new    = direction != int(prev["AI_ST_Direction"])

    dirs   = df_ai["AI_ST_Direction"].values
    streak = 1
    for i in range(len(dirs) - 2, -1, -1):
        if dirs[i] == direction: streak += 1
        else: break

    return dict(direction_raw=direction,
                direction="BULLISH 🟢" if direction == 1 else "BEARISH 🔴",
                regime=str(latest["AI_ST_Regime"]),
                multiplier=float(latest["AI_Multiplier"]),
                atr=float(latest["ATR"]),
                st_level=st_level,
                close=close,
                dist_pct=dist_pct,
                is_new_signal=is_new,
                streak=streak)


# ============================================================================
# INDEX ANALYZER CLASS
# ============================================================================

class IndexAnalyzer:
    """Comprehensive analyzer for NIFTY / BANKNIFTY and other NSE indices."""

    def __init__(self, index_name: str, period: str = "60d", interval: str = "1h"):
        self.index_name = index_name
        self.symbol     = INDEX_MAP.get(index_name, "^NSEI")
        self.period     = period
        self.interval   = interval
        self.data       = None
        self.ticker     = None
        self.pattern_detector = None

    def fetch_data(self) -> bool:
        try:
            import pytz
            ist = pytz.timezone("Asia/Kolkata")
            now_ist = datetime.now(ist)

            # ── Always use a fresh Ticker (no cache) ──────────────────────────
            self.ticker = yf.Ticker(self.symbol)

            if self.interval in ["1m", "5m", "15m", "30m", "1h"]:
                # Use start/end (not period=) — yfinance caches period= responses
                # and often returns only up to yesterday for intraday intervals.
                period_days = {
                    "5d": 5, "15d": 15, "30d": 30, "60d": 60,
                    "180d": 180, "730d": 730,
                }
                days_back = period_days.get(self.period, 60)
                start_dt  = (now_ist - timedelta(days=days_back)).strftime("%Y-%m-%d")
                end_dt    = (now_ist + timedelta(days=1)).strftime("%Y-%m-%d")

                self.data = self.ticker.history(
                    start=start_dt,
                    end=end_dt,
                    interval=self.interval,
                    prepost=False,
                    auto_adjust=True,
                    back_adjust=False,
                    repair=False,
                )
            else:
                self.data = self.ticker.history(
                    period=self.period,
                    interval=self.interval,
                    auto_adjust=True,
                )

            if self.data.empty:
                st.error(f"No data returned for {self.symbol}. Market may be closed or symbol invalid.")
                return False

            # ── Normalise the DatetimeIndex timezone ─────────────────────────
            # yfinance can return: tz-naive, UTC, or IST — handle all three.
            if self.interval in ["1m", "5m", "15m", "30m", "1h"]:
                idx = self.data.index

                # Step 1: get everything into IST
                try:
                    if idx.tz is None:
                        # Completely tz-naive — assume UTC then convert
                        self.data.index = idx.tz_localize("UTC").tz_convert("Asia/Kolkata")
                    elif str(idx.tz) in ("UTC", "utc", "Etc/UTC"):
                        self.data.index = idx.tz_convert("Asia/Kolkata")
                    else:
                        # Already some tz (could be IST already)
                        self.data.index = idx.tz_convert("Asia/Kolkata")
                except Exception:
                    # Last resort: strip whatever tz is there and re-localize as UTC→IST
                    self.data.index = self.data.index.tz_localize(None).tz_localize("UTC").tz_convert("Asia/Kolkata")

                # Step 2: filter pre/post market bars ONLY during live market hours.
                # Outside market hours keep all bars so analysis works anytime.
                import pytz as _pytz
                _ist     = _pytz.timezone("Asia/Kolkata")
                _now_ist = datetime.now(_ist)
                _wday    = _now_ist.weekday()          # 0=Mon … 6=Sun
                _hhmm    = _now_ist.hour * 60 + _now_ist.minute
                _market_open  = 9 * 60 + 15            # 09:15
                _market_close = 15 * 60 + 30           # 15:30
                _is_market_hours = (
                    _wday < 5 and                      # Mon–Fri
                    _market_open <= _hhmm <= _market_close
                )
                if _is_market_hours:
                    # Live session — strip pre/post market noise
                    self.data = self.data.between_time("09:15", "15:30")
                # else: market closed — keep all bars for full historical analysis

                # Step 3: strip tz so Plotly renders cleanly
                self.data.index = self.data.index.tz_localize(None)

            # Drop NaN OHLCV rows and zero-volume bars
            self.data = self.data.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            self.data = self.data[self.data["Volume"] > 0]
            self.data = self.data.sort_index()

            # Diagnostic: show what date range we actually got
            if len(self.data) > 0:
                latest_bar = self.data.index[-1]
                ist_now    = datetime.now(ist).replace(tzinfo=None)
                hours_old  = (ist_now - latest_bar).total_seconds() / 3600
                if hours_old > 28:
                    st.warning(
                        f"⚠️ Latest bar: {latest_bar.strftime('%d %b %Y %H:%M')} "
                        f"({hours_old:.0f}h ago). Market may be closed or data delayed."
                    )

            if len(self.data) < 20:
                st.error(
                    f"Not enough clean data bars ({len(self.data)}). "
                    f"Try a longer period or switch to 1d timeframe."
                )
                return False

            self.calculate_indicators()
            self.pattern_detector = IndexPatternDetector(self.data)
            return True

        except Exception as e:
            st.error(f"Error fetching data: {e}")
            import traceback
            st.code(traceback.format_exc(), language="python")
            return False

    def calculate_indicators(self):
        df = self.data

        # Trend MAs
        df["SMA_20"]  = SMAIndicator(df["Close"], 20).sma_indicator()
        df["SMA_21"]  = SMAIndicator(df["Close"], 21).sma_indicator()
        df["SMA_50"]  = SMAIndicator(df["Close"], 50).sma_indicator()
        df["SMA_200"] = SMAIndicator(df["Close"], 200).sma_indicator()
        df["EMA_9"]   = EMAIndicator(df["Close"], 9).ema_indicator()
        df["EMA_13"]  = EMAIndicator(df["Close"], 13).ema_indicator()
        df["EMA_20"]  = EMAIndicator(df["Close"], 20).ema_indicator()
        df["EMA_50"]  = EMAIndicator(df["Close"], 50).ema_indicator()
        df["EMA_200"] = EMAIndicator(df["Close"], 200).ema_indicator()

        # MACD
        macd = MACD(df["Close"])
        df["MACD"]        = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Hist"]   = macd.macd_diff()

        # RSI
        df["RSI"] = RSIIndicator(df["Close"]).rsi()

        # Stochastics
        stoch = StochasticOscillator(df["High"], df["Low"], df["Close"])
        df["Stoch_K"] = stoch.stoch()
        df["Stoch_D"] = stoch.stoch_signal()

        # Bollinger Bands
        bb = BollingerBands(df["Close"])
        df["BB_High"]  = bb.bollinger_hband()
        df["BB_Mid"]   = bb.bollinger_mavg()
        df["BB_Low"]   = bb.bollinger_lband()
        df["BB_Width"] = (df["BB_High"] - df["BB_Low"]) / df["BB_Mid"]

        # ATR + Volatility
        df["ATR"]           = AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
        df["ATR_Pct"]       = df["ATR"] / df["Close"] * 100
        df["Daily_Return"]  = df["Close"].pct_change() * 100
        df["Volatility_20"] = df["Daily_Return"].rolling(20).std()
        df["Volume_SMA"]    = df["Volume"].rolling(20).mean()

        # VWAP
        df["VWAP"] = ((df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3)
                      .cumsum() / df["Volume"].cumsum())

        # Pivot Points — intraday vs daily (no duplicate block)
        if self.interval in ["15m", "30m", "1h"]:
            window = 26  # ~one trading session
            df["Pivot"] = (df["High"].shift(window) + df["Low"].shift(window) + df["Close"].shift(window)) / 3
        else:
            df["Pivot"] = (df["High"] + df["Low"] + df["Close"]) / 3

        df["R1"] = 2 * df["Pivot"] - df["Low"]
        df["S1"] = 2 * df["Pivot"] - df["High"]
        df["R2"] = df["Pivot"] + (df["High"] - df["Low"])
        df["S2"] = df["Pivot"] - (df["High"] - df["Low"])

        # Camarilla pivots
        df["Cam_R3"] = df["Close"] + (df["High"] - df["Low"]) * 1.1666
        df["Cam_S3"] = df["Close"] - (df["High"] - df["Low"]) * 1.1666

        # OBV
        df["OBV"] = OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()

        # % from 52W high/low
        df["52W_High"] = df["High"].rolling(min(252, len(df))).max()
        df["52W_Low"]  = df["Low"].rolling(min(252, len(df))).min()
        df["Pct_from_52W_High"] = (df["Close"] - df["52W_High"]) / df["52W_High"] * 100
        df["Pct_from_52W_Low"]  = (df["Close"] - df["52W_Low"])  / df["52W_Low"]  * 100

        self.data = df

    # ── Signal generation ─────────────────────────────────────────────────────

    def get_trading_signal(self):
        df      = self.data
        current = df.iloc[-1]
        signals = []
        score   = 0

        if current["Close"] > current["SMA_20"]:
            signals.append("✅ Above 20 SMA (Short-term bullish)"); score += 1
        else:
            signals.append("❌ Below 20 SMA (Short-term bearish)"); score -= 1

        if current["Close"] > current["SMA_50"]:
            signals.append("✅ Above 50 SMA (Medium-term bullish)"); score += 1
        else:
            signals.append("❌ Below 50 SMA (Medium-term bearish)"); score -= 1

        if current["Close"] > current["SMA_200"]:
            signals.append("✅ Above 200 SMA (Long-term bullish)"); score += 2
        else:
            signals.append("❌ Below 200 SMA (Long-term bearish)"); score -= 2

        if current["MACD"] > current["MACD_Signal"]:
            signals.append("✅ MACD Bullish crossover"); score += 1
        else:
            signals.append("❌ MACD Bearish crossover"); score -= 1

        if current["RSI"] > 70:
            signals.append(f"⚠️ RSI Overbought ({current['RSI']:.1f})"); score -= 1
        elif current["RSI"] < 30:
            signals.append(f"✅ RSI Oversold ({current['RSI']:.1f}) — potential reversal"); score += 1
        else:
            signals.append(f"✅ RSI Neutral ({current['RSI']:.1f})")

        if current["Close"] > current["VWAP"]:
            signals.append("✅ Price above VWAP (Intraday bullish)"); score += 1
        else:
            signals.append("❌ Price below VWAP (Intraday bearish)"); score -= 1

        if current["Volume"] > current["Volume_SMA"]:
            signals.append("✅ Above-average volume"); score += 1
        else:
            signals.append("⚠️ Below-average volume")

        bb_pct = (current["Close"] - current["BB_Low"]) / (current["BB_High"] - current["BB_Low"])
        if bb_pct < 0.2:
            signals.append(f"📉 Near lower BB ({bb_pct*100:.0f}%) — oversold zone"); score += 1
        elif bb_pct > 0.8:
            signals.append(f"📈 Near upper BB ({bb_pct*100:.0f}%) — overbought zone"); score -= 1

        if   score >= 5:  overall = "🟢 STRONG BUY"
        elif score >= 3:  overall = "🟢 BUY"
        elif score >= -1: overall = "🟡 HOLD / NEUTRAL"
        elif score >= -3: overall = "🔴 SELL"
        else:             overall = "🔴 STRONG SELL"

        return overall, signals, score

    # ── Risk / Levels ─────────────────────────────────────────────────────────

    def get_key_levels(self):
        df      = self.data
        current = df.iloc[-1]
        atr     = float(current["ATR"])
        close   = float(current["Close"])
        return {
            "current":  close,
            "pivot":    float(current["Pivot"]),
            "r1": float(current["R1"]),  "s1": float(current["S1"]),
            "r2": float(current["R2"]),  "s2": float(current["S2"]),
            "cam_r3":   float(current["Cam_R3"]),
            "cam_s3":   float(current["Cam_S3"]),
            "bb_upper": float(current["BB_High"]),
            "bb_mid":   float(current["BB_Mid"]),
            "bb_lower": float(current["BB_Low"]),
            "vwap":     float(current["VWAP"]),
            "atr":      atr,
            "atr_pct":  float(current["ATR_Pct"]),
            "52w_high": float(current["52W_High"]),
            "52w_low":  float(current["52W_Low"]),
            "pct_from_52w_high": float(current["Pct_from_52W_High"]),
            "pct_from_52w_low":  float(current["Pct_from_52W_Low"]),
        }

    # ── Volume profile ────────────────────────────────────────────────────────

    def detect_volume_profile(self):
        df       = self.data.tail(200)
        num_bins = 50
        bins     = np.linspace(df["Low"].min(), df["High"].max(), num_bins)
        vap      = []
        for i in range(len(bins) - 1):
            mask = (df["Close"] >= bins[i]) & (df["Close"] < bins[i+1])
            vap.append(df.loc[mask, "Volume"].sum())
        vap = np.array(vap)
        if vap.sum() == 0:
            return {"poc_price": df["Close"].iloc[-1], "value_area_high": df["High"].max(),
                    "value_area_low": df["Low"].min(), "volume_distribution": vap,
                    "price_bins": bins, "high_volume_nodes": [], "low_volume_nodes": []}
        poc_idx   = np.argmax(vap)
        poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        target    = vap.sum() * 0.70
        sorted_i  = np.argsort(vap)[::-1]
        cum, va_idx = 0, []
        for idx in sorted_i:
            cum += vap[idx]; va_idx.append(idx)
            if cum >= target: break
        va_high = bins[max(va_idx) + 1]
        va_low  = bins[min(va_idx)]
        thr_hi  = np.percentile(vap[vap > 0], 80) if np.any(vap > 0) else 0
        thr_lo  = np.percentile(vap[vap > 0], 20) if np.any(vap > 0) else 0
        return {
            "poc_price": poc_price, "value_area_high": va_high,
            "value_area_low": va_low, "volume_distribution": vap,
            "price_bins": bins,
            "high_volume_nodes": bins[:-1][vap > thr_hi].tolist(),
            "low_volume_nodes":  bins[:-1][vap < thr_lo].tolist(),
        }

    # ── Index-specific stats ──────────────────────────────────────────────────

    def get_index_stats(self):
        df      = self.data
        cur     = df.iloc[-1]
        prev    = df.iloc[-2]
        chg     = cur["Close"] - prev["Close"]
        chg_pct = chg / prev["Close"] * 100

        if self.interval in ["15m", "30m"]:
            ret_1h = (cur["Close"] / df["Close"].iloc[-4]   - 1) * 100 if len(df) >= 4   else None
            ret_1d = (cur["Close"] / df["Close"].iloc[-26]  - 1) * 100 if len(df) >= 26  else None
            ret_1w = (cur["Close"] / df["Close"].iloc[-130] - 1) * 100 if len(df) >= 130 else None
            return {
                "current":    float(cur["Close"]),
                "change":     float(chg),
                "change_pct": float(chg_pct),
                "high":       float(cur["High"]),
                "low":        float(cur["Low"]),
                "open":       float(cur["Open"]),
                "volume":     float(cur["Volume"]),
                "ret_1w":     ret_1w,
                "ret_1m":     ret_1d,
                "ret_3m":     ret_1h,
                "ret_ytd":    None,
                "volatility_20d":      float(df["Volatility_20"].iloc[-1]),
                "atr_pct":             float(cur["ATR_Pct"]),
                "pct_from_52w_high":   float(cur["Pct_from_52W_High"]),
                "pct_from_52w_low":    float(cur["Pct_from_52W_Low"]),
            }
        else:
            ret_1w  = (cur["Close"] / df["Close"].iloc[-6]  - 1) * 100 if len(df) >= 6  else None
            ret_1m  = (cur["Close"] / df["Close"].iloc[-22] - 1) * 100 if len(df) >= 22 else None
            ret_3m  = (cur["Close"] / df["Close"].iloc[-66] - 1) * 100 if len(df) >= 66 else None
            ret_ytd = None
            try:
                jan1    = df[df.index.year == df.index[-1].year].iloc[0]["Close"]
                ret_ytd = (cur["Close"] / jan1 - 1) * 100
            except Exception:
                pass
            return {
                "current":    float(cur["Close"]),
                "change":     float(chg),
                "change_pct": float(chg_pct),
                "high":       float(cur["High"]),
                "low":        float(cur["Low"]),
                "open":       float(cur["Open"]),
                "volume":     float(cur["Volume"]),
                "ret_1w":     ret_1w,
                "ret_1m":     ret_1m,
                "ret_3m":     ret_3m,
                "ret_ytd":    ret_ytd,
                "volatility_20d":    float(df["Volatility_20"].iloc[-1]),
                "atr_pct":           float(cur["ATR_Pct"]),
                "pct_from_52w_high": float(cur["Pct_from_52W_High"]),
                "pct_from_52w_low":  float(cur["Pct_from_52W_Low"]),
            }


# ============================================================================
# CHART HELPERS
# ============================================================================

def create_candlestick_chart(analyzer: IndexAnalyzer, patterns=None):
    df  = analyzer.data.tail(200)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.50, 0.15, 0.15, 0.20],
                        subplot_titles=("Price Action — Indicators & Levels",
                                         "MACD", "RSI + Stochastics", "Volume"))

    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                  low=df["Low"], close=df["Close"], name="OHLC"), row=1, col=1)

    colours_ma = {"SMA_21": ("orange",1.2), "SMA_50": ("dodgerblue",1.5),
                  "SMA_200": ("tomato",2.0), "EMA_9": ("limegreen",1.2),
                  "EMA_20": ("cyan",1.2), "EMA_50": ("orchid",1.5)}
    for col, (clr, wid) in colours_ma.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col,
                                      line=dict(color=clr, width=wid)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["BB_High"], name="BB Upper",
                              line=dict(color="rgba(128,128,128,0.5)", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Low"],  name="BB Lower",
                              line=dict(color="rgba(128,128,128,0.5)", width=1, dash="dot"),
                              fill="tonexty", fillcolor="rgba(128,128,128,0.05)"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP",
                              line=dict(color="yellow", width=1.5, dash="dot")), row=1, col=1)

    kl = analyzer.get_key_levels()
    for name, val, clr in [("R2", kl["r2"], "red"), ("R1", kl["r1"], "salmon"),
                             ("Pivot", kl["pivot"], "white"), ("S1", kl["s1"], "lightgreen"),
                             ("S2", kl["s2"], "green")]:
        fig.add_hline(y=val, line_dash="dot", line_color=clr, line_width=1,
                      annotation_text=f"{name}: {val:.0f}", annotation_position="right",
                      row=1, col=1)

    if patterns:
        fig = _draw_index_patterns(fig, patterns, df)

    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],        name="MACD",   line=dict(color="blue",  width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(color="red",   width=1.5)), row=2, col=1)
    hist_c = ["green" if v >= 0 else "red" for v in df["MACD_Hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Hist", marker_color=hist_c), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"],     name="RSI",      line=dict(color="purple", width=2)),   row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_K"], name="Stoch %K", line=dict(color="orange", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_D"], name="Stoch %D", line=dict(color="pink",   width=1.5, dash="dot")), row=3, col=1)
    for lvl, clr in [(70,"red"),(30,"green"),(80,"darkred"),(20,"darkgreen")]:
        fig.add_hline(y=lvl, line_dash="dash", line_color=clr, line_width=0.8, row=3, col=1)

    vc = ["#26a69a" if df["Close"].iloc[i] >= df["Open"].iloc[i]
          else "#ef5350" for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],     name="Volume",   marker_color=vc), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Volume_SMA"], name="Vol SMA-20",
                              line=dict(color="orange", width=2)), row=4, col=1)

    fig.update_layout(title=f"{analyzer.index_name} — Technical Analysis Dashboard",
                      xaxis_rangeslider_visible=False, height=1200, showlegend=True,
                      hovermode="x unified",
                      plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font=dict(color="#fafafa"))
    fig.update_xaxes(showgrid=False, color="#555")
    fig.update_yaxes(showgrid=True, gridcolor="#1e1e2e", color="#999")
    return fig


def _draw_index_patterns(fig, patterns, df):
    for p in patterns:
        sig  = p.get("signal", "NEUTRAL")
        clr  = "#00cc66" if sig == "BULLISH" else "#ff4d4d" if sig == "BEARISH" else "#ffa500"
        for key, label, lclr, lstyle in [
            ("entry_price", "📍 ENTRY", clr,          "dash"),
            ("stop_loss",   "🛑 STOP",  "red",        "dot"),
            ("target_1",    "🎯 T1",    "green",      "dot"),
            ("target_2",    "🎯 T2",    "darkgreen",  "dot"),
        ]:
            v = p.get(key)
            if v and isinstance(v, (int, float)):
                fig.add_hline(y=v, line_dash=lstyle, line_color=lclr, line_width=1.5,
                              annotation_text=f"{label}: {v:.0f}",
                              annotation_position="right", row=1, col=1)
        if "support_zone" in p:
            fig.add_hrect(y0=p["support_zone"][0], y1=p["support_zone"][1],
                          fillcolor="rgba(0,255,100,0.08)", line_width=0, row=1, col=1)
        if "resistance_zone" in p:
            fig.add_hrect(y0=p["resistance_zone"][0], y1=p["resistance_zone"][1],
                          fillcolor="rgba(255,50,50,0.08)", line_width=0, row=1, col=1)
    return fig


def create_volume_profile_chart(analyzer: IndexAnalyzer):
    vp  = analyzer.detect_volume_profile()
    fig = go.Figure()
    pl  = (vp["price_bins"][:-1] + vp["price_bins"][1:]) / 2
    fig.add_trace(go.Bar(y=pl, x=vp["volume_distribution"], orientation="h",
                          name="Vol @ Price",
                          marker=dict(color="steelblue", line=dict(color="dodgerblue", width=0.5))))
    fig.add_hline(y=vp["poc_price"], line_dash="solid", line_color="gold",
                  annotation_text=f"POC: {vp['poc_price']:.0f}", line_width=2)
    fig.add_hrect(y0=vp["value_area_low"], y1=vp["value_area_high"],
                  fillcolor="rgba(255,100,100,0.15)", line_width=0,
                  annotation_text="Value Area (70%)", annotation_position="right")
    for hvn in vp["high_volume_nodes"][:3]:
        fig.add_hline(y=hvn, line_dash="dash", line_color="limegreen",
                      annotation_text="HVN", line_width=1)
    for lvn in vp["low_volume_nodes"][:3]:
        fig.add_hline(y=lvn, line_dash="dash", line_color="orange",
                      annotation_text="LVN", line_width=1)
    fig.update_layout(title="Volume Profile (last 200 bars)",
                      xaxis_title="Volume", yaxis_title="Index Level",
                      height=600, showlegend=True,
                      plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font=dict(color="#fafafa"))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#1e1e2e")
    return fig


# ============================================================================
# MCMC FAN CHART HELPERS
# ============================================================================

def create_mcmc_fan_chart(fs: Dict, index_name: str) -> go.Figure:
    import datetime
    dates = fs["forecast_dates"]
    fan   = fs["fan_bands"]
    paths = fs["sample_paths"]
    cp    = fs["current_price"]
    today = dates[0] - datetime.timedelta(days=1)
    all_d = [today] + list(dates)

    def pre(band): return [cp] + list(band)

    fig = go.Figure()
    for i in range(min(60, paths.shape[0])):
        fig.add_trace(go.Scatter(x=all_d, y=pre(paths[i]), mode="lines",
                                  line=dict(color="rgba(150,180,255,0.07)", width=1),
                                  showlegend=False, hoverinfo="skip"))
    for lo, hi, clr, name in [
        ("2.5", "97.5", "rgba(100,149,237,0.12)", "95% CI"),
        ("10",  "90",   "rgba(100,149,237,0.22)", "80% CI"),
        ("25",  "75",   "rgba(100,149,237,0.35)", "50% CI"),
    ]:
        fig.add_trace(go.Scatter(x=all_d + all_d[::-1],
                                  y=pre(fan[hi]) + pre(fan[lo])[::-1],
                                  fill="toself", fillcolor=clr,
                                  line=dict(color="rgba(0,0,0,0)"),
                                  name=name, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=all_d, y=pre(fan["50"]), mode="lines",
                              line=dict(color="#00e5ff", width=3), name="Median"))
    fig.add_hline(y=cp, line_dash="dot", line_color="#ffd600",
                  annotation_text=f"Current {cp:.0f}", annotation_position="left")
    fig.add_hline(y=fs["target_price"], line_dash="dash", line_color="#69ff47",
                  annotation_text=f"Target {fs['target_price']:.0f}", annotation_position="right")
    fig.update_layout(title=f"⛓️ MCMC Bayesian Forecast — {index_name} ({fs['forecast_days']}-Day)",
                      xaxis_title="Date", yaxis_title="Index Level",
                      height=550, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font=dict(color="#fafafa"), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1))
    fig.update_xaxes(showgrid=False, color="#555")
    fig.update_yaxes(showgrid=True, gridcolor="#1e1e2e", color="#999")
    return fig


def create_posterior_charts(mr: Dict, post: Dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Posterior P(μ | data) — Daily Drift",
                                         "Posterior P(σ | data) — Daily Volatility"))
    fig.add_trace(go.Histogram(x=mr["mu_samples"], nbinsx=80,
                                histnorm="probability density",
                                marker_color="rgba(100,149,237,0.55)",
                                name="μ posterior"), row=1, col=1)
    for x, clr, label in [(post["mu_mean"], "#00e5ff", "Mean"),
                           (post["mle_mu_daily"], "#ffd600", "MLE"),
                           (post["mu_ci_95_lo"],  "rgba(150,150,150,0.6)", ""),
                           (post["mu_ci_95_hi"],  "rgba(150,150,150,0.6)", "95% CI")]:
        fig.add_vline(x=x, line_color=clr, line_width=1.5,
                      line_dash="dot" if label == "MLE" else ("dash" if not label else "solid"),
                      annotation_text=label, row=1, col=1)
    fig.add_trace(go.Histogram(x=mr["sigma_samples"], nbinsx=80,
                                histnorm="probability density",
                                marker_color="rgba(255,100,100,0.55)",
                                name="σ posterior"), row=1, col=2)
    for x, clr, label in [(post["sigma_mean"], "#ff6d00", "Mean"),
                           (post["mle_sigma_daily"], "#ffd600", "MLE"),
                           (post["sigma_ci_95_lo"], "rgba(150,150,150,0.6)", ""),
                           (post["sigma_ci_95_hi"], "rgba(150,150,150,0.6)", "95% CI")]:
        fig.add_vline(x=x, line_color=clr, line_width=1.5,
                      line_dash="dot" if label == "MLE" else ("dash" if not label else "solid"),
                      annotation_text=label, row=1, col=2)
    fig.update_layout(title="Posterior Parameter Distributions", height=380,
                      plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font=dict(color="#fafafa"))
    fig.update_xaxes(showgrid=False, color="#555")
    fig.update_yaxes(showgrid=True, gridcolor="#1e1e2e", color="#999")
    return fig


def create_trace_plots(mr: Dict) -> go.Figure:
    mu_c, sig_c = mr["mu_chains"], mr["sigma_chains"]
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Trace: μ (daily drift)", "Trace: σ (daily vol)"),
                        vertical_spacing=0.12)
    pal = ["#00e5ff","#69ff47","#ff6d00","#d500f9","#ffd600","#ff1744"]
    for c in range(mu_c.shape[0]):
        clr = pal[c % len(pal)]
        x   = list(range(len(mu_c[c])))
        fig.add_trace(go.Scatter(x=x, y=mu_c[c],  mode="lines",
                                  line=dict(color=clr, width=0.8),
                                  name=f"Chain {c+1}", legendgroup=f"c{c}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=sig_c[c], mode="lines",
                                  line=dict(color=clr, width=0.8),
                                  name=f"Chain {c+1} σ", legendgroup=f"c{c}",
                                  showlegend=False), row=2, col=1)
    fig.update_layout(title="MCMC Trace Plots — Convergence Check", height=450,
                      plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font=dict(color="#fafafa"), hovermode="x unified")
    fig.update_xaxes(showgrid=False, color="#555", title_text="Iteration")
    fig.update_yaxes(showgrid=True, gridcolor="#1e1e2e", color="#999")
    return fig


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown('<div class="main-header">📊 NIFTY & BANKNIFTY Index Analyzer</div>',
                unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:gray;">Institutional-Grade Index Pattern & Forecasting Suite</p>',
                unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Settings")

        index_name = st.selectbox("Select Index", list(INDEX_MAP.keys()), index=0)

        timeframe = st.selectbox(
            "Timeframe",
            options=["15m", "30m", "1h", "1d", "1wk"],
            index=2,
            help="15m/30m/1h = intraday | 1d = daily | 1wk = weekly"
        )

        period_options = {
            "15m":  ["5d", "15d", "30d", "60d"],
            "30m":  ["5d", "15d", "30d", "60d"],
            "1h":   ["15d", "30d", "60d", "180d", "730d"],
            "1d":   ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            "1wk":  ["6mo", "1y", "2y", "5y"],
        }

        period = st.selectbox(
            "Lookback Period",
            options=period_options[timeframe],
            index=2
        )
        show_patterns_on_chart = st.checkbox("Show Patterns on Chart", value=True)

        st.markdown("---")
        st.markdown("### 🤖 AI Adaptive Supertrend")
        ai_atr = st.slider("ATR Period",       5, 30, 10, 1)
        ai_k   = st.slider("K-Means Clusters", 2,  5,  3, 1)
        st.caption("Auto-selects ATR multiplier per volatility regime")
        st.markdown("### 📊 Index Trading Notes")
        st.markdown("""
- No PE/PB/Div Yield (index metrics)
- Pivot levels key for options writers
- VIX correlation matters
- Volume = Futures + Cash combined
- PCR (Put/Call ratio) signals
        """)

        analyze_btn = st.button("🔍 Analyze Index", type="primary", use_container_width=True)

    if analyze_btn:
        with st.spinner(f"📡 Fetching {index_name} data…"):
            analyzer = IndexAnalyzer(index_name, period=period, interval=timeframe)
            ok = analyzer.fetch_data()

        if not ok:
            st.error(f"❌ Failed to load data for {index_name}. Check symbol or period.")
            return

        with st.spinner("🔎 Running pattern detection…"):
            trend_pat    = analyzer.pattern_detector.detect_all_trend_patterns()
            reversal_pat = analyzer.pattern_detector.detect_all_reversal_patterns()
            advanced_pat = analyzer.pattern_detector.detect_all_advanced_patterns()
            all_patterns = trend_pat + reversal_pat + advanced_pat

        with st.spinner("🤖 AI Adaptive Supertrend (K-Means)…"):
            try:
                ai_fig, df_ai, ai_buys, ai_sells = create_adaptive_supertrend_chart(
                    analyzer.data, atr_period=ai_atr, n_clusters=ai_k)
                ai_dash = get_ai_st_dashboard(df_ai)
                ai_ok   = True
            except Exception as e:
                st.warning(f"AI Supertrend: {e}")
                ai_ok = False

        st.markdown('<div class="sub-header">📈 Index Overview</div>', unsafe_allow_html=True)
        stats = analyzer.get_index_stats()

        badge_cls = "index-badge-bnifty" if "BANK" in index_name else "index-badge-nifty"
        st.markdown(f'<span class="{badge_cls}">{index_name}</span>', unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        delta_str = f"{stats['change']:+.2f} ({stats['change_pct']:+.2f}%)"
        with c1: st.metric("Index Level", f"{stats['current']:.2f}", delta_str)
        with c2: st.metric("Day High",    f"{stats['high']:.2f}")
        with c3: st.metric("Day Low",     f"{stats['low']:.2f}")
        with c4: st.metric("52W High",    f"{analyzer.data['52W_High'].iloc[-1]:.2f}",
                            f"{stats['pct_from_52w_high']:+.2f}% from ATH")
        with c5: st.metric("52W Low",     f"{analyzer.data['52W_Low'].iloc[-1]:.2f}",
                            f"{stats['pct_from_52w_low']:+.2f}% from ATL")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            if stats["ret_1w"] is not None: st.metric("1W Return",  f"{stats['ret_1w']:+.2f}%")
        with r2:
            if stats["ret_1m"] is not None: st.metric("1M Return",  f"{stats['ret_1m']:+.2f}%")
        with r3:
            if stats["ret_3m"] is not None: st.metric("3M Return",  f"{stats['ret_3m']:+.2f}%")
        with r4:
            if stats["ret_ytd"] is not None: st.metric("YTD Return", f"{stats['ret_ytd']:+.2f}%")

        v1, v2, v3 = st.columns(3)
        with v1: st.metric("20D Volatility (annualised)", f"{stats['volatility_20d']*np.sqrt(252):.1f}%")
        with v2: st.metric("Daily ATR %",  f"{stats['atr_pct']:.2f}%")
        with v3: st.metric("Daily Volume", f"{stats['volume']/1e7:.2f} Cr" if stats['volume'] > 1e7
                                            else f"{stats['volume']:.0f}")

        st.markdown('<div class="sub-header">🎯 Key Technical Levels</div>', unsafe_allow_html=True)
        kl = analyzer.get_key_levels()
        la, lb, lc, ld = st.columns(4)
        with la:
            st.markdown("**Pivot Points**")
            st.metric("R2",    f"{kl['r2']:.0f}")
            st.metric("R1",    f"{kl['r1']:.0f}")
            st.metric("Pivot", f"{kl['pivot']:.0f}")
            st.metric("S1",    f"{kl['s1']:.0f}")
            st.metric("S2",    f"{kl['s2']:.0f}")
        with lb:
            st.markdown("**Bollinger Bands**")
            st.metric("BB Upper", f"{kl['bb_upper']:.0f}")
            st.metric("BB Mid",   f"{kl['bb_mid']:.0f}")
            st.metric("BB Lower", f"{kl['bb_lower']:.0f}")
            st.metric("BB Width", f"{(kl['bb_upper']-kl['bb_lower'])/kl['bb_mid']*100:.2f}%")
        with lc:
            st.markdown("**Camarilla Pivots**")
            st.metric("Cam R3", f"{kl['cam_r3']:.0f}")
            st.metric("VWAP",   f"{kl['vwap']:.0f}")
            st.metric("Cam S3", f"{kl['cam_s3']:.0f}")
            st.metric("ATR",    f"{kl['atr']:.2f} ({kl['atr_pct']:.2f}%)")
        with ld:
            st.markdown("**Year Range**")
            st.metric("52W High",  f"{kl['52w_high']:.0f}", f"{kl['pct_from_52w_high']:+.2f}%")
            st.metric("52W Low",   f"{kl['52w_low']:.0f}",  f"{kl['pct_from_52w_low']:+.2f}%")
            mid52 = (kl["52w_high"] + kl["52w_low"]) / 2
            st.metric("Mid-Range", f"{mid52:.0f}")

        if ai_ok:
            st.markdown('<div class="sub-header">🤖 AI Adaptive Supertrend — Live Signal</div>',
                        unsafe_allow_html=True)
            if ai_dash["is_new_signal"]:
                if ai_dash["direction_raw"] == 1:
                    st.success("🚀 **NEW BUY SIGNAL** — AI Supertrend flipped **BULLISH**!")
                else:
                    st.error("🔻 **NEW SELL SIGNAL** — AI Supertrend flipped **BEARISH**!")

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: st.metric("AI-ST Direction",
                                "BULLISH" if ai_dash["direction_raw"] == 1 else "BEARISH",
                                delta="🟢 Uptrend" if ai_dash["direction_raw"] == 1 else "🔴 Downtrend")
            with c2: st.metric("Volatility Regime",   ai_dash["regime"])
            with c3: st.metric("Adaptive Multiplier", f"{ai_dash['multiplier']:.1f}×")
            with c4: st.metric("AI-ST Level",         f"{ai_dash['st_level']:.0f}")
            with c5: st.metric("Distance from Index", f"{ai_dash['dist_pct']:.2f}%",
                                delta=f"{ai_dash['streak']} bars in trend")

            cb1, cb2, cb3 = st.columns(3)
            with cb1: st.info(f"📊 AI-ST Buy Signals (last 200): **{len(ai_buys)}**")
            with cb2: st.info(f"📊 AI-ST Sell Signals (last 200): **{len(ai_sells)}**")
            with cb3:
                if len(ai_buys) > 0 and (len(ai_sells) == 0 or ai_buys.index[-1] > ai_sells.index[-1]):
                    last = "BUY 🟢"
                elif len(ai_sells) > 0:
                    last = "SELL 🔴"
                else:
                    last = "None"
                st.info(f"📊 Last Signal: **{last}**")

        st.markdown('<div class="sub-header">🎯 Trading Signal</div>', unsafe_allow_html=True)
        overall, signals, score = analyzer.get_trading_signal()
        s1, s2 = st.columns([1, 2])
        with s1:
            st.markdown(f"### {overall}")
            st.markdown(f"**Signal Score: {score}**")
        with s2:
            for sig in signals:
                st.markdown(sig)

        st.markdown('<div class="sub-header">📈 Index Pattern Detection</div>', unsafe_allow_html=True)
        pt1, pt2, pt3 = st.tabs(["Trend Continuation", "Reversal Patterns", "Advanced / Index-Specific"])

        def render_patterns(plist):
            if not plist:
                st.info("No patterns detected in current timeframe.")
                return
            bullish = [p for p in plist if p.get("signal") == "BULLISH"]
            bearish = [p for p in plist if p.get("signal") == "BEARISH"]
            neutral = [p for p in plist if p.get("signal") not in ["BULLISH", "BEARISH"]]
            for group, label in [(bullish, "🟢 Bullish"), (bearish, "🔴 Bearish"),
                                  (neutral, "⚡ Neutral/Breakout")]:
                if group:
                    st.markdown(f"### {label}")
                    for p in group:
                        icon = "🔻" if p.get("signal") == "BEARISH" else "🔹"
                        with st.expander(f"{icon} {p['pattern']} — {p['signal']}", expanded=True):
                            st.markdown(f"**{p['description']}**")
                            st.markdown(f"**Action:** {p['action']}")
                            if p.get("signal") == "BEARISH":
                                st.warning("⚠️ SHORT opportunity — consider options strategies")
                            c1, c2, c3, c4 = st.columns(4)
                            for col, key, label_k in [(c1, "entry_price", "📍 Entry"),
                                                       (c2, "stop_loss",   "🛑 Stop"),
                                                       (c3, "target_1",    "🎯 T1"),
                                                       (c4, "target_2",    "🎯 T2")]:
                                v = p.get(key)
                                if v:
                                    with col:
                                        st.markdown(f"**{label_k}:** {v:.0f}" if isinstance(v, float) else f"**{label_k}:** {v}")

        with pt1:
            st.success(f"✅ {len(trend_pat)} trend pattern(s) found")
            render_patterns(trend_pat)
        with pt2:
            st.success(f"✅ {len(reversal_pat)} reversal pattern(s) found")
            render_patterns(reversal_pat)
        with pt3:
            st.success(f"✅ {len(advanced_pat)} advanced pattern(s) found")
            render_patterns(advanced_pat)

        st.markdown('<div class="sub-header">📊 Charts & Forecasts</div>', unsafe_allow_html=True)
        ct1, ct2, ct3, ct4, ct5 = st.tabs([
            "Price Action & Indicators",
            "Volume Profile",
            "🤖 AI Adaptive Supertrend",
            "⛓️ MCMC Bayesian Forecast",
            "🎲 HMM Forecast",
        ])

        with ct1:
            if show_patterns_on_chart and all_patterns:
                st.info(f"📌 {len(all_patterns)} patterns overlaid on chart")
            fig_c = create_candlestick_chart(analyzer, all_patterns if show_patterns_on_chart else None)
            st.plotly_chart(fig_c, use_container_width=True)

        with ct2:
            st.plotly_chart(create_volume_profile_chart(analyzer), use_container_width=True)
            vp = analyzer.detect_volume_profile()
            v1, v2, v3 = st.columns(3)
            with v1: st.metric("Point of Control (POC)", f"{vp['poc_price']:.0f}")
            with v2: st.metric("Value Area High",         f"{vp['value_area_high']:.0f}")
            with v3: st.metric("Value Area Low",          f"{vp['value_area_low']:.0f}")

        with ct3:
            if ai_ok:
                st.plotly_chart(ai_fig, use_container_width=True)
                st.markdown("### 📋 Recent AI Supertrend Signals")
                rows = []
                for idx, row in ai_buys.iterrows():
                    rows.append({"Date": idx.strftime("%Y-%m-%d %H:%M"), "Type": "🟢 BUY",
                                 "Level": f"{row['Close']:.0f}", "AI-ST": f"{row['AI_Supertrend']:.0f}",
                                 "Mult": f"{row['AI_Multiplier']:.1f}×", "Regime": row["AI_ST_Regime"]})
                for idx, row in ai_sells.iterrows():
                    rows.append({"Date": idx.strftime("%Y-%m-%d %H:%M"), "Type": "🔴 SELL",
                                 "Level": f"{row['Close']:.0f}", "AI-ST": f"{row['AI_Supertrend']:.0f}",
                                 "Mult": f"{row['AI_Multiplier']:.1f}×", "Regime": row["AI_ST_Regime"]})
                if rows:
                    sdf = pd.DataFrame(rows).sort_values("Date", ascending=False).head(15)
                    st.dataframe(sdf, use_container_width=True, hide_index=True)

                st.markdown("### 🎯 Regime Distribution (last 200 bars)")
                rc    = df_ai["AI_ST_Regime"].value_counts()
                rcols = st.columns(min(len(rc), 3))
                emo   = {"Low Vol": "🟢", "Medium Vol": "🟡", "High Vol": "🔴"}
                for j, (rn, rv) in enumerate(rc.items()):
                    if j < 3:
                        with rcols[j]:
                            st.metric(f"{emo.get(rn,'⚫')} {rn}", f"{rv/len(df_ai)*100:.1f}%",
                                      f"{rv} bars")
            else:
                st.error("AI Supertrend unavailable. Use longer period.")

        with ct4:
            st.markdown("### ⛓️ MCMC Bayesian Price Forecast")
            mc1, mc2, mc3 = st.columns(3)
            with mc1: mcmc_days    = st.slider("Forecast Days",    10, 60, 30, 5,    key="mcd")
            with mc2: mcmc_chains  = st.slider("MCMC Chains",       2,  4,  4, 1,    key="mcc")
            with mc3: mcmc_samples = st.slider("Samples / Chain", 1000, 5000, 3000, 500, key="mcs")

            with st.spinner("⛓️ Running MCMC sampler…"):
                try:
                    mcmc_out = run_mcmc_analysis(analyzer.data, forecast_days=mcmc_days,
                                                  n_samples=mcmc_samples,
                                                  n_warmup=max(500, mcmc_samples // 2),
                                                  n_chains=mcmc_chains, n_paths=2000, seed=42)
                    mcmc_ok = True
                except Exception as e:
                    st.error(f"MCMC error: {e}"); mcmc_ok = False

            if mcmc_ok:
                fs   = mcmc_out["forecast_summary"]
                post = mcmc_out["posterior"]
                risk = mcmc_out["risk_metrics"]
                diag = mcmc_out["diagnostics"]
                mr   = mcmc_out["mcmc_result"]

                if diag["converged"]:
                    st.success(f"✅ MCMC Converged — R-hat μ={diag['r_hat_mu']:.4f}, "
                               f"σ={diag['r_hat_sigma']:.4f} | "
                               f"ESS μ={diag['ess_mu']:.0f}, σ={diag['ess_sigma']:.0f} | "
                               f"Accept={diag['accept_rate']:.1%}")
                else:
                    st.warning("⚠️ Convergence uncertain — try more samples.")

                if fs["direction"] == "BULLISH":
                    st.success(f"📈 **BULLISH** — Target {fs['target_price']:.0f} "
                               f"({fs['expected_return']:+.2f}%) | P(profit)={risk['prob_profit']:.1%}")
                elif fs["direction"] == "BEARISH":
                    st.error(f"📉 **BEARISH** — Target {fs['target_price']:.0f} "
                             f"({fs['expected_return']:+.2f}%) | P(loss>5%)={risk['prob_loss_5pct']:.1%}")
                else:
                    st.info(f"📊 **NEUTRAL** — Range {fs['ci_95_low']:.0f}–{fs['ci_95_high']:.0f}")

                k1, k2, k3, k4, k5, k6 = st.columns(6)
                with k1: st.metric("Current",     f"{fs['current_price']:.0f}")
                with k2: st.metric("Target",      f"{fs['target_price']:.0f}", f"{fs['expected_return']:+.2f}%")
                with k3: st.metric("95% CI Low",  f"{fs['ci_95_low']:.0f}")
                with k4: st.metric("95% CI High", f"{fs['ci_95_high']:.0f}")
                with k5: st.metric("Ann Drift",   f"{fs['ann_drift_mean']:+.1f}%")
                with k6: st.metric("Ann Vol",     f"{fs['ann_volatility']:.1f}%")

                st.plotly_chart(create_mcmc_fan_chart(fs, index_name), use_container_width=True)
                st.plotly_chart(create_posterior_charts(mr, post),     use_container_width=True)

                st.markdown("#### ⚠️ Bayesian Risk Metrics")
                rk1, rk2, rk3, rk4 = st.columns(4)
                with rk1:
                    st.metric("P(Profit)",    f"{risk['prob_profit']:.1%}")
                    st.metric("P(Gain >5%)",  f"{risk['prob_gain_5pct']:.1%}")
                with rk2:
                    st.metric("P(Gain >10%)", f"{risk['prob_gain_10pct']:.1%}")
                    st.metric("P(Loss >5%)",  f"{risk['prob_loss_5pct']:.1%}")
                with rk3:
                    st.metric("95% VaR",      f"{risk.get('var_95',0)*100:+.2f}%")
                    st.metric("95% CVaR",     f"{risk.get('cvar_95',0)*100:+.2f}%")
                with rk4:
                    st.metric("50% CI",       f"{fs['ci_50_low']:.0f}–{fs['ci_50_high']:.0f}")
                    st.metric("80% CI",       f"{fs['ci_80_low']:.0f}–{fs['ci_80_high']:.0f}")

                st.plotly_chart(create_trace_plots(mr), use_container_width=True)
                st.caption("✅ Good mixing = 'fuzzy caterpillars'. Diverging chains → increase samples/warmup.")

        with ct5:
            st.markdown("### 🎲 Hidden Markov Model (HMM) Forecast")
            with st.spinner("Running HMM analysis…"):
                hmm_r = run_hmm_analysis(analyzer.data, forecast_days=30)
            fc    = hmm_r["forecast"]
            char  = hmm_r["characteristics"]
            strat = hmm_r["strategy"]
            pers  = hmm_r["persistence"]

            h1, h2, h3, h4 = st.columns(4)
            with h1:
                st.metric("Current", f"{fc['current_price']:.0f}")
                st.metric("Target",  f"{fc['target_price']:.0f}", f"{fc['expected_return']:.2f}%")
            with h2:
                st.metric("Best Case",  f"{fc['best_case']:.0f}",
                          f"+{(fc['best_case']-fc['current_price'])/fc['current_price']*100:.1f}%")
                st.metric("Worst Case", f"{fc['worst_case']:.0f}",
                          f"{(fc['worst_case']-fc['current_price'])/fc['current_price']*100:.1f}%")
            with h3:
                st.metric("Signal",    strat["signal"])
                st.metric("Direction", fc["direction"])
            with h4:
                st.metric("Confidence", fc["confidence_level"])
                st.metric("Exp Vol",    f"{fc['expected_volatility']:.2f}%")

            st.markdown("---")
            st.markdown("#### 🔄 Regime Analysis")
            re1, re2, re3 = st.columns(3)
            with re1:
                st.markdown("**Current State**")
                cs   = fc["current_state"]
                prob = fc["current_state_probability"]
                if cs == "BULL":   st.success(f"🟢 {cs} ({prob:.1%})")
                elif cs == "BEAR": st.error(f"🔴 {cs} ({prob:.1%})")
                else:              st.info(f"🟡 {cs} ({prob:.1%})")
                st.markdown(f"Avg duration: **{pers[cs]['avg_duration']:.0f} days**")
            with re2:
                st.markdown("**Dominant Future**")
                dom = fc["dominant_regime"]
                dc  = fc["regime_confidence"]
                if dom == "BULL":  st.success(f"🟢 {dom} ({dc:.1%})")
                elif dom == "BEAR":st.error(f"🔴 {dom} ({dc:.1%})")
                else:              st.info(f"🟡 {dom} ({dc:.1%})")
                st.markdown(f"Bull: {fc['bull_probability']:.1%}  |  "
                            f"Bear: {fc['bear_probability']:.1%}  |  "
                            f"Side: {fc['sideways_probability']:.1%}")
            with re3:
                st.markdown("**Transition Matrix**")
                tm = pd.DataFrame(fc["state_transition_matrix"],
                                  columns=["→Bull", "→Bear", "→Side"],
                                  index=["Bull→", "Bear→", "Side→"])
                st.dataframe(tm.style.format("{:.1%}"), use_container_width=True)

            if fc["direction"] == "BULLISH":
                st.success(f"📈 BULLISH — Target {fc['target_price']:.0f} in 30 days "
                           f"({fc['expected_return']:.2f}% gain)")
            elif fc["direction"] == "BEARISH":
                st.error(f"📉 BEARISH — Target {fc['target_price']:.0f} in 30 days "
                         f"({fc['expected_return']:.2f}% loss)")
            else:
                st.info(f"📊 NEUTRAL — Range {fc['worst_case']:.0f}–{fc['best_case']:.0f}")

        st.markdown('<div class="sub-header">📋 Technical Indicators Summary</div>',
                    unsafe_allow_html=True)
        cur = analyzer.data.iloc[-1]
        ind_df = pd.DataFrame([
            {"Indicator": "RSI (14)",      "Value": f"{cur['RSI']:.2f}",
             "Signal": "Overbought" if cur["RSI"] > 70 else "Oversold" if cur["RSI"] < 30 else "Neutral"},
            {"Indicator": "MACD",          "Value": f"{cur['MACD']:.2f}",
             "Signal": "Bullish" if cur["MACD"] > cur["MACD_Signal"] else "Bearish"},
            {"Indicator": "Stoch %K",      "Value": f"{cur['Stoch_K']:.2f}",
             "Signal": "Overbought" if cur["Stoch_K"] > 80 else "Oversold" if cur["Stoch_K"] < 20 else "Neutral"},
            {"Indicator": "BB Width %",    "Value": f"{cur['BB_Width']*100:.2f}%",
             "Signal": "Squeeze" if cur["BB_Width"] < 0.04 else "Expansion" if cur["BB_Width"] > 0.10 else "Normal"},
            {"Indicator": "ATR %",         "Value": f"{cur['ATR_Pct']:.2f}%",
             "Signal": "High Vol" if cur["ATR_Pct"] > 2 else "Low Vol" if cur["ATR_Pct"] < 0.8 else "Normal"},
            {"Indicator": "OBV Trend",     "Value": f"{cur['OBV']:.0f}",
             "Signal": "Accumulation" if cur["OBV"] > analyzer.data["OBV"].mean() else "Distribution"},
            {"Indicator": "Above VWAP",    "Value": f"{cur['Close']:.0f} vs {cur['VWAP']:.0f}",
             "Signal": "Bullish" if cur["Close"] > cur["VWAP"] else "Bearish"},
            {"Indicator": "20D Vol (Ann)", "Value": f"{cur['Volatility_20']*np.sqrt(252):.1f}%",
             "Signal": "Elevated" if cur["Volatility_20"] * np.sqrt(252) > 20 else "Normal"},
        ])
        st.dataframe(ind_df, use_container_width=True, hide_index=True)
        st.success(f"✅ Analysis complete for **{index_name}**")

        # ── Multi-Timeframe Confluence ─────────────────────────────────────────
        st.markdown("### 📐 Multi-Timeframe Confluence")
        mtf_col1, mtf_col2, mtf_col3 = st.columns(3)

        timeframes_to_check = {
            "15m (Intraday)": ("60d", "15m"),
            "1h  (Swing)":    ("60d", "1h"),
            "1d  (Trend)":    ("1y",  "1d"),
        }

        for col, (tf_label, (tf_period, tf_interval)) in zip(
                [mtf_col1, mtf_col2, mtf_col3], timeframes_to_check.items()):
            with col:
                st.markdown(f"**{tf_label}**")
                with st.spinner(f"Loading {tf_label}..."):
                    tf_analyzer = IndexAnalyzer(index_name, period=tf_period, interval=tf_interval)
                    if tf_analyzer.fetch_data():
                        signal, _, score = tf_analyzer.get_trading_signal()
                        rsi_val   = tf_analyzer.data["RSI"].iloc[-1]
                        macd_bull = tf_analyzer.data["MACD"].iloc[-1] > tf_analyzer.data["MACD_Signal"].iloc[-1]
                        above_200 = tf_analyzer.data["Close"].iloc[-1] > tf_analyzer.data["SMA_200"].iloc[-1]
                        st.markdown(f"**{signal}** (score: {score})")
                        st.markdown(f"RSI: `{rsi_val:.1f}`")
                        st.markdown(f"MACD: {'🟢 Bull' if macd_bull else '🔴 Bear'}")
                        st.markdown(f"200 SMA: {'✅ Above' if above_200 else '❌ Below'}")


if __name__ == "__main__":
    main()
