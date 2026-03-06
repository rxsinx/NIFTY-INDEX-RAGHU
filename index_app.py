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

# ── Chart theme constants ─────────────────────────────────────────────────────
CHART_THEME = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="#111111", family="Arial, sans-serif"),
)
AXIS_STYLE = dict(
    showgrid=True, gridcolor="#e8e8e8", linecolor="#cccccc",
    zeroline=False, color="#333333"
)

def _style_subplot_titles(fig):
    """Give each subplot title a light pill background so it sits clearly above its panel."""
    for ann in fig.layout.annotations:
        ann.update(
            font=dict(size=12, color="#1e3a5f", family="Arial, sans-serif"),
            bgcolor="rgba(219,234,254,0.85)",
            bordercolor="#93c5fd",
            borderwidth=1,
            borderpad=4,
        )
    return fig

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

# ── Shared chart theme ────────────────────────────────────────────────────────
CHART_THEME = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="#111111", family="Arial, sans-serif"),
)
AXIS_STYLE = dict(showgrid=True, gridcolor="#e8e8e8", linecolor="#cccccc",
                  zeroline=False, color="#333333")

# ── Index symbol map ──────────────────────────────────────────────────────────
INDEX_MAP = {
    "NIFTY 50":     "^NSEI",
    "BANKNIFTY":    "^NSEBANK",
    "NIFTY IT":     "^CNXIT",
    "NIFTY FMCG":   "^CNXFMCG",
    "NIFTY AUTO":   "^CNXAUTO",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY MIDCAP": "^NSEMDCP50",    
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

    # subplot_titles places each label directly above its own panel
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.50, 0.18, 0.14, 0.18],
        subplot_titles=(
            "🤖 AI Adaptive Supertrend — K-Means",
            "ATR by Volatility Regime",
            "Adaptive Multiplier",
            "Volume"
        )
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing_line_color="#16a34a", decreasing_line_color="#dc2626"
    ), row=1, col=1)

    bull_st = df["AI_Supertrend"].where(df["AI_ST_Direction"] == 1)
    bear_st = df["AI_Supertrend"].where(df["AI_ST_Direction"] == -1)
    fig.add_trace(go.Scatter(x=df.index, y=bull_st, name="AI-ST Bull",
                              line=dict(color="#16a34a", width=2.5), mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bear_st, name="AI-ST Bear",
                              line=dict(color="#dc2626", width=2.5), mode="lines"), row=1, col=1)

    cluster_colours = {
        0: "rgba(22,163,74,0.07)",
        1: "rgba(234,179,8,0.07)",
        2: "rgba(220,38,38,0.07)"
    }
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
        fig.add_trace(go.Scatter(
            x=buy_signals.index, y=buy_signals["Low"] * 0.993,
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=14, color="#16a34a",
                        line=dict(color="white", width=1)),
            text=["BUY"] * len(buy_signals), textposition="bottom center",
            textfont=dict(color="#15803d", size=9), name="Buy Signal"
        ), row=1, col=1)
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals.index, y=sell_signals["High"] * 1.007,
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=14, color="#dc2626",
                        line=dict(color="white", width=1)),
            text=["SELL"] * len(sell_signals), textposition="top center",
            textfont=dict(color="#b91c1c", size=9), name="Sell Signal"
        ), row=1, col=1)

    c_line  = {0: "#16a34a", 1: "#ca8a04", 2: "#dc2626"}
    c_label = {0: "ATR Low Vol", 1: "ATR Med Vol", 2: "ATR High Vol"}
    for c_id in range(n_clusters):
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ATR"].where(df["ATR_Cluster"] == c_id),
            name=c_label.get(c_id, f"ATR Cluster {c_id}"),
            line=dict(color=c_line.get(c_id, "#888"), width=2), mode="lines"
        ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["AI_Multiplier"], name="Multiplier",
        line=dict(color="#7c3aed", width=2), mode="lines",
        fill="tozeroy", fillcolor="rgba(124,58,237,0.10)"
    ), row=3, col=1)

    vol_c = ["#16f02f" if df["Close"].iloc[i] >= df["Open"].iloc[i]
             else "#f51818" for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                          marker_color=vol_c, opacity=0.7), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Volume"].rolling(20).mean(),
        name="Vol MA-20", line=dict(color="#fff940", width=1.5, dash="dot")
    ), row=4, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False, height=1100,
        showlegend=True, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1,
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#ddd", borderwidth=1),
        **CHART_THEME
    )
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    fig.update_yaxes(title_text="Price",      title_font=dict(size=11), row=1, col=1)
    fig.update_yaxes(title_text="ATR",        title_font=dict(size=11), row=2, col=1)
    fig.update_yaxes(title_text="Multiplier", title_font=dict(size=11), row=3, col=1)
    fig.update_yaxes(title_text="Volume",     title_font=dict(size=11), row=4, col=1)


    # ── Black background for rows 2 (ATR), 3 (Multiplier), 4 (Volume) ─────────
    
    dark_tick  = dict(color="#302f2f")
    dark_grid  = "#333333"
    dark_zero  = "#555555"
    dark_title = dict(color="#5e5a5a", size=11)

    for axis_n, row_n in [("yaxis2", 2), ("yaxis3", 3), ("yaxis4", 4)]:
        fig.layout[axis_n].update(
            gridcolor=dark_grid, zerolinecolor=dark_zero,
            tickfont=dark_tick, title_font=dark_title,
        )
    for axis_n in ["xaxis2", "xaxis3", "xaxis4"]:
        fig.layout[axis_n].update(
            gridcolor=dark_grid, tickfont=dark_tick,
        )

    # Paint each dark panel with a black paper-rect shape
    for axis_n in ["yaxis2", "yaxis3", "yaxis4"]:
        domain = fig.layout[axis_n].domain
        if domain:
            fig.add_shape(
                type="rect",
                xref="paper", yref="paper",
                x0=0, x1=1,
                y0=domain[0], y1=domain[1],
                fillcolor="#0e1117",
                line_width=0,
                layer="below",
            )
        
        
    # Style subplot title annotations
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=12, color="#1e3a5f"), bgcolor="rgba(240,248,255,0.8)",
                   bordercolor="#93c5fd", borderwidth=1, borderpad=3)
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
                st_level=st_level, close=close,
                dist_pct=dist_pct, is_new_signal=is_new, streak=streak)




# ============================================================================
# VIX ANALYSIS
# ============================================================================

def fetch_vix_data(period: str = "1y") -> Optional[pd.DataFrame]:
    """Fetch India VIX data aligned to the same period as the index."""
    try:
        vix = yf.Ticker("^INDIAVIX")
        df  = vix.history(period=period)
        if df.empty:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df = df.sort_index()
        # Strip timezone for clean plotting
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return None


def create_vix_index_chart(index_df: pd.DataFrame, vix_df: pd.DataFrame,
                            index_name: str) -> go.Figure:
    """
    Dual-axis chart: Index price (left) + India VIX (right).
    Background shading shows VIX regimes (low/medium/high fear).
    Bottom panel: rolling 20-day correlation between VIX returns and Index returns.
    """
    # Normalise both indexes to tz-naive so intersection & arithmetic work
    def _strip_tz(df):
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        return df

    index_df = _strip_tz(index_df)
    vix_df   = _strip_tz(vix_df)

    # Align on common dates
    common = index_df.index.intersection(vix_df.index)
    idx    = index_df.loc[common].tail(252)   # last ~1 year of trading days
    vix    = vix_df.loc[common].tail(252)

    # Daily returns for correlation — must share the same (stripped) index
    idx_ret = idx["Close"].pct_change()
    vix_ret = vix["Close"].pct_change()
    # Align explicitly before rolling corr
    idx_ret, vix_ret = idx_ret.align(vix_ret, join="inner")
    rolling_corr = idx_ret.rolling(20).corr(vix_ret)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=(
            f"📊 {index_name} vs India VIX (Dual Axis)",
            "India VIX Level — Fear Gauge",
            "Rolling 20-Day Correlation (Index vs VIX)"
        ),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # ── Row 1: Index line (left axis) ────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=idx.index, y=idx["Close"],
        name=index_name,
        line=dict(color="#2563eb", width=2),
        hovertemplate=f"{index_name}: %{{y:,.0f}}<extra></extra>"
    ), row=1, col=1, secondary_y=False)

    # ── Row 1: VIX line (right axis) ────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=vix.index, y=vix["Close"],
        name="India VIX",
        line=dict(color="#dc2626", width=1.8, dash="dot"),
        hovertemplate="VIX: %{y:.2f}<extra></extra>"
    ), row=1, col=1, secondary_y=True)

    # ── Row 1: VIX fear-zone shading ────────────────────────────────────────
    vix_vals = vix["Close"]
    for lo, hi, clr, label in [
        (0,  15,   "rgba(22,163,74,0.06)",  "Low Fear (<15)"),
        (15, 20,   "rgba(234,179,8,0.06)",  "Moderate (15–20)"),
        (20, 25,   "rgba(249,115,22,0.07)", "Elevated (20–25)"),
        (25, 9999, "rgba(220,38,38,0.08)",  "High Fear (>25)"),
    ]:
        fig.add_hrect(y0=lo, y1=min(hi, float(vix_vals.max()) * 1.1),
                      fillcolor=clr, line_width=0,
                      row=1, col=1, secondary_y=True)

    # ── Row 2: VIX bar chart with color by level ────────────────────────────
    vix_colors = []
    for v in vix["Close"]:
        if v < 15:   vix_colors.append("#16a34a")
        elif v < 20: vix_colors.append("#ca8a04")
        elif v < 25: vix_colors.append("#f97316")
        else:        vix_colors.append("#dc2626")

    fig.add_trace(go.Bar(
        x=vix.index, y=vix["Close"],
        name="VIX Level",
        marker_color=vix_colors,
        opacity=0.8,
        hovertemplate="VIX: %{y:.2f}<extra></extra>"
    ), row=2, col=1)

    # VIX reference lines
    for lvl, clr, lbl in [(15, "#07f05d", "15"), (20, "#eda409", "20"), (25, "#f51414", "25")]:
        fig.add_hline(y=lvl, line_dash="dash", line_color=clr, line_width=1,
                      annotation_text=lbl, annotation_font=dict(size=9, color=clr),
                      annotation_position="right", row=2, col=1)

    # ── Row 3: Rolling correlation ──────────────────────────────────────────
    corr_colors = ["#dc2626" if c < 0 else "#2563eb" for c in rolling_corr.fillna(0)]
    fig.add_trace(go.Bar(
        x=rolling_corr.index, y=rolling_corr,
        name="20D Corr",
        marker_color=corr_colors,
        opacity=0.75,
        hovertemplate="Corr: %{y:.3f}<extra></extra>"
    ), row=3, col=1)
    fig.add_hline(y=0,    line_dash="solid", line_color="#888",   line_width=1, row=3, col=1)
    fig.add_hline(y=-0.5, line_dash="dash",  line_color="#fa2323",line_width=0.8,
                  annotation_text="-0.5", annotation_font=dict(size=9), row=3, col=1)
    fig.add_hline(y=0.5,  line_dash="dash",  line_color="#1348bd", line_width=0.8,
                  annotation_text="+0.5", annotation_font=dict(size=9), row=3, col=1)

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        height=900, hovermode="x unified", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1,
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#ddd", borderwidth=1),
        **CHART_THEME
    )
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    fig.update_yaxes(title_text=f"<b>{index_name}</b>", row=1, col=1, secondary_y=False,
                     title_font=dict(color="#2563eb", size=11))
    fig.update_yaxes(title_text="<b>VIX</b>", row=1, col=1, secondary_y=True,
                     title_font=dict(color="#dc2626", size=11),
                     showgrid=False)
    fig.update_yaxes(title_text="VIX Level", title_font=dict(size=10), row=2, col=1)
    fig.update_yaxes(title_text="Correlation", title_font=dict(size=10), row=3, col=1)
    fig = _style_subplot_titles(fig)
    return fig


def get_vix_summary(vix_df: pd.DataFrame) -> dict:
    """Return key VIX stats for the dashboard."""
    cur  = float(vix_df["Close"].iloc[-1])
    prev = float(vix_df["Close"].iloc[-2])
    chg  = cur - prev
    chg_pct = chg / prev * 100
    ma20 = float(vix_df["Close"].rolling(20).mean().iloc[-1])
    high52 = float(vix_df["Close"].rolling(min(252, len(vix_df))).max().iloc[-1])
    low52  = float(vix_df["Close"].rolling(min(252, len(vix_df))).min().iloc[-1])

    if cur < 15:   regime = "🟢 Low Fear"
    elif cur < 20: regime = "🟡 Moderate"
    elif cur < 25: regime = "🟠 Elevated"
    else:          regime = "🔴 High Fear / Panic"

    return dict(current=cur, change=chg, change_pct=chg_pct,
                ma20=ma20, high52=high52, low52=low52, regime=regime)

# ============================================================================
# FEAR-ADJUSTED INDEX  (Index CMP × VIX CMP)
# ============================================================================

def build_fear_index(index_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the Fear-Adjusted Index = Index_Close × VIX_Close.

    Interpretation
    ──────────────
    Rising  → market moving UP while fear stays high  (unsustainable / distribution risk)
    Falling → market falling AND/OR fear spiking      (panic, capitulation, potential bottom)
    Low & stable → calm bull market (healthy)
    High & stable → complacent rally (watch for reversal)
    """
    # Normalise tz
    idx = index_df.copy()
    vix = vix_df.copy()
    if idx.index.tz is not None:
        idx.index = idx.index.tz_localize(None)
    if vix.index.tz is not None:
        vix.index = vix.index.tz_localize(None)

    common = idx.index.intersection(vix.index)
    idx = idx.loc[common]
    vix = vix.loc[common]

    df = pd.DataFrame(index=common)
    df["Index_Close"] = idx["Close"].values
    df["VIX_Close"]   = vix["Close"].values
    df["FAI"]         = df["Index_Close"] * df["VIX_Close"]   # Fear-Adjusted Index

    # OHLC proxy for FAI (needed for Supertrend ATR)
    df["FAI_High"]  = (idx["High"].values  * vix["High"].values)
    df["FAI_Low"]   = (idx["Low"].values   * vix["Low"].values)
    df["FAI_Open"]  = (idx["Open"].values  * vix["Open"].values)
    df["FAI_Close"] = df["FAI"]
    return df


def _supertrend_on_series(high: pd.Series, low: pd.Series, close: pd.Series,
                           atr_period: int = 10, multiplier: float = 3.0):
    """Vanilla Supertrend applied to any OHLC series."""
    hl2   = (high + low) / 2.0
    # ATR via Wilder's method
    tr    = pd.concat([high - low,
                       (high - close.shift()).abs(),
                       (low  - close.shift()).abs()], axis=1).max(axis=1)
    atr   = tr.ewm(alpha=1/atr_period, adjust=False).mean()

    upper_b = hl2 + multiplier * atr
    lower_b = hl2 - multiplier * atr

    upper_f = upper_b.copy()
    lower_f = lower_b.copy()
    st      = pd.Series(np.nan, index=close.index)
    direction = pd.Series(1, index=close.index)
    cv = close.values

    for i in range(1, len(close)):
        upper_f.iloc[i] = (min(upper_b.iloc[i], upper_f.iloc[i-1])
                           if cv[i-1] <= upper_f.iloc[i-1] else upper_b.iloc[i])
        lower_f.iloc[i] = (max(lower_b.iloc[i], lower_f.iloc[i-1])
                           if cv[i-1] >= lower_f.iloc[i-1] else lower_b.iloc[i])

        if   cv[i] > upper_f.iloc[i-1]: direction.iloc[i] =  1
        elif cv[i] < lower_f.iloc[i-1]: direction.iloc[i] = -1
        else:                            direction.iloc[i] = direction.iloc[i-1]

        st.iloc[i] = lower_f.iloc[i] if direction.iloc[i] == 1 else upper_f.iloc[i]

    return st, direction, upper_f, lower_f, atr


def analyse_fai_regimes(df: pd.DataFrame) -> dict:
    """
    Derive Bullish / Bearish / Caution ranges from FAI + Supertrend direction.

    Zones
    ─────
    Bullish  : ST direction = +1  AND  FAI below its 20-day mean   → calm uptrend
    Caution  : ST direction = +1  AND  FAI above its 20-day mean   → rally with fear, watch
    Bearish  : ST direction = -1                                    → downtrend confirmed
    Extreme  : ST direction = -1  AND  FAI in top 10% of range     → panic, potential reversal
    """
    st, direction, uf, lf, atr = _supertrend_on_series(
        df["FAI_High"], df["FAI_Low"], df["FAI_Close"], atr_period=10, multiplier=3.0)

    df = df.copy()
    df["ST"]        = st
    df["ST_Dir"]    = direction
    df["FAI_MA20"]  = df["FAI"].rolling(20).mean()
    df["FAI_Std20"] = df["FAI"].rolling(20).std()

    fai_90 = df["FAI"].quantile(0.90)
    fai_10 = df["FAI"].quantile(0.10)

    def zone(row):
        if row["ST_Dir"] == 1 and row["FAI"] <= row["FAI_MA20"]:
            return "BULLISH"
        elif row["ST_Dir"] == 1 and row["FAI"] > row["FAI_MA20"]:
            return "CAUTION"
        elif row["ST_Dir"] == -1 and row["FAI"] >= fai_90:
            return "EXTREME FEAR"
        else:
            return "BEARISH"

    df["Zone"] = df.apply(zone, axis=1)

    latest       = df.iloc[-1]
    current_zone = latest["Zone"]
    current_fai  = float(latest["FAI"])
    st_level     = float(latest["ST"])
    fai_ma       = float(latest["FAI_MA20"])
    fai_std      = float(latest["FAI_Std20"])

    # Dynamic support / resistance bands from ST + std
    bullish_zone_hi = fai_ma
    bullish_zone_lo = fai_ma - 1.5 * fai_std
    bearish_zone_lo = fai_ma
    bearish_zone_hi = fai_ma + 2.0 * fai_std

    return dict(
        df=df, current_zone=current_zone,
        current_fai=current_fai, st_level=st_level,
        fai_ma=fai_ma, fai_std=fai_std,
        fai_90=fai_90, fai_10=fai_10,
        bullish_hi=bullish_zone_hi, bullish_lo=bullish_zone_lo,
        bearish_lo=bearish_zone_lo, bearish_hi=bearish_zone_hi,
    )


def create_fai_chart(fai_result: dict, index_name: str) -> go.Figure:
    """
    4-panel Fear-Adjusted Index chart:
      Row 1 : FAI line + Supertrend + zone shading + BB bands
      Row 2 : Raw Index Close
      Row 3 : Raw VIX Close (color-coded)
      Row 4 : FAI momentum (ROC-10)
    """
    df   = fai_result["df"].tail(252)
    st, direction = df["ST"], df["ST_Dir"]

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.45, 0.20, 0.18, 0.17],
        subplot_titles=(
            f"🧮 Fear-Adjusted Index ({index_name} × VIX) + Supertrend",
            f"📈 {index_name} Close",
            "😨 India VIX",
            "⚡ FAI Momentum (ROC-10)"
        )
    )

    # ── Zone background shading ───────────────────────────────────────────────
    zone_clr = {"BULLISH": "rgba(22,163,74,0.07)",
                 "CAUTION": "rgba(234,179,8,0.07)",
                 "BEARISH": "rgba(220,38,38,0.07)",
                 "EXTREME FEAR": "rgba(127,29,29,0.12)"}
    prev_zone, seg_start = df["Zone"].iloc[0], df.index[0]
    for dt, row in df.iterrows():
        z = row["Zone"]
        if z != prev_zone:
            fig.add_vrect(x0=seg_start, x1=dt,
                          fillcolor=zone_clr.get(prev_zone, "rgba(200,200,200,0.05)"),
                          layer="below", line_width=0, row=1, col=1)
            seg_start, prev_zone = dt, z
    fig.add_vrect(x0=seg_start, x1=df.index[-1],
                  fillcolor=zone_clr.get(prev_zone, "rgba(200,200,200,0.05)"),
                  layer="below", line_width=0, row=1, col=1)

    # ── Row 1: FAI + Supertrend ───────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["FAI"], name="FAI",
        line=dict(color="#2563eb", width=2),
        hovertemplate="FAI: %{y:,.0f}<extra></extra>"
    ), row=1, col=1)

    # Bollinger on FAI
    fai_bb_mid  = df["FAI_MA20"]
    fai_bb_std  = df["FAI_Std20"]
    fai_bb_up   = fai_bb_mid + 2 * fai_bb_std
    fai_bb_lo   = fai_bb_mid - 2 * fai_bb_std
    fig.add_trace(go.Scatter(x=df.index, y=fai_bb_up, name="BB Upper",
                              line=dict(color="rgba(100,100,200,0.35)", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=fai_bb_lo, name="BB Lower",
                              line=dict(color="rgba(100,100,200,0.35)", width=1, dash="dot"),
                              fill="tonexty", fillcolor="rgba(100,100,200,0.04)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=fai_bb_mid, name="FAI MA-20",
                              line=dict(color="#ca8a04", width=1.2, dash="dash")), row=1, col=1)

    # Supertrend lines
    bull_st = st.where(direction == 1)
    bear_st = st.where(direction == -1)
    fig.add_trace(go.Scatter(x=df.index, y=bull_st, name="ST Bull",
                              line=dict(color="#16a34a", width=2.5), mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bear_st, name="ST Bear",
                              line=dict(color="#dc2626", width=2.5), mode="lines"), row=1, col=1)

    # Buy/sell flip signals
    buys  = df[(direction == 1)  & (direction.shift(1) == -1)]
    sells = df[(direction == -1) & (direction.shift(1) ==  1)]
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys.index, y=buys["FAI"] * 0.992,
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=14, color="#16a34a",
                        line=dict(color="white", width=1)),
            text=["BUY"] * len(buys), textposition="bottom center",
            textfont=dict(color="#15803d", size=9), name="FAI Buy"
        ), row=1, col=1)
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells["FAI"] * 1.008,
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=14, color="#dc2626",
                        line=dict(color="white", width=1)),
            text=["SELL"] * len(sells), textposition="top center",
            textfont=dict(color="#b91c1c", size=9), name="FAI Sell"
        ), row=1, col=1)

    # Dynamic bullish / bearish range bands
    fig.add_hrect(y0=fai_result["bullish_lo"], y1=fai_result["bullish_hi"],
                  fillcolor="rgba(22,163,74,0.06)", line_width=0.5,
                  line_color="#16a34a",
                  annotation_text="Bullish Zone", annotation_position="right",
                  annotation_font=dict(color="#15803d", size=9), row=1, col=1)
    fig.add_hrect(y0=fai_result["bearish_lo"], y1=fai_result["bearish_hi"],
                  fillcolor="rgba(220,38,38,0.06)", line_width=0.5,
                  line_color="#dc2626",
                  annotation_text="Bearish Zone", annotation_position="right",
                  annotation_font=dict(color="#b91c1c", size=9), row=1, col=1)

    # ── Row 2: Index Close ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Index_Close"], name=index_name,
        line=dict(color="#2563eb", width=1.8),
        hovertemplate=f"{index_name}: %{{y:,.0f}}<extra></extra>"
    ), row=2, col=1)

    # ── Row 3: VIX bar ───────────────────────────────────────────────────────
    vix_c = ["#16a34a" if v < 15 else "#ca8a04" if v < 20
              else "#f97316" if v < 25 else "#dc2626" for v in df["VIX_Close"]]
    fig.add_trace(go.Bar(
        x=df.index, y=df["VIX_Close"], name="VIX",
        marker_color=vix_c, opacity=0.8,
        hovertemplate="VIX: %{y:.2f}<extra></extra>"
    ), row=3, col=1)
    for lvl, clr in [(15, "#16a34a"), (20, "#ca8a04"), (25, "#dc2626")]:
        fig.add_hline(y=lvl, line_dash="dash", line_color=clr, line_width=0.8,
                      annotation_text=str(lvl), annotation_font=dict(size=8),
                      annotation_position="right", row=3, col=1)

    # ── Row 4: ROC-10 momentum of FAI ────────────────────────────────────────
    roc = df["FAI"].pct_change(10) * 100
    roc_c = ["#16a34a" if v >= 0 else "#dc2626" for v in roc.fillna(0)]
    fig.add_trace(go.Bar(
        x=df.index, y=roc, name="FAI ROC-10",
        marker_color=roc_c, opacity=0.75,
        hovertemplate="ROC-10: %{y:.2f}%<extra></extra>"
    ), row=4, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="#888", line_width=1, row=4, col=1)

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        height=1000, hovermode="x unified", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1,
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#ddd", borderwidth=1),
        **CHART_THEME
    )
    fig.update_xaxes(**AXIS_STYLE, tickfont=dict(color="#111111", size=11))
    fig.update_yaxes(**AXIS_STYLE, tickfont=dict(color="#111111", size=11))
    fig.update_yaxes(title_text="FAI",        title_font=dict(size=10), row=1, col=1)
    fig.update_yaxes(title_text=index_name,   title_font=dict(size=10), row=2, col=1)
    fig.update_yaxes(title_text="VIX",        title_font=dict(size=10), row=3, col=1)
    fig.update_yaxes(title_text="ROC-10 (%)", title_font=dict(size=10), row=4, col=1)
    fig = _style_subplot_titles(fig)
    return fig, buys, sells

# ============================================================================
# INDEX ANALYZER CLASS
# ============================================================================

class IndexAnalyzer:
    def __init__(self, index_name: str, period: str = "1y"):
        self.index_name = index_name
        self.symbol     = INDEX_MAP.get(index_name, "^NSEI")
        self.period     = period
        self.data       = None
        self.ticker     = None
        self.pattern_detector = None

    def fetch_data(self) -> bool:
        try:
            self.ticker = yf.Ticker(self.symbol)
            self.data   = self.ticker.history(period=self.period)
            if self.data.empty:
                st.error(f"No data for {self.symbol}")
                return False
            self.calculate_indicators()
            self.pattern_detector = IndexPatternDetector(self.data)
            return True
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return False

    def calculate_indicators(self):
        df = self.data
        df["SMA_20"]  = SMAIndicator(df["Close"], 20).sma_indicator()
        df["SMA_21"]  = SMAIndicator(df["Close"], 21).sma_indicator()
        df["SMA_50"]  = SMAIndicator(df["Close"], 50).sma_indicator()
        df["SMA_200"] = SMAIndicator(df["Close"], 200).sma_indicator()
        df["EMA_9"]   = EMAIndicator(df["Close"], 9).ema_indicator()
        df["EMA_13"]  = EMAIndicator(df["Close"], 13).ema_indicator()
        df["EMA_20"]  = EMAIndicator(df["Close"], 20).ema_indicator()
        df["EMA_50"]  = EMAIndicator(df["Close"], 50).ema_indicator()
        df["EMA_200"] = EMAIndicator(df["Close"], 200).ema_indicator()

        macd = MACD(df["Close"])
        df["MACD"]        = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Hist"]   = macd.macd_diff()

        df["RSI"] = RSIIndicator(df["Close"]).rsi()

        stoch = StochasticOscillator(df["High"], df["Low"], df["Close"])
        df["Stoch_K"] = stoch.stoch()
        df["Stoch_D"] = stoch.stoch_signal()

        bb = BollingerBands(df["Close"])
        df["BB_High"]  = bb.bollinger_hband()
        df["BB_Mid"]   = bb.bollinger_mavg()
        df["BB_Low"]   = bb.bollinger_lband()
        df["BB_Width"] = (df["BB_High"] - df["BB_Low"]) / df["BB_Mid"]

        df["ATR"]           = AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
        df["ATR_Pct"]       = df["ATR"] / df["Close"] * 100
        df["Daily_Return"]  = df["Close"].pct_change() * 100
        df["Volatility_20"] = df["Daily_Return"].rolling(20).std()
        df["Volume_SMA"]    = df["Volume"].rolling(20).mean()

        df["VWAP"] = ((df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3)
                      .cumsum() / df["Volume"].cumsum())

        df["Pivot"] = (df["High"] + df["Low"] + df["Close"]) / 3
        df["R1"]    = 2 * df["Pivot"] - df["Low"]
        df["S1"]    = 2 * df["Pivot"] - df["High"]
        df["R2"]    = df["Pivot"] + (df["High"] - df["Low"])
        df["S2"]    = df["Pivot"] - (df["High"] - df["Low"])

        df["Cam_R3"] = df["Close"] + (df["High"] - df["Low"]) * 1.1666
        df["Cam_S3"] = df["Close"] - (df["High"] - df["Low"]) * 1.1666

        df["OBV"] = OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()

        df["52W_High"] = df["High"].rolling(min(252, len(df))).max()
        df["52W_Low"]  = df["Low"].rolling(min(252, len(df))).min()
        df["Pct_from_52W_High"] = (df["Close"] - df["52W_High"]) / df["52W_High"] * 100
        df["Pct_from_52W_Low"]  = (df["Close"] - df["52W_Low"])  / df["52W_Low"]  * 100

        self.data = df

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
            signals.append("✅ Price above VWAP"); score += 1
        else:
            signals.append("❌ Price below VWAP"); score -= 1

        if current["Volume"] > current["Volume_SMA"]:
            signals.append("✅ Above-average volume"); score += 1
        else:
            signals.append("⚠️ Below-average volume")

        bb_pct = (current["Close"] - current["BB_Low"]) / (current["BB_High"] - current["BB_Low"])
        if bb_pct < 0.2:
            signals.append(f"📉 Near lower BB ({bb_pct*100:.0f}%) — oversold"); score += 1
        elif bb_pct > 0.8:
            signals.append(f"📈 Near upper BB ({bb_pct*100:.0f}%) — overbought"); score -= 1

        if   score >= 5:  overall = "🟢 STRONG BUY"
        elif score >= 3:  overall = "🟢 BUY"
        elif score >= -1: overall = "🟡 HOLD / NEUTRAL"
        elif score >= -3: overall = "🔴 SELL"
        else:             overall = "🔴 STRONG SELL"

        return overall, signals, score

    def get_key_levels(self):
        df      = self.data
        current = df.iloc[-1]
        return {
            "current":  float(current["Close"]),
            "pivot":    float(current["Pivot"]),
            "r1": float(current["R1"]),  "s1": float(current["S1"]),
            "r2": float(current["R2"]),  "s2": float(current["S2"]),
            "cam_r3":   float(current["Cam_R3"]),
            "cam_s3":   float(current["Cam_S3"]),
            "bb_upper": float(current["BB_High"]),
            "bb_mid":   float(current["BB_Mid"]),
            "bb_lower": float(current["BB_Low"]),
            "vwap":     float(current["VWAP"]),
            "atr":      float(current["ATR"]),
            "atr_pct":  float(current["ATR_Pct"]),
            "52w_high": float(current["52W_High"]),
            "52w_low":  float(current["52W_Low"]),
            "pct_from_52w_high": float(current["Pct_from_52W_High"]),
            "pct_from_52w_low":  float(current["Pct_from_52W_Low"]),
        }

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

    def get_index_stats(self):
        df      = self.data
        cur     = df.iloc[-1]
        prev    = df.iloc[-2]
        chg     = cur["Close"] - prev["Close"]
        chg_pct = chg / prev["Close"] * 100

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
            "ret_1w":     ret_1w, "ret_1m": ret_1m,
            "ret_3m":     ret_3m, "ret_ytd": ret_ytd,
            "volatility_20d":    float(df["Volatility_20"].iloc[-1]),
            "atr_pct":           float(cur["ATR_Pct"]),
            "pct_from_52w_high": float(cur["Pct_from_52W_High"]),
            "pct_from_52w_low":  float(cur["Pct_from_52W_Low"]),
        }


# ============================================================================
# CHART HELPERS
# ============================================================================

def _style_subplot_titles(fig):
    """Style every subplot title annotation: dark text, light blue pill background."""
    for ann in fig.layout.annotations:
        ann.update(
            font=dict(size=12, color="#1e3a5f", family="Arial, sans-serif"),
            bgcolor="rgba(219,234,254,0.85)",
            bordercolor="#93c5fd",
            borderwidth=1,
            borderpad=4,
        )
    return fig


def create_candlestick_chart(analyzer: IndexAnalyzer, patterns=None):
    df = analyzer.data.tail(200)

    # subplot_titles positions each title directly above its own panel
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.50, 0.15, 0.15, 0.20],
        subplot_titles=(
            "📈 Price Action — Indicators & Levels",
            "📊 MACD",
            "📉 RSI & Stochastics",
            "📦 Volume"
        )
    )

    # ── Row 1: Candlestick ────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing_line_color="#16a34a", decreasing_line_color="#dc2626"
    ), row=1, col=1)

    colours_ma = {
        "SMA_21":  ("#f97316", 1.2), "SMA_50":  ("#2563eb", 1.5),
        "SMA_200": ("#dc2626", 2.0), "EMA_9":   ("#16a34a", 1.2),
        "EMA_20":  ("#0891b2", 1.2), "EMA_50":  ("#7c3aed", 1.5),
    }
    for col_name, (clr, wid) in colours_ma.items():
        if col_name in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col_name], name=col_name,
                                      line=dict(color=clr, width=wid)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["BB_High"], name="BB Upper",
                              line=dict(color="rgba(100,100,100,0.4)", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Low"], name="BB Lower",
                              line=dict(color="rgba(100,100,100,0.4)", width=1, dash="dot"),
                              fill="tonexty", fillcolor="rgba(100,100,100,0.05)"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP",
                              line=dict(color="#ca8a04", width=1.5, dash="dot")), row=1, col=1)

    kl = analyzer.get_key_levels()
    for name, val, clr in [
        ("R2", kl["r2"], "#dc2626"), ("R1", kl["r1"], "#f87171"),
        ("Pivot", kl["pivot"], "#374151"),
        ("S1", kl["s1"], "#4ade80"), ("S2", kl["s2"], "#16a34a"),
    ]:
        fig.add_hline(y=val, line_dash="dot", line_color=clr, line_width=1,
                      annotation_text=f"{name}: {val:.0f}",
                      annotation_font=dict(color=clr, size=10),
                      annotation_position="right", row=1, col=1)

    if patterns:
        fig = _draw_index_patterns(fig, patterns, df)

    # ── Row 2: MACD ───────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                              line=dict(color="#2563eb", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
                              line=dict(color="#dc2626", width=1.5)), row=2, col=1)
    hist_c = ["#16a34a" if v >= 0 else "#dc2626" for v in df["MACD_Hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram",
                          marker_color=hist_c, opacity=0.7), row=2, col=1)

    # ── Row 3: RSI + Stoch ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                              line=dict(color="#7c3aed", width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_K"], name="Stoch %K",
                              line=dict(color="#f97316", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_D"], name="Stoch %D",
                              line=dict(color="#fb923c", width=1.5, dash="dot")), row=3, col=1)
    for lvl, clr, dash in [(70, "#dc2626", "dash"), (30, "#16a34a", "dash"),
                            (80, "#7f1d1d", "dot"),  (20, "#14532d", "dot")]:
        fig.add_hline(y=lvl, line_dash=dash, line_color=clr, line_width=0.8,
                      annotation_text=str(lvl),
                      annotation_font=dict(size=9, color=clr),
                      annotation_position="left", row=3, col=1)

    # ── Row 4: Volume ─────────────────────────────────────────────────────────
    vc = ["#16a34a" if df["Close"].iloc[i] >= df["Open"].iloc[i]
          else "#dc2626" for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                          marker_color=vc, opacity=0.75), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Volume_SMA"], name="Vol MA-20",
                              line=dict(color="#f97316", width=2)), row=4, col=1)

    fig.update_layout(
        title=dict(
            text=f"<b>{analyzer.index_name}</b> — Technical Analysis Dashboard",
            font=dict(size=16, color="#1e3a5f")
        ),
        xaxis_rangeslider_visible=False, height=1250,
        showlegend=True, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.005,
                    xanchor="right", x=1,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#ddd", borderwidth=1),
        **CHART_THEME
    )
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    fig.update_yaxes(title_text="Price (₹)",   title_font=dict(size=10), row=1, col=1)
    fig.update_yaxes(title_text="MACD",        title_font=dict(size=10), row=2, col=1)
    fig.update_yaxes(title_text="RSI / Stoch", title_font=dict(size=10), row=3, col=1)
    fig.update_yaxes(title_text="Volume",      title_font=dict(size=10), row=4, col=1)

    fig = _style_subplot_titles(fig)
    return fig


def _draw_index_patterns(fig, patterns, df):
    for p in patterns:
        sig  = p.get("signal", "NEUTRAL")
        clr  = "#16a34a" if sig == "BULLISH" else "#dc2626" if sig == "BEARISH" else "#f97316"
        for key, label, lclr, lstyle in [
            ("entry_price", "📍 ENTRY", clr,       "dash"),
            ("stop_loss",   "🛑 STOP",  "#dc2626", "dot"),
            ("target_1",    "🎯 T1",    "#16a34a", "dot"),
            ("target_2",    "🎯 T2",    "#15803d", "dot"),
        ]:
            v = p.get(key)
            if v and isinstance(v, (int, float)):
                fig.add_hline(y=v, line_dash=lstyle, line_color=lclr, line_width=1.5,
                              annotation_text=f"{label}: {v:.0f}",
                              annotation_position="right", row=1, col=1)
        if "support_zone" in p:
            fig.add_hrect(y0=p["support_zone"][0], y1=p["support_zone"][1],
                          fillcolor="rgba(22,163,74,0.07)", line_width=0, row=1, col=1)
        if "resistance_zone" in p:
            fig.add_hrect(y0=p["resistance_zone"][0], y1=p["resistance_zone"][1],
                          fillcolor="rgba(220,38,38,0.07)", line_width=0, row=1, col=1)
    return fig


def create_volume_profile_chart(analyzer: IndexAnalyzer):
    vp  = analyzer.detect_volume_profile()
    fig = go.Figure()
    pl  = (vp["price_bins"][:-1] + vp["price_bins"][1:]) / 2
    fig.add_trace(go.Bar(y=pl, x=vp["volume_distribution"], orientation="h",
                          name="Vol @ Price",
                          marker=dict(color="#3b82f6", line=dict(color="#2563eb", width=0.3)),
                          opacity=0.75))
    fig.add_hline(y=vp["poc_price"], line_dash="solid", line_color="#ca8a04",
                  annotation_text=f"POC: {vp['poc_price']:.0f}",
                  annotation_font=dict(color="#92400e"), line_width=2)
    fig.add_hrect(y0=vp["value_area_low"], y1=vp["value_area_high"],
                  fillcolor="rgba(59,130,246,0.08)", line_width=0,
                  annotation_text="Value Area 70%", annotation_position="right")
    for hvn in vp["high_volume_nodes"][:3]:
        fig.add_hline(y=hvn, line_dash="dash", line_color="#16a34a",
                      annotation_text="HVN", line_width=1)
    for lvn in vp["low_volume_nodes"][:3]:
        fig.add_hline(y=lvn, line_dash="dash", line_color="#f97316",
                      annotation_text="LVN", line_width=1)
    fig.update_layout(
        title=dict(text="<b>Volume Profile</b> (last 200 bars)",
                   font=dict(size=14, color="#1e3a5f")),
        xaxis_title="Volume", yaxis_title="Index Level (₹)",
        height=600, showlegend=True, **CHART_THEME
    )
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


# ============================================================================
# MCMC / HMM CHART HELPERS
# ============================================================================


def create_posterior_charts(mr: Dict, post: Dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Posterior P(μ|data) — Daily Drift",
                                        "Posterior P(σ|data) — Daily Volatility"))
    fig.add_trace(go.Histogram(x=mr["mu_samples"], nbinsx=80,
                                histnorm="probability density",
                                marker_color="rgba(59,130,246,0.55)",
                                name="μ posterior"), row=1, col=1)
    for x, clr, label in [(post["mu_mean"], "#2563eb", "Mean"),
                           (post["mle_mu_daily"], "#ca8a04", "MLE"),
                           (post["mu_ci_95_lo"],  "rgba(150,150,150,0.7)", ""),
                           (post["mu_ci_95_hi"],  "rgba(150,150,150,0.7)", "95% CI")]:
        fig.add_vline(x=x, line_color=clr, line_width=1.5,
                      line_dash="dot" if label == "MLE" else ("dash" if not label else "solid"),
                      annotation_text=label, row=1, col=1)

    fig.add_trace(go.Histogram(x=mr["sigma_samples"], nbinsx=80,
                                histnorm="probability density",
                                marker_color="rgba(220,38,38,0.55)",
                                name="σ posterior"), row=1, col=2)
    for x, clr, label in [(post["sigma_mean"], "#dc2626", "Mean"),
                           (post["mle_sigma_daily"], "#ca8a04", "MLE"),
                           (post["sigma_ci_95_lo"], "rgba(150,150,150,0.7)", ""),
                           (post["sigma_ci_95_hi"], "rgba(150,150,150,0.7)", "95% CI")]:
        fig.add_vline(x=x, line_color=clr, line_width=1.5,
                      line_dash="dot" if label == "MLE" else ("dash" if not label else "solid"),
                      annotation_text=label, row=1, col=2)

    fig.update_layout(title=dict(text="<b>Posterior Parameter Distributions</b>",
                                 font=dict(size=13, color="#1e3a5f")),
                      height=380, **CHART_THEME)
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    fig = _style_subplot_titles(fig)
    return fig


def create_trace_plots(mr: Dict) -> go.Figure:
    mu_c, sig_c = mr["mu_chains"], mr["sigma_chains"]
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Trace: μ (daily drift)", "Trace: σ (daily vol)"),
                        vertical_spacing=0.14)
    pal = ["#2563eb", "#16a34a", "#f97316", "#7c3aed", "#ca8a04", "#dc2626"]
    for c in range(mu_c.shape[0]):
        clr = pal[c % len(pal)]
        x   = list(range(len(mu_c[c])))
        fig.add_trace(go.Scatter(x=x, y=mu_c[c], mode="lines",
                                  line=dict(color=clr, width=0.8),
                                  name=f"Chain {c+1}", legendgroup=f"c{c}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=sig_c[c], mode="lines",
                                  line=dict(color=clr, width=0.8),
                                  name=f"Chain {c+1} σ", legendgroup=f"c{c}",
                                  showlegend=False), row=2, col=1)
    fig.update_layout(title=dict(text="<b>MCMC Trace Plots — Convergence Check</b>",
                                 font=dict(size=13, color="#1e3a5f")),
                      height=450, hovermode="x unified", **CHART_THEME)
    fig.update_xaxes(**AXIS_STYLE, title_text="Iteration")
    fig.update_yaxes(**AXIS_STYLE)
    fig = _style_subplot_titles(fig)
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
        period     = st.selectbox("Analysis Period",
                                   ["1mo","3mo","6mo","1y","2y","5y"], index=3)
        show_patterns_on_chart = st.checkbox("Show Patterns on Chart", value=True)

        st.markdown("---")
        st.markdown("### 🤖 AI Adaptive Supertrend")
        ai_atr = st.slider("ATR Period", 5, 30, 10, 1)
        ai_k   = st.slider("K-Means Clusters", 2, 5, 3, 1)
        st.caption("Auto-selects ATR multiplier per volatility regime")

        
        st.markdown("---")
        st.markdown("### 📊 Index Trading Notes")
        st.markdown("""
- Pivot levels key for options writers
- VIX correlation matters
- Volume = Futures + Cash combined
        """)
        analyze_btn = st.button("🔍 Analyze Index", type="primary", use_container_width=True)

    if analyze_btn:
        with st.spinner(f"📡 Fetching {index_name} data…"):
            analyzer = IndexAnalyzer(index_name, period)
            ok = analyzer.fetch_data()

        if not ok:
            st.error(f"❌ Failed to load data for {index_name}.")
            return

        # Fetch India VIX
        with st.spinner("📡 Fetching India VIX…"):
            vix_df = fetch_vix_data(period=period)
            vix_ok = vix_df is not None and len(vix_df) >= 5

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

        # ── Overview ──────────────────────────────────────────────────────────
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
            if stats["ret_1w"]  is not None: st.metric("1W Return",  f"{stats['ret_1w']:+.2f}%")
        with r2:
            if stats["ret_1m"]  is not None: st.metric("1M Return",  f"{stats['ret_1m']:+.2f}%")
        with r3:
            if stats["ret_3m"]  is not None: st.metric("3M Return",  f"{stats['ret_3m']:+.2f}%")
        with r4:
            if stats["ret_ytd"] is not None: st.metric("YTD Return", f"{stats['ret_ytd']:+.2f}%")

        v1, v2, v3 = st.columns(3)
        with v1: st.metric("20D Vol (Ann)", f"{stats['volatility_20d']*np.sqrt(252):.1f}%")
        with v2: st.metric("Daily ATR %",   f"{stats['atr_pct']:.2f}%")
        with v3: st.metric("Daily Volume",  f"{stats['volume']/1e7:.2f} Cr"
                                             if stats["volume"] > 1e7 else f"{stats['volume']:.0f}")

        # ── India VIX quick metrics ───────────────────────────────────────────
        if vix_ok:
            vix_sum = get_vix_summary(vix_df)
            st.markdown("#### 😨 India VIX — Fear Gauge")
            x1, x2, x3, x4, x5 = st.columns(5)
            vix_delta = f"{vix_sum['change']:+.2f} ({vix_sum['change_pct']:+.2f}%)"
            with x1: st.metric("India VIX",    f"{vix_sum['current']:.2f}", vix_delta)
            with x2: st.metric("Regime",       vix_sum["regime"])
            with x3: st.metric("VIX MA-20",    f"{vix_sum['ma20']:.2f}",
                                "Above MA" if vix_sum["current"] > vix_sum["ma20"] else "Below MA")
            with x4: st.metric("52W High",     f"{vix_sum['high52']:.2f}")
            with x5: st.metric("52W Low",      f"{vix_sum['low52']:.2f}")
            # Interpretation hint
            if vix_sum["current"] > 20:
                st.warning(f"⚠️ VIX at **{vix_sum['current']:.1f}** — elevated fear. "
                           "Options premiums high; index often oversold near these levels.")
            elif vix_sum["current"] < 13:
                st.info(f"😴 VIX at **{vix_sum['current']:.1f}** — extreme complacency. "
                        "Consider hedges; low VIX often precedes sharp moves.")
            else:
                st.success(f"✅ VIX at **{vix_sum['current']:.1f}** — normal range.")

        # ── Key Levels ────────────────────────────────────────────────────────
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
            st.metric("Mid-Range", f"{(kl['52w_high']+kl['52w_low'])/2:.0f}")

        # ── AI Supertrend ─────────────────────────────────────────────────────
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
            with c5: st.metric("Distance",            f"{ai_dash['dist_pct']:.2f}%",
                                delta=f"{ai_dash['streak']} bars in trend")

            cb1, cb2, cb3 = st.columns(3)
            with cb1: st.info(f"📊 Buy Signals (last 200 bars): **{len(ai_buys)}**")
            with cb2: st.info(f"📊 Sell Signals (last 200 bars): **{len(ai_sells)}**")
            with cb3:
                if len(ai_buys) > 0 and (len(ai_sells) == 0 or
                                          ai_buys.index[-1] > ai_sells.index[-1]):
                    last = "BUY 🟢"
                elif len(ai_sells) > 0:
                    last = "SELL 🔴"
                else:
                    last = "None"
                st.info(f"📊 Last Signal: **{last}**")

        # ── Trading Signal ────────────────────────────────────────────────────
        st.markdown('<div class="sub-header">🎯 Trading Signal</div>', unsafe_allow_html=True)
        overall, signals, score = analyzer.get_trading_signal()
        s1, s2 = st.columns([1, 2])
        with s1:
            st.markdown(f"### {overall}")
            st.markdown(f"**Signal Score: {score}**")
        with s2:
            for sig in signals:
                st.markdown(sig)

        # ── Pattern Detection ─────────────────────────────────────────────────
        st.markdown('<div class="sub-header">📈 Index Pattern Detection</div>',
                    unsafe_allow_html=True)
        pt1, pt2, pt3 = st.tabs(["Trend Continuation", "Reversal Patterns",
                                  "Advanced / Index-Specific"])

        def render_patterns(plist):
            if not plist:
                st.info("No patterns detected in current timeframe.")
                return
            bullish = [p for p in plist if p.get("signal") == "BULLISH"]
            bearish = [p for p in plist if p.get("signal") == "BEARISH"]
            neutral = [p for p in plist if p.get("signal") not in ["BULLISH","BEARISH"]]
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
                            for col, key, lbl in [(c1,"entry_price","📍 Entry"),
                                                   (c2,"stop_loss","🛑 Stop"),
                                                   (c3,"target_1","🎯 T1"),
                                                   (c4,"target_2","🎯 T2")]:
                                v = p.get(key)
                                if v:
                                    with col:
                                        st.markdown(f"**{lbl}:** {v:.0f}"
                                                    if isinstance(v, float) else f"**{lbl}:** {v}")

        with pt1:
            st.success(f"✅ {len(trend_pat)} trend pattern(s) found")
            render_patterns(trend_pat)
        with pt2:
            st.success(f"✅ {len(reversal_pat)} reversal pattern(s) found")
            render_patterns(reversal_pat)
        with pt3:
            st.success(f"✅ {len(advanced_pat)} advanced pattern(s) found")
            render_patterns(advanced_pat)

        # ── Charts ────────────────────────────────────────────────────────────
        st.markdown('<div class="sub-header">📊 Charts & Forecasts</div>', unsafe_allow_html=True)
        ct1, ct2, ct3, ct4, ct5, ct6, ct7 = st.tabs([
            "Price Action & Indicators", "Volume Profile",
            "🤖 AI Adaptive Supertrend", "⛓️ MCMC Bayesian Forecast",
            "🎲 HMM Forecast", "😨 VIX Analysis", "🧮 Fear-Adjusted Index",
        ])

        with ct1:
            if show_patterns_on_chart and all_patterns:
                st.info(f"📌 {len(all_patterns)} patterns overlaid on chart")
            fig_c = create_candlestick_chart(
                analyzer, all_patterns if show_patterns_on_chart else None)
            st.plotly_chart(fig_c, use_container_width=True)

        with ct2:
            st.plotly_chart(create_volume_profile_chart(analyzer), use_container_width=True)
            vp = analyzer.detect_volume_profile()
            v1, v2, v3 = st.columns(3)
            with v1: st.metric("POC",             f"{vp['poc_price']:.0f}")
            with v2: st.metric("Value Area High", f"{vp['value_area_high']:.0f}")
            with v3: st.metric("Value Area Low",  f"{vp['value_area_low']:.0f}")

        with ct3:
            if ai_ok:
                st.plotly_chart(ai_fig, use_container_width=True)
                st.markdown("### 📋 Recent AI Supertrend Signals")
                rows = []
                for idx, row in ai_buys.iterrows():
                    rows.append({"Date": idx.strftime("%Y-%m-%d"), "Type": "🟢 BUY",
                                 "Level": f"{row['Close']:.0f}",
                                 "AI-ST": f"{row['AI_Supertrend']:.0f}",
                                 "Mult":  f"{row['AI_Multiplier']:.1f}×",
                                 "Regime": row["AI_ST_Regime"]})
                for idx, row in ai_sells.iterrows():
                    rows.append({"Date": idx.strftime("%Y-%m-%d"), "Type": "🔴 SELL",
                                 "Level": f"{row['Close']:.0f}",
                                 "AI-ST": f"{row['AI_Supertrend']:.0f}",
                                 "Mult":  f"{row['AI_Multiplier']:.1f}×",
                                 "Regime": row["AI_ST_Regime"]})
                if rows:
                    sdf = pd.DataFrame(rows).sort_values("Date", ascending=False).head(15)
                    st.dataframe(sdf, use_container_width=True, hide_index=True)

                st.markdown("### 🎯 Regime Distribution (last 200 bars)")
                rc    = df_ai["AI_ST_Regime"].value_counts()
                rcols = st.columns(min(len(rc), 3))
                emo   = {"Low Vol":"🟢","Medium Vol":"🟡","High Vol":"🔴"}
                for j, (rn, rv) in enumerate(rc.items()):
                    if j < 3:
                        with rcols[j]:
                            st.metric(f"{emo.get(rn,'⚫')} {rn}",
                                      f"{rv/len(df_ai)*100:.1f}%", f"{rv} bars")
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
                    mcmc_out = run_mcmc_analysis(
                        analyzer.data, forecast_days=mcmc_days,
                        n_samples=mcmc_samples, n_warmup=max(500, mcmc_samples//2),
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

                k1,k2,k3,k4,k5,k6 = st.columns(6)
                with k1: st.metric("Current",     f"{fs['current_price']:.0f}")
                with k2: st.metric("Target",      f"{fs['target_price']:.0f}",
                                   f"{fs['expected_return']:+.2f}%")
                with k3: st.metric("95% CI Low",  f"{fs['ci_95_low']:.0f}")
                with k4: st.metric("95% CI High", f"{fs['ci_95_high']:.0f}")
                with k5: st.metric("Ann Drift",   f"{fs['ann_drift_mean']:+.1f}%")
                with k6: st.metric("Ann Vol",     f"{fs['ann_volatility']:.1f}%")

                st.plotly_chart(create_posterior_charts(mr, post), use_container_width=True)

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
                st.caption("✅ Good mixing = 'fuzzy caterpillars'. Diverging → increase samples/warmup.")

        with ct5:
            st.markdown("### 🎲 Hidden Markov Model (HMM) Forecast")
            with st.spinner("Running HMM analysis…"):
                hmm_r = run_hmm_analysis(analyzer.data, forecast_days=30)
            fc    = hmm_r["forecast"]
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
                if cs=="BULL":   st.success(f"🟢 {cs} ({prob:.1%})")
                elif cs=="BEAR": st.error(f"🔴 {cs} ({prob:.1%})")
                else:            st.info(f"🟡 {cs} ({prob:.1%})")
                st.markdown(f"Avg duration: **{pers[cs]['avg_duration']:.0f} days**")
            with re2:
                st.markdown("**Dominant Future**")
                dom = fc["dominant_regime"]
                dc  = fc["regime_confidence"]
                if dom=="BULL":  st.success(f"🟢 {dom} ({dc:.1%})")
                elif dom=="BEAR":st.error(f"🔴 {dom} ({dc:.1%})")
                else:            st.info(f"🟡 {dom} ({dc:.1%})")
                st.markdown(f"Bull: {fc['bull_probability']:.1%} | "
                            f"Bear: {fc['bear_probability']:.1%} | "
                            f"Side: {fc['sideways_probability']:.1%}")
            with re3:
                st.markdown("**Transition Matrix**")
                tm = pd.DataFrame(fc["state_transition_matrix"],
                                  columns=["→Bull","→Bear","→Side"],
                                  index=["Bull→","Bear→","Side→"])
                st.dataframe(tm.style.format("{:.1%}"), use_container_width=True)

            if fc["direction"]=="BULLISH":
                st.success(f"📈 BULLISH — Target {fc['target_price']:.0f} "
                           f"({fc['expected_return']:.2f}% gain)")
            elif fc["direction"]=="BEARISH":
                st.error(f"📉 BEARISH — Target {fc['target_price']:.0f} "
                         f"({fc['expected_return']:.2f}% loss)")
            else:
                st.info(f"📊 NEUTRAL — Range {fc['worst_case']:.0f}–{fc['best_case']:.0f}")

        with ct6:
            st.markdown("### 😨 India VIX vs Index Analysis")
            if vix_ok:
                vix_sum = get_vix_summary(vix_df)

                # Key VIX stats row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current VIX",  f"{vix_sum['current']:.2f}",
                              f"{vix_sum['change']:+.2f} ({vix_sum['change_pct']:+.2f}%)")
                with col2:
                    st.metric("Fear Regime",  vix_sum["regime"])
                with col3:
                    st.metric("VIX MA-20",    f"{vix_sum['ma20']:.2f}",
                              "⬆ Above MA" if vix_sum["current"] > vix_sum["ma20"] else "⬇ Below MA")
                with col4:
                    vix_range_pct = (vix_sum["current"] - vix_sum["low52"]) / (vix_sum["high52"] - vix_sum["low52"]) * 100
                    st.metric("VIX Percentile (52W)", f"{vix_range_pct:.0f}th pct")

                # VIX regime guide
                st.markdown("""
| VIX Level | Regime | Market Implication |
|-----------|--------|--------------------|
| < 15 | 🟢 Low Fear | Complacency — watch for reversal risk |
| 15–20 | 🟡 Moderate | Normal market conditions |
| 20–25 | 🟠 Elevated | Increased uncertainty, hedge positions |
| > 25 | 🔴 High Fear | Panic / crash risk — potential buy zone |
""")
                # Chart
                vix_chart = create_vix_index_chart(analyzer.data, vix_df, index_name)
                st.plotly_chart(vix_chart, use_container_width=True)

                # Correlation insight
                # Strip tz from both before intersection
                idx_data = analyzer.data.copy()
                vix_data = vix_df.copy()
                if idx_data.index.tz is not None:
                    idx_data.index = idx_data.index.tz_localize(None)
                if vix_data.index.tz is not None:
                    vix_data.index = vix_data.index.tz_localize(None)
                common = idx_data.index.intersection(vix_data.index)
                if len(common) > 22:
                    idx_r = idx_data.loc[common, "Close"].pct_change().dropna()
                    vix_r = vix_data.loc[common, "Close"].pct_change().dropna()
                    idx_r, vix_r = idx_r.align(vix_r, join="inner")
                    corr_1m = idx_r.tail(22).corr(vix_r.tail(22))
                    corr_3m = idx_r.tail(66).corr(vix_r.tail(66))
                    corr_1y = idx_r.corr(vix_r)

                    st.markdown("#### 📐 Correlation: Index Returns vs VIX Returns")
                    cc1, cc2, cc3 = st.columns(3)
                    with cc1: st.metric("1-Month Correlation", f"{corr_1m:.3f}")
                    with cc2: st.metric("3-Month Correlation", f"{corr_3m:.3f}")
                    with cc3: st.metric("1-Year Correlation",  f"{corr_1y:.3f}")

                    if corr_1m < -0.4:
                        st.success("📉 Strong negative correlation — VIX rising = Index falling (normal fear relationship).")
                    elif corr_1m > 0.1:
                        st.warning("⚠️ Unusual positive correlation — VIX and Index moving together (possible trend confusion).")
                    else:
                        st.info("🔄 Weak correlation — VIX and Index decoupled recently.")
            else:
                st.warning("⚠️ India VIX data unavailable. yfinance may not have `^INDIAVIX` at this time.")
        
        with ct7:
            st.markdown("### 🧮 Fear-Adjusted Index — Index × VIX Supertrend")
            st.caption("FAI = Index Close × VIX Close. Supertrend applied to this composite "
                       "to detect fear-adjusted trend regimes that raw price alone misses.")
            if vix_ok:
                try:
                    fai_df  = build_fear_index(analyzer.data, vix_df)
                    fai_res = analyse_fai_regimes(fai_df)

                    # ── Current regime banner ─────────────────────────────────
                    zone = fai_res["current_zone"]
                    if zone == "BULLISH":
                        st.success(f"🟢 **{zone}** — FAI Supertrend bullish, fear low. Healthy uptrend.")
                    elif zone == "CAUTION":
                        st.warning(f"🟡 **{zone}** — FAI Supertrend bullish BUT fear elevated. "
                                   "Rally may be unsustainable; tighten stops.")
                    elif zone == "EXTREME FEAR":
                        st.error(f"🔴 **{zone}** — FAI at panic levels. "
                                 "Potential capitulation / reversal zone. Watch for ST flip.")
                    else:
                        st.error(f"🔴 **{zone}** — FAI Supertrend bearish. Downtrend confirmed.")

                    # ── Key metrics ───────────────────────────────────────────
                    f1, f2, f3, f4, f5 = st.columns(5)
                    with f1: st.metric("FAI (Current)",   f"{fai_res['current_fai']:,.0f}")
                    with f2: st.metric("FAI Supertrend",  f"{fai_res['st_level']:,.0f}")
                    with f3: st.metric("FAI MA-20",       f"{fai_res['fai_ma']:,.0f}")
                    with f4: st.metric("Bullish Zone",
                                       f"{fai_res['bullish_lo']:,.0f} – {fai_res['bullish_hi']:,.0f}")
                    with f5: st.metric("Bearish Zone",
                                       f"{fai_res['bearish_lo']:,.0f} – {fai_res['bearish_hi']:,.0f}")

                    # ── Zone distribution table ───────────────────────────────
                    st.markdown("#### 📊 Zone Distribution (last 252 days)")
                    zone_counts = fai_res["df"].tail(252)["Zone"].value_counts()
                    total = zone_counts.sum()
                    zc1, zc2, zc3, zc4 = st.columns(4)
                    zone_cols = {"BULLISH": (zc1, "🟢"), "CAUTION": (zc2, "🟡"),
                                 "BEARISH": (zc3, "🔴"), "EXTREME FEAR": (zc4, "🟣")}
                    for zn, (col, emo) in zone_cols.items():
                        cnt = int(zone_counts.get(zn, 0))
                        with col:
                            st.metric(f"{emo} {zn}", f"{cnt/total*100:.1f}%", f"{cnt} days")

                    # ── Reference guide ───────────────────────────────────────
                    st.markdown("""
| Zone | ST Dir | FAI vs MA-20 | Meaning | Action |
|------|--------|-------------|---------|--------|
| 🟢 Bullish | ↑ | Below | Calm uptrend, low fear | Hold longs |
| 🟡 Caution | ↑ | Above | Rally with fear — unstable | Trail stops |
| 🔴 Bearish | ↓ | Any | Downtrend confirmed | Reduce longs / hedge |
| 🟣 Extreme Fear | ↓ | Top 10% | Panic / capitulation | Watch for ST flip → buy |
""")

                    # ── Chart ─────────────────────────────────────────────────
                    fai_fig, fai_buys, fai_sells = create_fai_chart(fai_res, index_name)
                    st.plotly_chart(fai_fig, use_container_width=True)

                    # ── Signal table ──────────────────────────────────────────
                    st.markdown("#### 📋 FAI Supertrend Signal History")
                    sig_rows = []
                    for dt, row in fai_buys.iterrows():
                        sig_rows.append({"Date": dt.strftime("%Y-%m-%d"), "Signal": "🟢 BUY",
                                         "FAI": f"{row['FAI']:,.0f}",
                                         "Index": f"{row['Index_Close']:,.0f}",
                                         "VIX": f"{row['VIX_Close']:.2f}"})
                    for dt, row in fai_sells.iterrows():
                        sig_rows.append({"Date": dt.strftime("%Y-%m-%d"), "Signal": "🔴 SELL",
                                         "FAI": f"{row['FAI']:,.0f}",
                                         "Index": f"{row['Index_Close']:,.0f}",
                                         "VIX": f"{row['VIX_Close']:.2f}"})
                    if sig_rows:
                        sig_df = pd.DataFrame(sig_rows).sort_values("Date", ascending=False).head(20)
                        st.dataframe(sig_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No Supertrend flips in the current window.")

                except Exception as e:
                    st.error(f"FAI chart error: {e}")
            else:
                st.warning("⚠️ India VIX data unavailable — cannot build Fear-Adjusted Index.")
            
        # ── Indicators Summary ────────────────────────────────────────────────
        st.markdown('<div class="sub-header">📋 Technical Indicators Summary</div>',
                    unsafe_allow_html=True)
        cur = analyzer.data.iloc[-1]
        ind_df = pd.DataFrame([
            {"Indicator": "RSI (14)",      "Value": f"{cur['RSI']:.2f}",
             "Signal": "Overbought" if cur["RSI"]>70 else "Oversold" if cur["RSI"]<30 else "Neutral"},
            {"Indicator": "MACD",          "Value": f"{cur['MACD']:.2f}",
             "Signal": "Bullish" if cur["MACD"]>cur["MACD_Signal"] else "Bearish"},
            {"Indicator": "Stoch %K",      "Value": f"{cur['Stoch_K']:.2f}",
             "Signal": "Overbought" if cur["Stoch_K"]>80 else "Oversold" if cur["Stoch_K"]<20 else "Neutral"},
            {"Indicator": "BB Width %",    "Value": f"{cur['BB_Width']*100:.2f}%",
             "Signal": "Squeeze" if cur["BB_Width"]<0.04 else "Expansion" if cur["BB_Width"]>0.10 else "Normal"},
            {"Indicator": "ATR %",         "Value": f"{cur['ATR_Pct']:.2f}%",
             "Signal": "High Vol" if cur["ATR_Pct"]>2 else "Low Vol" if cur["ATR_Pct"]<0.8 else "Normal"},
            {"Indicator": "OBV Trend",     "Value": f"{cur['OBV']:.0f}",
             "Signal": "Accumulation" if cur["OBV"]>analyzer.data["OBV"].mean() else "Distribution"},
            {"Indicator": "Above VWAP",    "Value": f"{cur['Close']:.0f} vs {cur['VWAP']:.0f}",
             "Signal": "Bullish" if cur["Close"]>cur["VWAP"] else "Bearish"},
            {"Indicator": "20D Vol (Ann)", "Value": f"{cur['Volatility_20']*np.sqrt(252):.1f}%",
             "Signal": "Elevated" if cur["Volatility_20"]*np.sqrt(252)>20 else "Normal"},
        ])
        st.dataframe(ind_df, use_container_width=True, hide_index=True)
        st.success(f"✅ Analysis complete for **{index_name}**")
        st.markdown("---")
        st.markdown("### 📐 Pattern Categories")
        st.markdown("""
                        
                    | Index Trend Patterns             | Reversal Patterns          | Index-Specific           |
                    |----------------------------------|----------------------------|--------------------------|
                    | ✅Ascending / Descending Triangle|✅ Head & Shoulders / Inverse H&S|✅ Range Breakout, Cup & Handle|
                    | ✅ Symmetrical Triangle|✅ Double Top / Double Bottom| ✅ Flat Base, Mean Reversion|
                    | ✅ Bull Flag / Bear Flag|✅ Triple Top / Triple Bottom|✅ Elliott Wave, Wyckoff Acc/Dist|
                    | ✅ Rising / Falling Wedge|        |        |
                    | ✅ Pennant (Bull/Bear)|        |        |
                        
                               
        """)


if __name__ == "__main__":
    main()
