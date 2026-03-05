"""
Index Pattern Detector — NIFTY / BANKNIFTY
===========================================
Detects 20 chart patterns adapted for broad-market indices
(no earnings / PE / dividend context; purely price-volume action).

Categories
----------
  Trend Continuation  (7) : Ascending/Descending/Symmetrical Triangle,
                             Bull/Bear Flag, Rising/Falling Wedge, Pennant
  Reversal            (6) : Head & Shoulders, Inverse H&S,
                             Double Top/Bottom, Triple Top/Bottom
  Advanced / Index    (7) : Range Breakout, Cup & Handle, Flat Base/Shelf,
                             Mean Reversion, Elliott Wave,
                             Wyckoff Accumulation, Wyckoff Distribution
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class IndexPatternDetector:
    """Pattern detection engine tailored to index behaviour."""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _find_peaks(self, prices, window=5):
        peaks = []
        for i in range(window, len(prices) - window):
            if prices[i] >= max(prices[i-window:i]) and prices[i] >= max(prices[i+1:i+window+1]):
                peaks.append((i, prices[i]))
        return peaks

    def _find_troughs(self, prices, window=5):
        troughs = []
        for i in range(window, len(prices) - window):
            if prices[i] <= min(prices[i-window:i]) and prices[i] <= min(prices[i+1:i+window+1]):
                troughs.append((i, prices[i]))
        return troughs

    # ──────────────────────────────────────────────────────────────────────────
    # TREND CONTINUATION PATTERNS
    # ──────────────────────────────────────────────────────────────────────────

    def detect_ascending_triangle(self, lookback=40) -> Dict:
        df = self.data.tail(lookback).copy()
        if len(df) < 30:
            return {"detected": False}
        score = 0
        highs = df["High"].values
        lows  = df["Low"].values

        res_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        res_var   = np.std(highs) / np.mean(highs)
        if res_var < 0.02 and abs(res_slope) < 0.5:
            score += 0.4

        sup_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        if sup_slope > 0:
            score += 0.3

        if np.polyfit(range(len(df)), df["Volume"].values, 1)[0] < 0:
            score += 0.2

        if df["Close"].iloc[-1] > np.mean(highs[-5:]) * 0.98:
            score += 0.1

        if score > 0.7:
            resistance = float(np.mean(highs[-5:]))
            support    = float(df["Low"].min())
            height     = resistance - support
            cp         = float(df["Close"].iloc[-1])
            return {
                "detected": True, "pattern": "Ascending Triangle", "signal": "BULLISH",
                "confidence": "MEDIUM", "score": score,
                "description": "Flat resistance + rising support. Buyers accumulating before breakout.",
                "entry_price": round(resistance * 1.005),
                "stop_loss":   round(support * 0.995),
                "target_1":    round(resistance + height),
                "target_2":    round(resistance + height * 1.5),
                "action": "BUY on close above resistance with rising volume",
                "support_zone":    (support * 0.998, support * 1.002),
                "resistance_zone": (resistance * 0.998, resistance * 1.002),
            }
        return {"detected": False}

    def detect_descending_triangle(self, lookback=40) -> Dict:
        df = self.data.tail(lookback).copy()
        if len(df) < 30:
            return {"detected": False}
        score = 0
        highs = df["High"].values
        lows  = df["Low"].values

        sup_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        sup_var   = np.std(lows) / np.mean(lows)
        if sup_var < 0.02 and abs(sup_slope) < 0.5:
            score += 0.4

        res_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        if res_slope < 0:
            score += 0.3

        if np.polyfit(range(len(df)), df["Volume"].values, 1)[0] < 0:
            score += 0.2

        if df["Close"].iloc[-1] < np.mean(lows[-5:]) * 1.02:
            score += 0.1

        if score > 0.7:
            support    = float(np.mean(lows[-5:]))
            resistance = float(df["High"].max())
            height     = resistance - support
            return {
                "detected": True, "pattern": "Descending Triangle", "signal": "BEARISH",
                "confidence": "MEDIUM", "score": score,
                "description": "Flat support + lower highs. Distribution before breakdown.",
                "entry_price": round(support * 0.995),
                "stop_loss":   round(resistance * 1.005),
                "target_1":    round(support - height),
                "target_2":    round(support - height * 1.5),
                "action": "SHORT on close below support with rising volume",
                "support_zone":    (support * 0.998, support * 1.002),
                "resistance_zone": (resistance * 0.998, resistance * 1.002),
            }
        return {"detected": False}

    def detect_symmetrical_triangle(self, lookback=40) -> Dict:
        df = self.data.tail(lookback).copy()
        if len(df) < 30:
            return {"detected": False}
        score = 0
        highs = df["High"].values
        lows  = df["Low"].values

        res_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        sup_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        if res_slope < -0.5 and sup_slope > 0.5:
            score += 0.4

        early_range = df.head(10)["High"].max() - df.head(10)["Low"].min()
        late_range  = df.tail(10)["High"].max() - df.tail(10)["Low"].min()
        if late_range < early_range * 0.6:
            score += 0.3

        if np.polyfit(range(len(df)), df["Volume"].values, 1)[0] < 0:
            score += 0.3

        if score > 0.7:
            cp         = float(df["Close"].iloc[-1])
            resistance = float(df["High"].max())
            support    = float(df["Low"].min())
            height     = resistance - support
            return {
                "detected": True, "pattern": "Symmetrical Triangle", "signal": "NEUTRAL",
                "confidence": "MEDIUM", "score": score,
                "description": "Coiling price action. Breakout direction determines trade.",
                "entry_price": None,
                "stop_loss":   None,
                "target_1":    round(resistance + height),
                "target_2":    round(support - height),
                "action": "WAIT — BUY above resistance OR SHORT below support with volume",
                "support_zone":    (support * 0.998, support * 1.002),
                "resistance_zone": (resistance * 0.998, resistance * 1.002),
            }
        return {"detected": False}

    def detect_bull_flag(self, lookback=25) -> Dict:
        df = self.data.tail(lookback).copy()
        if len(df) < 20:
            return {"detected": False}
        score = 0
        pole_len  = min(8, len(df) // 2)
        pole_data = df.head(pole_len)
        pole_gain = (pole_data["Close"].iloc[-1] - pole_data["Close"].iloc[0]) / pole_data["Close"].iloc[0]
        if pole_gain < 0.05:
            return {"detected": False}
        score += 0.3

        flag_data  = df.tail(len(df) - pole_len)
        flag_slope = np.polyfit(range(len(flag_data)), flag_data["Close"].values, 1)[0]
        if flag_slope <= 0.5:
            score += 0.3
        flag_range = (flag_data["High"].max() - flag_data["Low"].min()) / flag_data["Close"].mean()
        if flag_range < 0.08:
            score += 0.2
        if flag_data["Volume"].mean() < pole_data["Volume"].mean() * 0.75:
            score += 0.2

        if score > 0.7:
            fh = float(flag_data["High"].max())
            fl = float(flag_data["Low"].min())
            pole_h = pole_data["Close"].iloc[-1] - pole_data["Close"].iloc[0]
            return {
                "detected": True, "pattern": "Bull Flag", "signal": "BULLISH",
                "confidence": "HIGH" if score > 0.8 else "MEDIUM", "score": score,
                "description": "Brief pause after strong up-move. Continuation higher expected.",
                "entry_price": round(fh * 1.003),
                "stop_loss":   round(fl * 0.997),
                "target_1":    round(fh + pole_h),
                "target_2":    round(fh + pole_h * 1.5),
                "action": "BUY on breakout above flag with volume surge",
            }
        return {"detected": False}

    def detect_bear_flag(self, lookback=25) -> Dict:
        df = self.data.tail(lookback).copy()
        if len(df) < 20:
            return {"detected": False}
        score = 0
        pole_len  = min(8, len(df) // 2)
        pole_data = df.head(pole_len)
        pole_loss = (pole_data["Close"].iloc[-1] - pole_data["Close"].iloc[0]) / pole_data["Close"].iloc[0]
        if pole_loss > -0.03:
            return {"detected": False}
        score += 0.3

        flag_data  = df.tail(len(df) - pole_len)
        flag_slope = np.polyfit(range(len(flag_data)), flag_data["Close"].values, 1)[0]
        if flag_slope >= -0.5:
            score += 0.3
        flag_range = (flag_data["High"].max() - flag_data["Low"].min()) / flag_data["Close"].mean()
        if flag_range < 0.08:
            score += 0.2
        if flag_data["Volume"].mean() < pole_data["Volume"].mean() * 0.75:
            score += 0.2

        if score > 0.7:
            fh = float(flag_data["High"].max())
            fl = float(flag_data["Low"].min())
            pole_h = abs(pole_data["Close"].iloc[0] - pole_data["Close"].iloc[-1])
            return {
                "detected": True, "pattern": "Bear Flag", "signal": "BEARISH",
                "confidence": "HIGH" if score > 0.8 else "MEDIUM", "score": score,
                "description": "Counter-trend bounce after sharp decline. Downside continuation expected.",
                "entry_price": round(fl * 0.997),
                "stop_loss":   round(fh * 1.003),
                "target_1":    round(fl - pole_h),
                "target_2":    round(fl - pole_h * 1.5),
                "action": "SHORT on breakdown below flag with volume surge",
            }
        return {"detected": False}

    def detect_rising_wedge(self, lookback=35) -> Dict:
        df = self.data.tail(lookback).copy()
        if len(df) < 25:
            return {"detected": False}
        score = 0
        highs = df["High"].values
        lows  = df["Low"].values
        ht = np.polyfit(range(len(highs)), highs, 1)[0]
        lt = np.polyfit(range(len(lows)),  lows,  1)[0]

        if ht > 0 and lt > 0 and lt > ht * 0.7:
            score += 0.4
        early_range = df.head(10)["High"].max() - df.head(10)["Low"].min()
        late_range  = df.tail(10)["High"].max() - df.tail(10)["Low"].min()
        if late_range < early_range * 0.65:
            score += 0.3
        if np.polyfit(range(len(df)), df["Volume"].values, 1)[0] < 0:
            score += 0.3

        if score > 0.7:
            upper = float(df["High"].max())
            lower = float(df["Low"].min())
            height = upper - lower
            return {
                "detected": True, "pattern": "Rising Wedge", "signal": "BEARISH",
                "confidence": "MEDIUM", "score": score,
                "description": "Rising but converging price action. Bearish divergence — buying exhaustion.",
                "entry_price": round(lower * 0.997),
                "stop_loss":   round(upper * 1.003),
                "target_1":    round(lower - height),
                "target_2":    round(lower - height * 1.5),
                "action": "SHORT on break below lower trendline with volume",
            }
        return {"detected": False}

    def detect_falling_wedge(self, lookback=35) -> Dict:
        df = self.data.tail(lookback).copy()
        if len(df) < 25:
            return {"detected": False}
        score = 0
        highs = df["High"].values
        lows  = df["Low"].values
        ht = np.polyfit(range(len(highs)), highs, 1)[0]
        lt = np.polyfit(range(len(lows)),  lows,  1)[0]

        if ht < 0 and lt < 0 and lt < ht * 0.7:
            score += 0.4
        early_range = df.head(10)["High"].max() - df.head(10)["Low"].min()
        late_range  = df.tail(10)["High"].max() - df.tail(10)["Low"].min()
        if late_range < early_range * 0.65:
            score += 0.3
        if np.polyfit(range(len(df)), df["Volume"].values, 1)[0] < 0:
            score += 0.3

        if score > 0.7:
            upper = float(df["High"].max())
            lower = float(df["Low"].min())
            height = upper - lower
            return {
                "detected": True, "pattern": "Falling Wedge", "signal": "BULLISH",
                "confidence": "MEDIUM", "score": score,
                "description": "Declining but converging price action. Selling exhaustion — bullish reversal ahead.",
                "entry_price": round(upper * 1.003),
                "stop_loss":   round(lower * 0.997),
                "target_1":    round(upper + height),
                "target_2":    round(upper + height * 1.5),
                "action": "BUY on break above upper trendline with volume",
            }
        return {"detected": False}

    def detect_pennant(self, lookback=25) -> Dict:
        df = self.data.tail(lookback).copy()
        if len(df) < 18:
            return {"detected": False}
        pole_len  = min(7, len(df) // 3)
        pole_data = df.head(pole_len)
        move = (pole_data["Close"].iloc[-1] - pole_data["Close"].iloc[0]) / pole_data["Close"].iloc[0]
        is_bull = move > 0.04
        is_bear = move < -0.04
        if not (is_bull or is_bear):
            return {"detected": False}
        score = 0.3
        pdata = df.tail(len(df) - pole_len)
        if len(pdata) < 6:
            return {"detected": False}
        er = pdata.head(5)["High"].max() - pdata.head(5)["Low"].min()
        lr = pdata.tail(5)["High"].max() - pdata.tail(5)["Low"].min()
        if lr < er * 0.5:
            score += 0.4
        if pdata["Volume"].mean() < pole_data["Volume"].mean() * 0.7:
            score += 0.3

        if score > 0.7:
            ph = float(pdata["High"].max())
            pl = float(pdata["Low"].min())
            pr = abs(pole_data["Close"].iloc[-1] - pole_data["Close"].iloc[0])
            if is_bull:
                return {
                    "detected": True, "pattern": "Bullish Pennant", "signal": "BULLISH",
                    "confidence": "HIGH" if score > 0.8 else "MEDIUM", "score": score,
                    "description": "Small converging triangle after strong up-move. Quick continuation expected.",
                    "entry_price": round(ph * 1.003),
                    "stop_loss":   round(pl * 0.997),
                    "target_1":    round(ph + pr),
                    "target_2":    round(ph + pr * 1.5),
                    "action": "BUY on upside breakout with volume",
                }
            else:
                return {
                    "detected": True, "pattern": "Bearish Pennant", "signal": "BEARISH",
                    "confidence": "HIGH" if score > 0.8 else "MEDIUM", "score": score,
                    "description": "Small converging triangle after sharp decline. Continuation lower expected.",
                    "entry_price": round(pl * 0.997),
                    "stop_loss":   round(ph * 1.003),
                    "target_1":    round(pl - pr),
                    "target_2":    round(pl - pr * 1.5),
                    "action": "SHORT on downside breakdown with volume",
                }
        return {"detected": False}

    # ──────────────────────────────────────────────────────────────────────────
    # REVERSAL PATTERNS
    # ──────────────────────────────────────────────────────────────────────────

    def detect_head_and_shoulders(self, lookback=60) -> Dict:
        df = self.data.tail(lookback).copy()
        if len(df) < 45:
            return {"detected": False}
        prices = df["Close"].values
        peaks  = self._find_peaks(prices, window=5)
        troughs= self._find_troughs(prices, window=5)
        score  = 0

        if len(peaks) >= 3 and len(troughs) >= 2:
            ls, hd, rs = peaks[-3], peaks[-2], peaks[-1]
            if hd[1] > ls[1] and hd[1] > rs[1]:
                score += 0.3
                if abs(ls[1] - rs[1]) / ls[1] < 0.05:
                    score += 0.2
                if len(troughs) >= 2:
                    neckline = (troughs[-2][1] + troughs[-1][1]) / 2
                    if prices[-1] < neckline * 1.01:
                        score += 0.3
                    if df["Volume"].iloc[-10:].mean() < df["Volume"].iloc[-30:-10].mean():
                        score += 0.2

        if score > 0.7:
            neckline = float(df["Low"].tail(30).mean())
            head_val = float(df["High"].tail(40).max())
            height   = head_val - neckline
            return {
                "detected": True, "pattern": "Head & Shoulders", "signal": "BEARISH",
                "confidence": "HIGH" if score > 0.85 else "MEDIUM", "score": score,
                "description": "Bearish reversal — three peaks with middle highest. Neckline is key.",
                "entry_price": round(neckline * 0.997),
                "stop_loss":   round(head_val * 1.003),
                "target_1":    round(neckline - height),
                "target_2":    round(neckline - height * 1.5),
                "action": "SHORT on neckline breakdown with elevated volume",
                "support_zone":    (neckline * 0.995, neckline * 1.005),
            }
        return {"detected": False}

    def detect_inverse_head_and_shoulders(self, lookback=60) -> Dict:
        df = self.data.tail(lookback).copy()
        if len(df) < 45:
            return {"detected": False}
        prices  = df["Close"].values
        troughs = self._find_troughs(prices, window=5)
        peaks   = self._find_peaks(prices, window=5)
        score   = 0

        if len(troughs) >= 3 and len(peaks) >= 2:
            ls, hd, rs = troughs[-3], troughs[-2], troughs[-1]
            if hd[1] < ls[1] and hd[1] < rs[1]:
                score += 0.3
                if abs(ls[1] - rs[1]) / ls[1] < 0.05:
                    score += 0.2
                if len(peaks) >= 2:
                    neckline = (peaks[-2][1] + peaks[-1][1]) / 2
                    if prices[-1] > neckline * 0.98:
                        score += 0.3
                    if df["Volume"].iloc[-10:].mean() > df["Volume"].iloc[-30:-10].mean():
                        score += 0.2

        if score > 0.7:
            neckline = float(df["High"].tail(30).mean())
            head_val = float(df["Low"].tail(40).min())
            height   = neckline - head_val
            return {
                "detected": True, "pattern": "Inverse Head & Shoulders", "signal": "BULLISH",
                "confidence": "HIGH" if score > 0.85 else "MEDIUM", "score": score,
                "description": "Bullish reversal — three troughs with middle deepest. Classic bottom pattern.",
                "entry_price": round(neckline * 1.003),
                "stop_loss":   round(head_val * 0.997),
                "target_1":    round(neckline + height),
                "target_2":    round(neckline + height * 1.5),
                "action": "BUY on neckline breakout with rising volume",
                "resistance_zone": (neckline * 0.997, neckline * 1.003),
            }
        return {"detected": False}

    def detect_double_top(self, lookback=50) -> Dict:
        df = self.data.tail(lookback).copy()
        if len(df) < 35:
            return {"detected": False}
        prices = df["Close"].values
        peaks  = self._find_peaks(prices, window=4)
        score  = 0

        if len(peaks) >= 2:
            p1, p2 = peaks[-2], peaks[-1]
            if abs(p1[1] - p2[1]) / p1[1] < 0.03 and p2[0] - p1[0] > 8:
                score += 0.35
                trough_sec = prices[p1[0]:p2[0]]
                if len(trough_sec) > 0:
                    neckline = float(np.min(trough_sec))
                    score += 0.25
                    if prices[-1] < neckline * 1.01:
                        score += 0.25
                    if df["Volume"].iloc[p2[0]] < df["Volume"].iloc[p1[0]]:
                        score += 0.15

        if score > 0.7:
            top      = float(df["High"].max())
            neckline = float(df["Low"].tail(30).min())
            height   = top - neckline
            return {
                "detected": True, "pattern": "Double Top", "signal": "BEARISH",
                "confidence": "HIGH" if score > 0.85 else "MEDIUM", "score": score,
                "description": "M-shaped resistance. Two failed attempts at the same level.",
                "entry_price": round(neckline * 0.997),
                "stop_loss":   round(top * 1.003),
                "target_1":    round(neckline - height),
                "target_2":    round(neckline - height * 1.5),
                "action": "SHORT on neckline breakdown with volume",
            }
        return {"detected": False}

    def detect_double_bottom(self, lookback=50) -> Dict:
        df = self.data.tail(lookback).copy()
        if len(df) < 35:
            return {"detected": False}
        prices  = df["Close"].values
        troughs = self._find_troughs(prices, window=4)
        score   = 0

        if len(troughs) >= 2:
            t1, t2 = troughs[-2], troughs[-1]
            if abs(t1[1] - t2[1]) / t1[1] < 0.03 and t2[0] - t1[0] > 8:
                score += 0.35
                peak_sec = prices[t1[0]:t2[0]]
                if len(peak_sec) > 0:
                    neckline = float(np.max(peak_sec))
                    score += 0.25
                    if prices[-1] > neckline * 0.98:
                        score += 0.25
                    if df["Volume"].iloc[t2[0]] > df["Volume"].iloc[t1[0]] * 0.8:
                        score += 0.15

        if score > 0.7:
            bottom   = float(df["Low"].min())
            neckline = float(df["High"].tail(30).max())
            height   = neckline - bottom
            return {
                "detected": True, "pattern": "Double Bottom", "signal": "BULLISH",
                "confidence": "HIGH" if score > 0.85 else "MEDIUM", "score": score,
                "description": "W-shaped support. Strong demand zone tested twice.",
                "entry_price": round(neckline * 1.003),
                "stop_loss":   round(bottom * 0.997),
                "target_1":    round(neckline + height),
                "target_2":    round(neckline + height * 1.5),
                "action": "BUY on neckline breakout with volume",
            }
        return {"detected": False}

    def detect_triple_top(self, lookback=70) -> Dict:
        df = self.data.tail(lookback).copy()
        if len(df) < 50:
            return {"detected": False}
        prices = df["Close"].values
        peaks  = self._find_peaks(prices, window=4)
        score  = 0

        if len(peaks) >= 3:
            p1, p2, p3 = peaks[-3], peaks[-2], peaks[-1]
            avg = (p1[1] + p2[1] + p3[1]) / 3
            if all(abs(p[1] - avg) / avg < 0.03 for p in [p1, p2, p3]):
                score += 0.4
                if p3[1] < p2[1] and p2[1] < p1[1]:
                    score += 0.2
                trough1 = np.min(prices[p1[0]:p2[0]]) if p2[0] > p1[0] else prices[-1]
                trough2 = np.min(prices[p2[0]:p3[0]]) if p3[0] > p2[0] else prices[-1]
                neckline = (trough1 + trough2) / 2
                if prices[-1] < neckline * 1.01:
                    score += 0.25
                if df["Volume"].tail(3).mean() > df["Volume"].mean() * 1.2:
                    score += 0.15

        if score > 0.7:
            top      = float(df["High"].max())
            neckline = float(df["Low"].tail(40).min())
            height   = top - neckline
            return {
                "detected": True, "pattern": "Triple Top", "signal": "BEARISH",
                "confidence": "HIGH" if score > 0.85 else "MEDIUM", "score": score,
                "description": "Three rejections at same resistance. Very high-probability reversal.",
                "entry_price": round(neckline * 0.997),
                "stop_loss":   round(top * 1.003),
                "target_1":    round(neckline - height),
                "target_2":    round(neckline - height * 1.5),
                "action": "SHORT on neckline break. Higher conviction than double top.",
            }
        return {"detected": False}

    def detect_triple_bottom(self, lookback=70) -> Dict:
        df = self.data.tail(lookback).copy()
        if len(df) < 50:
            return {"detected": False}
        prices  = df["Close"].values
        troughs = self._find_troughs(prices, window=4)
        score   = 0

        if len(troughs) >= 3:
            t1, t2, t3 = troughs[-3], troughs[-2], troughs[-1]
            avg = (t1[1] + t2[1] + t3[1]) / 3
            if all(abs(t[1] - avg) / avg < 0.03 for t in [t1, t2, t3]):
                score += 0.4
                peak1 = np.max(prices[t1[0]:t2[0]]) if t2[0] > t1[0] else prices[-1]
                peak2 = np.max(prices[t2[0]:t3[0]]) if t3[0] > t2[0] else prices[-1]
                neckline = (peak1 + peak2) / 2
                if prices[-1] > neckline * 0.98:
                    score += 0.3
                if df["Volume"].tail(3).mean() > df["Volume"].mean() * 1.2:
                    score += 0.15
                score += 0.15  # Three tests = higher reliability

        if score > 0.7:
            bottom   = float(df["Low"].min())
            neckline = float(df["High"].tail(40).max())
            height   = neckline - bottom
            return {
                "detected": True, "pattern": "Triple Bottom", "signal": "BULLISH",
                "confidence": "HIGH" if score > 0.85 else "MEDIUM", "score": score,
                "description": "Three tests of major support. High-probability reversal zone.",
                "entry_price": round(neckline * 1.003),
                "stop_loss":   round(bottom * 0.997),
                "target_1":    round(neckline + height),
                "target_2":    round(neckline + height * 1.5),
                "action": "BUY on neckline breakout. Higher conviction than double bottom.",
            }
        return {"detected": False}

    # ──────────────────────────────────────────────────────────────────────────
    # ADVANCED / INDEX-SPECIFIC PATTERNS
    # ──────────────────────────────────────────────────────────────────────────

    def detect_range_breakout(self, lookback=30) -> Dict:
        """Detect a range consolidation followed by breakout — key for index options."""
        df = self.data.tail(lookback).copy()
        if len(df) < 20:
            return {"detected": False}

        base = df.iloc[:-5]
        breakout_bar = df.iloc[-1]
        rng = (base["High"].max() - base["Low"].min()) / base["Close"].mean()
        score = 0

        if rng < 0.04:   # Very tight range (< 4 %)
            score += 0.4
        elif rng < 0.07:
            score += 0.25

        vol_ratio = breakout_bar["Volume"] / df["Volume"].mean()
        if vol_ratio > 1.5:
            score += 0.3

        is_bullish = breakout_bar["Close"] > base["High"].max()
        is_bearish = breakout_bar["Close"] < base["Low"].min()

        if is_bullish or is_bearish:
            score += 0.3

        if score > 0.7:
            resistance = float(base["High"].max())
            support    = float(base["Low"].min())
            height     = resistance - support
            sig  = "BULLISH" if is_bullish else ("BEARISH" if is_bearish else "NEUTRAL")
            return {
                "detected": True, "pattern": "Range Breakout", "signal": sig,
                "confidence": "HIGH" if score > 0.85 else "MEDIUM", "score": score,
                "description": f"Index broke out of {rng*100:.1f}% range on {vol_ratio:.1f}× volume.",
                "entry_price": round(breakout_bar["Close"]),
                "stop_loss":   round(support * 0.997) if is_bullish else round(resistance * 1.003),
                "target_1":    round(resistance + height) if is_bullish else round(support - height),
                "target_2":    round(resistance + height*2) if is_bullish else round(support - height*2),
                "action": f"{'BUY' if is_bullish else 'SHORT'} on breakout confirmation",
            }
        return {"detected": False}

    def detect_cup_and_handle(self, lookback=120) -> Dict:
        df = self.data.tail(lookback).copy()
        if len(df) < 60:
            return {"detected": False}
        prices = df["Close"].values
        score  = 0

        min_idx = np.argmin(prices)
        if min_idx < 20 or min_idx > len(prices) - 20:
            return {"detected": False}

        cup_left  = prices[:min_idx]
        cup_right = prices[min_idx:]
        cup_top   = max(cup_left[0] if len(cup_left) else 0,
                        cup_right[-1] if len(cup_right) else 0)
        cup_bot   = prices[min_idx]
        depth = (cup_top - cup_bot) / cup_top if cup_top > 0 else 0

        if 0.10 <= depth <= 0.45:
            score += 0.3

        handle_start = int(len(df) * 0.80)
        hdata = df.iloc[handle_start:]
        if len(hdata) < 6:
            return {"detected": False}

        cup_mid = (cup_top + cup_bot) / 2
        if hdata["Close"].mean() > cup_mid:
            score += 0.2
        if hdata["Volume"].mean() < df["Volume"].iloc[:handle_start].mean() * 0.75:
            score += 0.2
        if np.polyfit(range(len(cup_left)), cup_left, 1)[0] < 0 and \
           np.polyfit(range(len(cup_right)), cup_right, 1)[0] > 0:
            score += 0.1
        h_range = (hdata["High"].max() - hdata["Low"].min()) / hdata["Close"].mean()
        if h_range < 0.12:
            score += 0.2

        if score > 0.7:
            h_high = float(hdata["High"].max())
            h_low  = float(hdata["Low"].min())
            cp     = float(df["Close"].iloc[-1])
            return {
                "detected": True, "pattern": "Cup & Handle", "signal": "BULLISH",
                "confidence": "HIGH" if score > 0.85 else "MEDIUM", "score": score,
                "description": "U-shaped consolidation + tight handle. Classic accumulation before breakout.",
                "entry_price": round(h_high * 1.003),
                "stop_loss":   round(h_low * 0.997),
                "target_1":    round(cp + (cup_top - cup_bot)),
                "target_2":    round(cp + (cup_top - cup_bot) * 1.5),
                "action": "BUY on handle breakout with 2× average volume",
            }
        return {"detected": False}

    def detect_flat_base(self, lookback=25) -> Dict:
        df = self.data.tail(lookback).copy()
        if len(df) < 15:
            return {"detected": False}
        score = 0

        rng = (df["High"].max() - df["Low"].min()) / df["Close"].mean()
        if rng < 0.08:
            score += 0.4

        vol_cv = df["Volume"].std() / df["Volume"].mean()
        if vol_cv < 0.45:
            score += 0.25

        recent_lows = df["Low"].tail(5)
        if all(l > df["Low"].min() * 0.98 for l in recent_lows):
            score += 0.2

        if np.polyfit(range(len(df)), df["Volume"].values, 1)[0] < 0:
            score += 0.15

        if score > 0.7:
            base_high = float(df["High"].max())
            base_low  = float(df["Low"].min())
            cp        = float(df["Close"].iloc[-1])
            return {
                "detected": True, "pattern": "Flat Base / Shelf", "signal": "BULLISH",
                "confidence": "MEDIUM", "score": score,
                "description": "Index consolidating in tight range — institutional accumulation phase.",
                "entry_price": round(base_high * 1.003),
                "stop_loss":   round(base_low * 0.997),
                "target_1":    round(cp * 1.08),
                "target_2":    round(cp * 1.15),
                "action": "BUY on pivot-point breakout with rising volume",
            }
        return {"detected": False}

    def detect_mean_reversion(self, lookback=60) -> Dict:
        df = self.data.tail(lookback).copy()
        if len(df) < 25:
            return {"detected": False}
        cp   = float(df["Close"].iloc[-1])
        s20  = float(df["Close"].tail(20).mean())
        sd20 = float(df["Close"].tail(20).std())
        rsi  = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else 50.0
        bb_h = float(df["BB_High"].iloc[-1]) if "BB_High" in df.columns else s20 + 2*sd20
        bb_l = float(df["BB_Low"].iloc[-1])  if "BB_Low"  in df.columns else s20 - 2*sd20
        score = 0

        dist = (cp - s20) / sd20 if sd20 > 0 else 0
        is_bull = dist < -1.5
        is_bear = dist >  1.5

        if abs(dist) > 2.0:
            score += 0.35
        elif abs(dist) > 1.5:
            score += 0.25

        if (is_bull and rsi < 35) or (is_bear and rsi > 65):
            score += 0.3
        if (is_bull and cp < bb_l) or (is_bear and cp > bb_h):
            score += 0.2

        recent_vol = df["Volume"].tail(3).mean()
        if recent_vol > df["Volume"].mean() * 1.4:
            score += 0.15

        if score > 0.7:
            if is_bull:
                return {
                    "detected": True, "pattern": "Mean Reversion (Bullish)", "signal": "BULLISH",
                    "confidence": "HIGH" if score > 0.85 else "MEDIUM", "score": score,
                    "description": f"Index {abs(dist):.1f}σ below mean. Statistical edge to the upside.",
                    "entry_price": round(cp),
                    "stop_loss":   round(cp * 0.97),
                    "target_1":    round(s20),
                    "target_2":    round(bb_h),
                    "action": "BUY oversold extreme — target mean/upper BB",
                }
            else:
                return {
                    "detected": True, "pattern": "Mean Reversion (Bearish)", "signal": "BEARISH",
                    "confidence": "HIGH" if score > 0.85 else "MEDIUM", "score": score,
                    "description": f"Index {abs(dist):.1f}σ above mean. Reversion to mean expected.",
                    "entry_price": round(cp),
                    "stop_loss":   round(cp * 1.03),
                    "target_1":    round(s20),
                    "target_2":    round(bb_l),
                    "action": "SHORT overbought extreme — target mean/lower BB",
                }
        return {"detected": False}

    def detect_elliott_wave(self, lookback=100) -> Dict:
        df = self.data.tail(lookback).copy()
        if len(df) < 60:
            return {"detected": False}
        prices = df["Close"].values
        peaks  = self._find_peaks(prices, window=5)
        troughs= self._find_troughs(prices, window=5)
        score  = 0

        # Need 5-wave sequence
        swings = sorted(peaks + troughs, key=lambda x: x[0])
        if len(swings) >= 9:
            alternating = all(swings[i][0] != swings[i+1][0] for i in range(len(swings)-1))
            if alternating:
                score += 0.3
            # Check basic Elliott rules on last 5 waves
            if len(swings) >= 9 and swings[-9][1] < swings[-8][1]:  # Bullish impulse start
                w1 = swings[-8][1] - swings[-9][1]
                w3 = swings[-6][1] - swings[-7][1]
                w5 = swings[-4][1] - swings[-5][1]
                if w3 >= w1 and w3 >= w5:
                    score += 0.25  # Wave 3 not shortest
                w2_retrace = (swings[-8][1] - swings[-7][1]) / w1 if w1 > 0 else 1
                if w2_retrace < 0.99:
                    score += 0.2
                score += 0.25  # Give base credit for having enough swings

        elif len(swings) >= 5:
            score += 0.35

        if score > 0.65:
            cp = float(df["Close"].iloc[-1])
            recent_high = float(df["High"].tail(30).max())
            recent_low  = float(df["Low"].tail(30).min())

            if len(swings) >= 9:
                w5_top = float(swings[-4][1])
                w4_low = float(swings[-5][1])
                height = w5_top - w4_low
                c38 = w5_top - height * 0.382
                c50 = w5_top - height * 0.50
                c62 = w5_top - height * 0.618
                return {
                    "detected": True, "pattern": "Elliott Wave (5-3 Complete)", "signal": "NEUTRAL",
                    "confidence": "MEDIUM", "score": score,
                    "description": "5-wave impulse complete. ABC correction likely before next leg up.",
                    "entry_price": round(c50),
                    "stop_loss":   round(c62 * 0.997),
                    "target_1":    round(w5_top * 1.05),
                    "target_2":    round(w5_top * 1.12),
                    "action": "WAIT for ABC correction, BUY at Wave-C completion (~50% Fib)",
                }
            else:
                return {
                    "detected": True, "pattern": "Elliott Wave (Impulse In Progress)", "signal": "BULLISH",
                    "confidence": "MEDIUM", "score": score,
                    "description": "Impulse wave in progress. Buy pullbacks within the trend.",
                    "entry_price": round(cp),
                    "stop_loss":   round(recent_low * 0.997),
                    "target_1":    round(cp * 1.08),
                    "target_2":    round(cp * 1.15),
                    "action": "BUY on Wave 2/4 pullbacks within impulse",
                }
        return {"detected": False}

    def detect_wyckoff_accumulation(self, lookback=80) -> Dict:
        df = self.data.tail(lookback).copy()
        if len(df) < 50:
            return {"detected": False}
        prices  = df["Close"].values
        volumes = df["Volume"].values
        score   = 0

        # Phase A: Selling Climax
        sc_idx = np.argmax(volumes[:len(volumes)//3]) if len(volumes) > 6 else 0
        if sc_idx > 0:
            if volumes[sc_idx] > np.mean(volumes) * 2.5 and prices[sc_idx] < prices[sc_idx-5]*0.96:
                score += 0.2

        # Phase B: Trading Range
        mid = df.iloc[len(df)//3 : 2*len(df)//3]
        if len(mid) > 5:
            rng = (mid["High"].max() - mid["Low"].min()) / mid["Close"].mean()
            if rng < 0.12:
                score += 0.2
            if mid["Volume"].mean() < df["Volume"].iloc[:len(df)//3].mean() * 0.75:
                score += 0.1

        # Phase C: Spring (lower low on lower volume)
        recent = df.tail(len(df)//3)
        if len(recent) > 5:
            rl = recent["Low"].min()
            ml = mid["Low"].min() if len(mid) > 0 else rl
            if rl < ml * 0.99:
                li = recent["Low"].idxmin()
                if df.loc[li, "Volume"] < df["Volume"].mean() * 0.85:
                    score += 0.25

        # Phase D: Sign of Strength
        last10 = df.tail(10)
        if last10["Close"].iloc[-1] > last10["Close"].iloc[0]:
            if last10["Volume"].tail(5).mean() > last10["Volume"].head(5).mean():
                score += 0.25

        if score > 0.65:
            rh = float(df["High"].tail(40).max())
            rl = float(df["Low"].tail(40).min())
            cp = float(df["Close"].iloc[-1])
            return {
                "detected": True, "pattern": "Wyckoff Accumulation", "signal": "BULLISH",
                "confidence": "HIGH" if score > 0.80 else "MEDIUM", "score": score,
                "description": "Selling climax → range → spring → sign of strength. Smart money loading.",
                "entry_price": round(rh * 1.003),
                "stop_loss":   round(rl * 0.997),
                "target_1":    round(cp * 1.10),
                "target_2":    round(rh + (rh - rl)),
                "action": "BUY on Phase-E markup breakout with volume expansion",
            }
        return {"detected": False}

    def detect_wyckoff_distribution(self, lookback=80) -> Dict:
        df = self.data.tail(lookback).copy()
        if len(df) < 50:
            return {"detected": False}
        prices  = df["Close"].values
        volumes = df["Volume"].values
        score   = 0

        # Phase A: Buying Climax
        bc_idx = np.argmax(volumes[:len(volumes)//3]) if len(volumes) > 6 else 0
        if bc_idx > 0:
            if volumes[bc_idx] > np.mean(volumes) * 2.5 and prices[bc_idx] > prices[bc_idx-5]*1.04:
                score += 0.2

        # Phase B: Range at top
        mid = df.iloc[len(df)//3 : 2*len(df)//3]
        if len(mid) > 5:
            rng = (mid["High"].max() - mid["Low"].min()) / mid["Close"].mean()
            if rng < 0.12:
                score += 0.2

        # Phase C: Upthrust (higher high on lower vol)
        recent = df.tail(len(df)//3)
        if len(recent) > 5:
            rh = recent["High"].max()
            mh = mid["High"].max() if len(mid) > 0 else rh
            if rh > mh * 1.005:
                hi = recent["High"].idxmax()
                if df.loc[hi, "Volume"] < df["Volume"].mean():
                    score += 0.25

        # Phase D: Sign of Weakness
        last10 = df.tail(10)
        if last10["Close"].iloc[-1] < last10["Close"].iloc[0]:
            if last10["Volume"].tail(5).mean() > last10["Volume"].head(5).mean():
                score += 0.25

        if score > 0.65:
            rh = float(df["High"].tail(40).max())
            rl = float(df["Low"].tail(40).min())
            cp = float(df["Close"].iloc[-1])
            return {
                "detected": True, "pattern": "Wyckoff Distribution", "signal": "BEARISH",
                "confidence": "HIGH" if score > 0.80 else "MEDIUM", "score": score,
                "description": "Buying climax → range → upthrust → sign of weakness. Smart money exiting.",
                "entry_price": round(rl * 0.997),
                "stop_loss":   round(rh * 1.003),
                "target_1":    round(cp * 0.92),
                "target_2":    round(rl - (rh - rl)),
                "action": "SHORT on Phase-E markdown breakdown with volume expansion",
            }
        return {"detected": False}

    # ──────────────────────────────────────────────────────────────────────────
    # BATCH DETECTION
    # ──────────────────────────────────────────────────────────────────────────

    def detect_all_trend_patterns(self) -> List[Dict]:
        detectors = [
            self.detect_ascending_triangle,
            self.detect_descending_triangle,
            self.detect_symmetrical_triangle,
            self.detect_bull_flag,
            self.detect_bear_flag,
            self.detect_rising_wedge,
            self.detect_falling_wedge,
            self.detect_pennant,
        ]
        return [r for d in detectors if (r := d()).get("detected")]

    def detect_all_reversal_patterns(self) -> List[Dict]:
        detectors = [
            self.detect_head_and_shoulders,
            self.detect_inverse_head_and_shoulders,
            self.detect_double_top,
            self.detect_double_bottom,
            self.detect_triple_top,
            self.detect_triple_bottom,
        ]
        return [r for d in detectors if (r := d()).get("detected")]

    def detect_all_advanced_patterns(self) -> List[Dict]:
        detectors = [
            self.detect_range_breakout,
            self.detect_cup_and_handle,
            self.detect_flat_base,
            self.detect_mean_reversion,
            self.detect_elliott_wave,
            self.detect_wyckoff_accumulation,
            self.detect_wyckoff_distribution,
        ]
        return [r for d in detectors if (r := d()).get("detected")]
