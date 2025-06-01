"""
Elder Triple Screen trading system — **long‑only, soft‑trailing** back‑test
===========================================================================
This version implements the user‑requested tweaks:

*   Only **long** positions (shorts отключены).
*   Start capital ← **10 000 $**.
*   Daily Stochastic levels moved to **30 / 70** (чуть чаще коррекции).
*   Initial protective stop = **ATR × 2**; after цена проходит **+1 R**, включается
    трейлинг‑стоп (ATR × 2 от последней цены).
*   Выход по тренду — когда недельная MACD‑гистограмма ≤ 0 (раньше, чем по
    классическому «смена знака» для лонгов).

Data requirements
-----------------
CSV with **hourly** OHLCV for a single instrument, columns:
    `datetime, open, high, low, close, volume`

Run:
    python elder_triple_screen_backtest.py  (после указания пути к CSV)
Outputs:
    `trades.csv`, `equity_curve.csv`, `equity_curve.png`, консольная сводка.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CONFIG = {
    "FILE_PATH": "DB.csv",  # path to hourly data CSV
    "initial_capital": 10_000,             # start with 10k USD
    "risk_per_trade": 0.02,               # 2 % of equity each trade
    # MACD params (weekly)
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    # Stochastic params (daily)
    "stoch_k": 14,
    "stoch_d": 3,
    "stoch_smooth": 3,
    "stoch_oversold": 30,
    "stoch_overbought": 70,               # not used but kept
    # ATR (hourly)
    "atr_period": 14,
    "atr_mult_stop": 2.0,                 # initial & trailing = ATR*2
    # Commission
    "commission_perc": 0.0005,            # 0.05 % per leg
}

# ---------------------------------------------------------------------------
# INDICATOR HELPERS
# ---------------------------------------------------------------------------

def macd(close: pd.Series, fast: int, slow: int, signal: int):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int, d_period: int, smooth: int):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_raw = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k = k_raw.rolling(window=smooth).mean()
    d = k.rolling(window=d_period).mean()
    return k, d


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# ---------------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return df.resample(rule, label="right", closed="right").agg(agg).dropna()

# ---------------------------------------------------------------------------
# BACK‑TEST ENGINE
# ---------------------------------------------------------------------------
class TripleScreenBacktester:
    def __init__(self, hourly: pd.DataFrame, cfg: dict):
        self.hourly = hourly.copy()
        self.cfg = cfg
        self.prepare_frames()
        self.trades = []

    def prepare_frames(self):
        # Weekly MACD
        weekly = resample_ohlc(self.hourly, "W-FRI")
        _, _, weekly["macd_hist"] = macd(
            weekly["close"], self.cfg["macd_fast"], self.cfg["macd_slow"], self.cfg["macd_signal"]
        )
        # Daily Stochastic
        daily = resample_ohlc(self.hourly, "D")
        k, d = stochastic(
            daily["high"], daily["low"], daily["close"],
            self.cfg["stoch_k"], self.cfg["stoch_d"], self.cfg["stoch_smooth"]
        )
        daily["stoch_k"] = k
        # Hourly ATR
        self.hourly["atr"] = atr(
            self.hourly["high"], self.hourly["low"], self.hourly["close"], self.cfg["atr_period"]
        )
        # Merge
        self.hourly = self.hourly.join(weekly["macd_hist"], how="left").ffill()
        self.hourly = self.hourly.join(daily[["stoch_k"]], how="left").ffill()

    # ---------------------------------------------------------
    def run(self):
        equity = self.cfg["initial_capital"]
        position = 0             # 0 flat, 1 long
        entry_px = stop_px = qty = np.nan
        r_value = np.nan         # risk per share
        trail_active = False
        equity_curve = []

        for bar in self.hourly.itertuples():
            dt = bar.Index
            o, h, l, c = bar.open, bar.high, bar.low, bar.close
            atr_val = bar.atr
            macd_hist = bar.macd_hist
            stoch_k = bar.stoch_k

            bull_trend = macd_hist > 0            # filter
            oversold = stoch_k < self.cfg["stoch_oversold"]

            # -------------------- ENTRY --------------------
            if position == 0 and bull_trend and oversold and not np.isnan(atr_val):
                prev_high = self.hourly.loc[:dt].iloc[-2].high if len(self.hourly.loc[:dt]) > 1 else np.nan
                if h > prev_high and c > prev_high:   # trigger
                    risk_cash = equity * self.cfg["risk_per_trade"]
                    entry_px = prev_high
                    stop_px = entry_px - self.cfg["atr_mult_stop"] * atr_val
                    r_value = entry_px - stop_px
                    qty = risk_cash / r_value
                    commission = entry_px * qty * self.cfg["commission_perc"]
                    equity -= commission
                    position = 1
                    trail_active = False
                    self.trades.append({
                        "entry_dt": dt, "entry_px": entry_px, "size": qty, "entry_commission": commission
                    })
                    continue

            # ----------------- POSITION MGT -----------------
            if position == 1:
                # Activate trailing after +1R move
                if not trail_active and (c - entry_px) >= r_value:
                    trail_active = True
                if trail_active:
                    stop_px = max(stop_px, c - self.cfg["atr_mult_stop"] * atr_val)

                exit_flag = None
                if l <= stop_px:
                    exit_px = stop_px
                    exit_flag = "stop_hit"
                elif macd_hist <= 0:               # weekly trend gone
                    exit_px = c
                    exit_flag = "trend_flip"
                else:
                    exit_px = None

                if exit_px is not None:
                    commission = exit_px * qty * self.cfg["commission_perc"]
                    pnl = (exit_px - entry_px) * qty - commission - self.trades[-1]["entry_commission"]
                    equity += pnl
                    self.trades[-1].update({
                        "exit_dt": dt, "exit_px": exit_px, "exit_commission": commission,
                        "pnl": pnl, "exit_reason": exit_flag
                    })
                    position = 0
                    entry_px = stop_px = qty = np.nan

            equity_curve.append((dt, equity))

        self.equity_curve = pd.DataFrame(equity_curve, columns=["datetime", "equity"]).set_index("datetime")

    # ---------------------------------------------------------
    def stats(self):
        trades = pd.DataFrame(self.trades)
        closed = trades.dropna(subset=["exit_dt"])
        if closed.empty:
            print("No completed trades.")
            return
        total_return = (self.equity_curve.iloc[-1].equity / self.cfg["initial_capital"] - 1) * 100
        win_rate = (closed.pnl > 0).mean() * 100
        avg_win = closed.loc[closed.pnl > 0, "pnl"].mean()
        avg_loss = closed.loc[closed.pnl <= 0, "pnl"].mean()
        expectancy = (win_rate/100) * avg_win + (1 - win_rate/100) * avg_loss

        print("\n--- BACK‑TEST SUMMARY (long‑only, soft‑trailing) ---")
        print(f"Trades executed      : {len(closed)}")
        print(f"Win rate             : {win_rate:.1f} %")
        print(f"Average win          : {avg_win:,.2f}")
        print(f"Average loss         : {avg_loss:,.2f}")
        print(f"Expectancy per trade : {expectancy:,.2f}")
        print(f"Total return         : {total_return:,.2f} %")

        closed.to_csv("trades.csv", index=False)
        self.equity_curve.to_csv("equity_curve.csv")
        self.equity_curve.equity.plot(title="Equity Curve", figsize=(10, 4))
        plt.tight_layout()
        plt.savefig("equity_curve.png", dpi=150)
        print("\nFiles generated: trades.csv, equity_curve.csv, equity_curve.png")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = CONFIG
    path = Path(cfg["FILE_PATH"])
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    hourly_df = load_data(path)
    tester = TripleScreenBacktester(hourly_df, cfg)
    tester.run()
    tester.stats()
