"""
Batch back‑test — Elder Triple Screen (long‑only, soft‑trailing)
================================================================
Runs the strategy on **every CSV file** inside a folder (default: `data/`).
Outputs a single **`summary.csv`** with one row per ticker:

| ticker | trades | win_rate_% | avg_win | avg_loss | total_return_% |

Key strategy parameters are unchanged:
* hourly data; only long setups; Stoch 30/70; ATR×2 initial & trailing; exit on weekly‑MACD‑hist ≤ 0.
* Start capital = 10 000 $, risk per trade = 2 %.

Run:
    python triple_screen_batch.py   # (rename file as you like)

If you want to keep individual equity/trade logs → set `SAVE_INDIVIDUAL = True` in CONFIG.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CONFIG = {
    "DATA_DIR": "data",          # folder with *.csv hourly files
    "initial_capital": 10_000,
    "risk_per_trade": 0.02,
    # MACD parameters (weekly)
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    # Stochastic parameters (daily)
    "stoch_k": 14,
    "stoch_d": 3,
    "stoch_smooth": 3,
    "stoch_oversold": 30,
    "stoch_overbought": 70,
    # ATR parameters (hourly)
    "atr_period": 14,
    "atr_mult_stop": 2.0,
    # Commissions
    "commission_perc": 0.0005,
    # saving options
    "SAVE_INDIVIDUAL": False,     # set True to save per‑ticker equity/trades png/csv
}

# ---------------------------------------------------------------------------
# INDICATOR HELPERS
# ---------------------------------------------------------------------------

def macd(close, fast, slow, signal):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def stochastic(high, low, close, k_period, d_period, smooth):
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k_raw = 100 * (close - lowest) / (highest - lowest)
    k = k_raw.rolling(smooth).mean()
    d = k.rolling(d_period).mean()
    return k, d


def atr(high, low, close, period):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ---------------------------------------------------------------------------
# DATA HELPERS
# ---------------------------------------------------------------------------

def load_csv(path: Path):
    df = pd.read_csv(path, parse_dates=["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    req = {"open", "high", "low", "close"}
    if not req.issubset(df.columns):
        raise ValueError(f"{path.name} missing columns {req - set(df.columns)}")
    return df


def resample_ohlc(df, rule):
    return df.resample(rule, label="right", closed="right").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

# ---------------------------------------------------------------------------
# BACK‑TEST ENGINE (same as before but returns stats dict)
# ---------------------------------------------------------------------------
class TripleScreenBT:
    def __init__(self, hourly: pd.DataFrame, cfg: dict):
        self.h = hourly.copy()
        self.cfg = cfg
        self._prep()
        self.trades = []
        self.equity_curve = None

    def _prep(self):
        weekly = resample_ohlc(self.h, "W-FRI")
        _, _, weekly["macd_hist"] = macd(weekly["close"], self.cfg["macd_fast"], self.cfg["macd_slow"], self.cfg["macd_signal"])
        daily = resample_ohlc(self.h, "D")
        k, _ = stochastic(daily["high"], daily["low"], daily["close"], self.cfg["stoch_k"], self.cfg["stoch_d"], self.cfg["stoch_smooth"])
        daily["stoch_k"] = k
        self.h["atr"] = atr(self.h["high"], self.h["low"], self.h["close"], self.cfg["atr_period"])
        self.h = self.h.join(weekly["macd_hist"], how="left").ffill()
        self.h = self.h.join(daily[["stoch_k"]], how="left").ffill()

    def run(self):
        eq = self.cfg["initial_capital"]
        pos = 0
        entry = stop = qty = np.nan
        r_val = np.nan
        trail = False
        curve = []

        for bar in self.h.itertuples():
            dt, h, l, c, atr_val, macd_hist, stoch_k = bar.Index, bar.high, bar.low, bar.close, bar.atr, bar.macd_hist, bar.stoch_k
            bull = macd_hist > 0
            oversold = stoch_k < self.cfg["stoch_oversold"]
            # entry
            if pos == 0 and bull and oversold and not np.isnan(atr_val):
                prev_high = self.h.loc[:dt].iloc[-2].high if len(self.h.loc[:dt]) > 1 else np.nan
                if h > prev_high and c > prev_high:
                    risk_cash = eq * self.cfg["risk_per_trade"]
                    entry = prev_high
                    stop = entry - self.cfg["atr_mult_stop"] * atr_val
                    r_val = entry - stop
                    qty = risk_cash / r_val
                    eq -= entry * qty * self.cfg["commission_perc"]
                    pos = 1
                    trail = False
                    self.trades.append({"entry_dt": dt, "entry_px": entry, "size": qty, "entry_comm": entry * qty * self.cfg["commission_perc"]})
                    continue
            # manage
            if pos == 1:
                if not trail and (c - entry) >= r_val:
                    trail = True
                if trail:
                    stop = max(stop, c - self.cfg["atr_mult_stop"] * atr_val)
                exit_price = None
                if l <= stop:
                    exit_price = stop
                    reason = "stop"
                elif macd_hist <= 0:
                    exit_price = c
                    reason = "trend_flip"
                if exit_price is not None:
                    exit_comm = exit_price * qty * self.cfg["commission_perc"]
                    pnl = (exit_price - entry) * qty - exit_comm - self.trades[-1]["entry_comm"]
                    eq += pnl
                    self.trades[-1].update({"exit_dt": dt, "exit_px": exit_price, "exit_comm": exit_comm, "pnl": pnl, "exit_reason": reason})
                    pos = 0
            curve.append((dt, eq))
        self.equity_curve = pd.DataFrame(curve, columns=["datetime", "equity"]).set_index("datetime")

    def summary(self):
        df = pd.DataFrame(self.trades).dropna(subset=["exit_dt"])
        if df.empty:
            return {"trades": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0, "total_return": 0}
        wins = df[df.pnl > 0]
        losses = df[df.pnl <= 0]
        total_ret = (self.equity_curve.iloc[-1].equity / self.cfg["initial_capital"] - 1) * 100
        return {
            "trades": len(df),
            "win_rate": round((len(wins) / len(df)) * 100, 1),
            "avg_win": wins.pnl.mean() if not wins.empty else 0,
            "avg_loss": losses.pnl.mean() if not losses.empty else 0,
            "total_return": round(total_ret, 2),
        }

# ---------------------------------------------------------------------------
# BATCH RUNNER
# ---------------------------------------------------------------------------

def main():
    cfg = CONFIG.copy()
    data_dir = Path(cfg.pop("DATA_DIR"))
    save_individual = cfg.pop("SAVE_INDIVIDUAL", False)

    rows = []
    for csv_file in sorted(data_dir.glob("*.csv")):
        ticker = csv_file.stem.upper()
        try:
            hourly = load_csv(csv_file)
        except Exception as e:
            print(f"Skipping {ticker}: {e}")
            continue
        bt = TripleScreenBT(hourly, cfg)
        bt.run()
        stats = bt.summary()
        rows.append({"ticker": ticker, **stats})
        print(f"{ticker}: {stats}")

        if save_individual and stats["trades"] > 0:
            out_base = data_dir / f"{ticker}_"
            # trades / equity
            pd.DataFrame(bt.trades).to_csv(out_base.with_suffix("_trades.csv"), index=False)
            bt.equity_curve.to_csv(out_base.with_suffix("_equity.csv"))
            bt.equity_curve.equity.plot(title=f"Equity {ticker}")
            plt.tight_layout(); plt.savefig(out_base.with_suffix("_equity.png"), dpi=150); plt.clf()

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv("summary.csv", index=False)
    print("\nSaved summary.csv with", len(summary_df), "tickers.")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
