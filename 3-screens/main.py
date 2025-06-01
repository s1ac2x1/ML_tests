"""
Elder Triple Screen trading system — full back‑test implementation
---------------------------------------------------------------
*   **Time‑frames:**
        1. Weekly   (first screen – trend filter)
        2. Daily    (second screen – oscillator correction)
        3. Hourly   (third screen – trigger / execution)
*   **Indicators used (defaults can be changed):**
        • Weekly MACD histogram   – trend direction  (bullish > 0, bearish < 0)
        • Daily Stochastic (14,3,3)  – oversold / overbought
        • Hourly ATR(14)           – initial & trailing stop

Data requirements
-----------------
A CSV file with **hourly OHLCV** data for a single instrument,
containing at least the following columns and a header row:
    datetime, open, high, low, close, volume
The script will read     ``your_hourly_file.csv``.  Change the path or add
CLI arguments if needed.

Libraries:  pandas, numpy, matplotlib (for the optional equity curve plot).
Install with:
    pip install pandas numpy matplotlib

Run:
    python elder_triple_screen_backtest.py  # after editing the FILE_PATH constant

Outputs:
    • Summary statistics printed to console
    • equity_curve.png — cumulative equity curve (optional)
    • trades.csv       — detailed log of all closed trades

Feel free to tweak parameters in the CONFIG dictionary near the top.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CONFIG = {
    "FILE_PATH": "AES_h.csv",  # path to hourly data CSV
    "initial_capital": 100_000,            # starting cash for back‑test
    "risk_per_trade": 0.02,               # 2% of equity per trade
    # MACD parameters (weekly)
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    # Stochastic parameters (daily)
    "stoch_k": 14,
    "stoch_d": 3,
    "stoch_smooth": 3,
    "stoch_overbought": 80,
    "stoch_oversold": 20,
    # ATR parameters (hourly)
    "atr_period": 14,
    "atr_mult_stop": 2.0,                 # stop‑loss = ATR * N
    # Commissions & slippage
    "commission_perc": 0.0005,            # 0.05% each trade leg
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
    # Ensure the dataframe has required columns
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample to rule (e.g. 'D', 'W-FRI') returning full OHLCV."""
    ohlc_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return df.resample(rule, label="right", closed="right").agg(ohlc_dict).dropna()

# ---------------------------------------------------------------------------
# BACK‑TEST ENGINE
# ---------------------------------------------------------------------------
class TripleScreenBacktester:
    def __init__(self, hourly: pd.DataFrame, config: dict):
        self.hourly = hourly.copy()
        self.cfg = config
        self.prepare_frames()
        self.trades = []  # list of dicts for each completed trade

    # ---------------------------------------------------------
    def prepare_frames(self):
        # Weekly frame (first screen)
        self.weekly = resample_ohlc(self.hourly, "W-FRI")
        _, _, self.weekly["macd_hist"] = macd(
            self.weekly["close"],
            self.cfg["macd_fast"],
            self.cfg["macd_slow"],
            self.cfg["macd_signal"],
        )
        # Daily frame (second screen)
        self.daily = resample_ohlc(self.hourly, "D")
        k, d = stochastic(
            self.daily["high"],
            self.daily["low"],
            self.daily["close"],
            self.cfg["stoch_k"],
            self.cfg["stoch_d"],
            self.cfg["stoch_smooth"],
        )
        self.daily["stoch_k"] = k
        self.daily["stoch_d"] = d
        # ATR on hourly
        self.hourly["atr"] = atr(
            self.hourly["high"],
            self.hourly["low"],
            self.hourly["close"],
            self.cfg["atr_period"],
        )
        # Map weekly & daily indicators down to hourly rows via reindex + ffill
        self.hourly = self.hourly.join(self.weekly["macd_hist"].rename("macd_hist"), how="left").ffill()
        self.hourly = self.hourly.join(self.daily[["stoch_k", "stoch_d"]], how="left").ffill()

    # ---------------------------------------------------------
    def run(self):
        equity = self.cfg["initial_capital"]
        position = 0       # +1 long, -1 short, 0 flat
        entry_price = np.nan
        stop_price = np.nan
        quantity = 0
        peak_equity = equity
        equity_curve = []

        data = self.hourly.itertuples()
        for bar in data:
            dt = bar.Index
            price_o = bar.open
            price_h = bar.high
            price_l = bar.low
            price_c = bar.close
            atr_val = bar.atr
            macd_hist = bar.macd_hist
            stoch_k = bar.stoch_k
            # Determine bias from first screen
            trend = "bull" if macd_hist > 0 else "bear"
            # Determine daily oscillator status
            daily_signal = None
            if stoch_k < self.cfg["stoch_oversold"]:
                daily_signal = "oversold"
            elif stoch_k > self.cfg["stoch_overbought"]:
                daily_signal = "overbought"
            # ----------------------------------------------------------------
            # ENTRY LOGIC (third screen)
            # ----------------------------------------------------------------
            if position == 0 and not np.isnan(atr_val):
                risk_cash = equity * self.cfg["risk_per_trade"]
                if trend == "bull" and daily_signal == "oversold":
                    # Buy stop: break above previous bar high
                    prev_high = self.hourly.loc[:dt].iloc[-2].high if len(self.hourly.loc[:dt]) > 1 else np.nan
                    if price_h > prev_high and price_c > prev_high:
                        entry_price = prev_high
                        stop_price = entry_price - self.cfg["atr_mult_stop"] * atr_val
                        quantity = risk_cash / (entry_price - stop_price)
                        commission = entry_price * quantity * self.cfg["commission_perc"]
                        equity -= commission
                        position = +1
                        self.trades.append({
                            "entry_dt": dt,
                            "direction": "LONG",
                            "entry_px": entry_price,
                            "size": quantity,
                            "entry_commission": commission,
                        })
                        continue  # move to next bar
                elif trend == "bear" and daily_signal == "overbought":
                    # Sell stop: break below previous bar low
                    prev_low = self.hourly.loc[:dt].iloc[-2].low if len(self.hourly.loc[:dt]) > 1 else np.nan
                    if price_l < prev_low and price_c < prev_low:
                        entry_price = prev_low
                        stop_price = entry_price + self.cfg["atr_mult_stop"] * atr_val
                        quantity = risk_cash / (stop_price - entry_price)
                        commission = entry_price * quantity * self.cfg["commission_perc"]
                        equity -= commission
                        position = -1
                        self.trades.append({
                            "entry_dt": dt,
                            "direction": "SHORT",
                            "entry_px": entry_price,
                            "size": quantity,
                            "entry_commission": commission,
                        })
                        continue

            # ----------------------------------------------------------------
            # POSITION MANAGEMENT
            # ----------------------------------------------------------------
            if position != 0:
                # Update trailing stop
                if position == 1:
                    # Long – trail stop with highest close – ATR*mult
                    stop_price = max(stop_price, price_h - self.cfg["atr_mult_stop"] * atr_val)
                    exit_reason = None
                    if price_l <= stop_price:
                        exit_price = stop_price
                        exit_reason = "stop"
                    elif trend == "bear":
                        exit_price = price_c
                        exit_reason = "trend_flip"
                    else:
                        exit_price = None
                    if exit_price is not None:
                        commission = exit_price * quantity * self.cfg["commission_perc"]
                        pnl = (exit_price - entry_price) * quantity - commission - self.trades[-1]["entry_commission"]
                        equity += pnl
                        self.trades[-1].update({
                            "exit_dt": dt,
                            "exit_px": exit_price,
                            "exit_commission": commission,
                            "pnl": pnl,
                            "exit_reason": exit_reason,
                        })
                        position = 0
                        entry_price = np.nan
                        stop_price = np.nan
                        quantity = 0

                elif position == -1:
                    # Short – trail stop with lowest close + ATR*mult
                    stop_price = min(stop_price, price_l + self.cfg["atr_mult_stop"] * atr_val)
                    exit_reason = None
                    if price_h >= stop_price:
                        exit_price = stop_price
                        exit_reason = "stop"
                    elif trend == "bull":
                        exit_price = price_c
                        exit_reason = "trend_flip"
                    else:
                        exit_price = None
                    if exit_price is not None:
                        commission = exit_price * quantity * self.cfg["commission_perc"]
                        pnl = (entry_price - exit_price) * quantity - commission - self.trades[-1]["entry_commission"]
                        equity += pnl
                        self.trades[-1].update({
                            "exit_dt": dt,
                            "exit_px": exit_price,
                            "exit_commission": commission,
                            "pnl": pnl,
                            "exit_reason": exit_reason,
                        })
                        position = 0
                        entry_price = np.nan
                        stop_price = np.nan
                        quantity = 0

            peak_equity = max(peak_equity, equity)
            equity_curve.append((dt, equity))

        self.equity_curve = pd.DataFrame(equity_curve, columns=["datetime", "equity"]).set_index("datetime")

    # ---------------------------------------------------------
    def stats(self):
        trades = pd.DataFrame(self.trades)
        closed = trades.dropna(subset=["exit_dt"]).copy()
        if closed.empty:
            print("No completed trades.")
            return
        total_return = (self.equity_curve.iloc[-1].equity / self.cfg["initial_capital"] - 1) * 100
        win_trades = closed[closed.pnl > 0]
        loss_trades = closed[closed.pnl <= 0]
        win_rate = len(win_trades) / len(closed) * 100
        avg_win = win_trades.pnl.mean() if not win_trades.empty else 0
        avg_loss = loss_trades.pnl.mean() if not loss_trades.empty else 0
        expectancy = (win_rate/100) * avg_win + ((100-win_rate)/100) * avg_loss

        print("\n--- BACK‑TEST SUMMARY ---")
        print(f"Trades executed      : {len(closed)}")
        print(f"Win rate             : {win_rate:.1f}%")
        print(f"Average win          : {avg_win:,.2f}")
        print(f"Average loss         : {avg_loss:,.2f}")
        print(f"Expectancy per trade : {expectancy:,.2f}")
        print(f"Total return         : {total_return:,.2f}%")
        # Save trades & equity
        closed.to_csv("trades.csv", index=False)
        self.equity_curve.to_csv("equity_curve.csv")
        # Equity plot
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
