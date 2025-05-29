"""
Triple Screen Strategy — Grid Search Version
Compatible with vectorbt <= 0.24

Changes vs. previous script
---------------------------
1. **Смягчён фильтр дневного отката**: порог стохастика (K‑линии) теперь
   параметризован `STOCH_THR` (по умолчанию [20, 30, 40]).
2. **Гибкие стоп / тейк**: множители `SL_MULT` и `TP_MULT` передаются в функцию
   бэк‑теста и входят в сетку.
3. **Grid Search**: перебираются все комбинации `(STOCH_THR, SL_MULT, TP_MULT)`
   и выводятся ТОП‑10 результатов по Sharpe Ratio.

Файлы данных
~~~~~~~~~~~~
    quotes/AES_4h.csv   – 4‑hour bars
    quotes/AES_d.csv    – daily bars
    quotes/AES_w.csv    – weekly bars
Колонки: datetime, open, high, low, close, volume

Установка
~~~~~~~~~
```bash
pip install pandas pandas_ta "vectorbt<0.25" tabulate matplotlib
```

Запуск
~~~~~~
```bash
python triple_screen_grid.py
```
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pandas_ta as ta
import vectorbt as vbt
from tabulate import tabulate

# ────────────────────────── CONFIG & HYPER‑PARAMETERS ──────────────────────────
H4_FILE = Path("quotes/AES_4h.csv")
D_FILE  = Path("quotes/AES_d.csv")
W_FILE  = Path("quotes/AES_w.csv")
TZ       = "US/Eastern"
TICKER   = "AES"

# Capital & trading assumptions
INIT_CASH   = 100_000
FEE_PCT     = 0.001          # 0.1 %
POSITION_UNITS = 100          # buy 100 акций на сделку
PRICE_TICK  = 0.01

# Grid search ranges
STOCH_THR:   List[int]   = [20, 30, 40]           # стохастик ≤ threshold
SL_MULT:     List[float] = [1.5, 2.0, 3.0]         # SL  × ATR
TP_MULT:     List[float] = [2.0, 3.0, 4.0]         # TP  × ATR

REQ_COLS = ["datetime", "open", "high", "low", "close", "volume"]

# ───────────────────────────── CSV LOADER ──────────────────────────────

def load_csv(path: Path, tz: str, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    df = pd.read_csv(path)
    if [c.lower() for c in df.columns[:6]] != REQ_COLS:
        raise ValueError(f"{label}: first columns must be {REQ_COLS}")
    df.columns = [c.lower() for c in df.columns]
    ts = pd.to_datetime(df["datetime"], errors="raise")
    ts = ts.dt.tz_localize(tz) if ts.dt.tz is None else ts.dt.tz_convert(tz)
    return df.drop(columns="datetime").set_index(ts).sort_index()

# ───────────────────────────── INDICATORS ──────────────────────────────

def weekly_indicators(w: pd.DataFrame) -> pd.DataFrame:
    macd = ta.macd(w["close"], fast=12, slow=26, signal=9)
    w["macd_hist"]   = macd["MACDh_12_26_9"]
    w["ema13"]       = ta.ema(w["close"], length=13)
    w["ema13_slope"] = w["ema13"].diff()
    return w

def daily_indicators(d: pd.DataFrame) -> pd.DataFrame:
    stoch = ta.stoch(d["high"], d["low"], d["close"], k=14, d=3, smooth_k=3)
    d["stoch_d"] = stoch["STOCHd_14_3_3"]
    d["force13"] = ta.efi(d["close"], d["volume"], length=13)
    return d[["stoch_d", "force13"]]

def h4_indicators(h4: pd.DataFrame) -> pd.DataFrame:
    h4["atr14"] = ta.atr(h4["high"], h4["low"], h4["close"], length=14)
    return h4

# ─────────────────────── SIGNAL LOGIC & BACK‑TEST ──────────────────────

def build_signals(df: pd.DataFrame, stoch_thr: int) -> pd.Series:
    trend_long  = (df["macd_hist"] > 0) & (df["ema13_slope"] > 0)
    pullback    = (df["stoch_d"] < stoch_thr) & (df["force13"] < 0)
    entry_setup = trend_long & pullback
    return entry_setup.shift(1).fillna(False)


def backtest(df: pd.DataFrame, entries: pd.Series, sl_mult: float, tp_mult: float) -> vbt.Portfolio:
    limit_price = (df["high"].shift(1) + PRICE_TICK).where(entries)
    price_entry = limit_price.fillna(df["close"])  # vectorbt требует цену на каждой свече

    sl_pct = (sl_mult * df["atr14"]) / df["close"]
    tp_pct = (tp_mult * df["atr14"]) / df["close"]

    return vbt.Portfolio.from_signals(
        close=df["close"],
        entries=entries,
        exits=None,
        size=POSITION_UNITS,
        price=price_entry,
        sl_stop=sl_pct,
        tp_stop=tp_pct,
        fees=FEE_PCT,
        init_cash=INIT_CASH,
        freq="4h",
        direction="longonly",
    )

# ───────────────────────────── MAIN ────────────────────────────────────

def main():
    print("Loading CSVs …")
    h4 = load_csv(H4_FILE, TZ, "4‑hour")
    d  = load_csv(D_FILE,  TZ, "Daily")
    w  = load_csv(W_FILE,  TZ, "Weekly")

    print("Calculating indicators …")
    h4 = h4_indicators(h4)
    d  = daily_indicators(d)
    w  = weekly_indicators(w)

    print("Building master frame …")
    df = h4.join(d, how="left").ffill()
    df = df.join(w[["macd_hist", "ema13_slope"]], how="left").ffill()
    df = df.dropna(subset=["macd_hist", "stoch_d", "atr14"])

    combos: List[Tuple[int, float, float]] = list(itertools.product(STOCH_THR, SL_MULT, TP_MULT))
    results = []

    print(f"Testing {len(combos)} parameter sets …")
    for st_thr, sl, tp in combos:
        entries = build_signals(df, st_thr)
        pf = backtest(df, entries, sl, tp)
        sharpe = pf.stats()["Sharpe Ratio"]
        total_ret = pf.stats()["Total Return [%]"]
        results.append((st_thr, sl, tp, sharpe, total_ret))

    # Sort by Sharpe desc
    results.sort(key=lambda x: x[3], reverse=True)

    print("\nTop‑10 parameter sets by Sharpe:\n")
    table = tabulate(
        [(i+1, *r) for i, r in enumerate(results[:10])],
        headers=["Rank", "Stoch≤", "SL×ATR", "TP×ATR", "Sharpe", "Total Ret %"],
        floatfmt=(".0f", ".0f", ".2f", ".2f", ".2f", ".2f"),
        tablefmt="github"
    )
    print(table)

    best = results[0]
    print(f"\nRunning full back‑test with best params: Stoch≤{best[0]}, SL={best[1]}×ATR, TP={best[2]}×ATR …")

    best_entries = build_signals(df, best[0])
    best_pf = backtest(df, best_entries, best[1], best[2])

    print("\n=== Triple Screen •", TICKER, "• Best run ===")
    print(best_pf.stats().to_string(float_format="{:.2f}".format))

    try:
        best_pf.plot(title=f"{TICKER} – Triple Screen (best params)").show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
