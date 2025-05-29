import pandas as pd

file_path = "AES_h.csv"
df = pd.read_csv(file_path, parse_dates=["datetime"]).set_index("datetime")

# ── Агрегация в недельные бары ────────────────────────────────────────────────
# 'W-FRI' → неделя заканчивается в пятницу 23:59; при желании поменяй на
# 'W-SUN', 'W-MON' и т. д., либо на 'W' (по умолчанию заканчивается в воскресенье).
weekly_ohlcv = (
    df.resample("W-FRI", label="right", closed="right")
      .agg({
          "open":   "first",
          "high":   "max",
          "low":    "min",
          "close":  "last",
          "volume": "sum"
      })
      .dropna()
)

# Сохраняем результат
weekly_ohlcv.to_csv("AES_w.csv")