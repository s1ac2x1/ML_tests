import pandas as pd

file_path = "AES_h.csv"
df = pd.read_csv(file_path, parse_dates=["datetime"]).set_index("datetime")

daily_ohlcv = (
    df.resample("1D", label="right", closed="right")  # сутки заканчиваются в 24:00
      .agg({
          "open":   "first",
          "high":   "max",
          "low":    "min",
          "close":  "last",
          "volume": "sum"
      })
      .dropna()
)

daily_ohlcv.to_csv("AES_d.csv")