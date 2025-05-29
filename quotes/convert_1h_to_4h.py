import pandas as pd

# 1) Загрузка поминутного файла (замени путь при необходимости)
file_path = "AES_h.csv"
df = pd.read_csv(file_path, parse_dates=["datetime"]).set_index("datetime")

# 2) Агрегация в 4-часовые бары (OHLC + объём)
ohlcv_4h = (
    df.resample("4H", label="right", closed="right")      # границы баров: 00-04-08-12-16-20
      .agg({
          "open":   "first",   # цена открытия – первая в окне
          "high":   "max",     # максимум
          "low":    "min",     # минимум
          "close":  "last",    # закрытие – последняя в окне
          "volume": "sum"      # суммарный объём
      })
      .dropna()                # бары без сделок отбрасываем
)

# 3) Сохраняем или используем дальше
ohlcv_4h.to_csv("AES_4h.csv")