import pandas as pd

# Путь к вашему файлу с часовыми котировками
file_path = 'AES_1h.csv'

# Читаем CSV, пропуская строки с метаданными,
# парсим столбец 'Price' как даты и делаем его индексом
df = pd.read_csv(
    file_path,
    skiprows=[1, 2],           # пропускаем строки с тикером и дополнительным заголовком
    parse_dates=['Price'],     # колонка 'Price' содержит временные метки
    index_col='Price'
)
df.index.name = 'Datetime'

# Приводим колонки OHLCV к числовому типу (если ещё не в нём)
df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric)

# Агрегация в недельные бары (OHLCV)
# Используем «W-FRI», чтобы неделя шла с субботы по пятницу (рыночный календарь)
df_weekly = df.resample(
    'W-FRI',        # недельный интервал, кончается в пятницу
    label='right',  # в индексе ставится конец интервала (пятница в 00:00)
    closed='right'  # включаем пятничный момент в данные недели
).agg({
    'Open': 'first',   # первая цена в неделю
    'High': 'max',     # максимум за неделю
    'Low': 'min',      # минимум за неделю
    'Close': 'last',   # последняя цена в неделю
    'Volume': 'sum'    # суммарный объём за неделю
})

# Убираем периоды без сделок
df_weekly = df_weekly.dropna(subset=['Open'])

# Сохраняем результат
output_path = 'AES_weekly.csv'
df_weekly.to_csv(output_path)