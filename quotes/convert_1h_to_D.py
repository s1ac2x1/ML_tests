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

# В случае, если числовые колонки имеют тип object, приводим их к числу
df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric)

# Агрегация в ежедневные бары (OHLCV)
# - Open   = первая цена за день
# - High   = максимум за день
# - Low    = минимум за день
# - Close  = последняя цена за день
# - Volume = суммарный объём за день
df_daily = df.resample(
    'D',             # дневной интервал
    label='right',   # в индексе ставится конец интервала (полночь следующего дня)
    closed='right'   # включаем правую границу (последний час дня)
).agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

# Убираем дни, где данных не было
df_daily = df_daily.dropna(subset=['Open'])

# Сохраняем результат в новый CSV
output_path = 'AES_daily.csv'
df_daily.to_csv(output_path)