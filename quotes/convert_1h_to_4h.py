import pandas as pd

# Путь к вашему файлу
file_path = 'AES_1h.csv'

# Читаем CSV, пропуская строки с метаданными,
# парсим столбец 'Price' как даты и сразу ставим его в индекс
df = pd.read_csv(
    file_path,
    skiprows=[1, 2],               # пропускаем строки с тикером и заголовком 'Datetime'
    parse_dates=['Price'],         # колонка 'Price' содержит метку времени
    index_col='Price'
)

# Переименуем индекс для наглядности
df.index.name = 'Datetime'

# Преобразуем все колонки в числовой тип
df = df.astype({
    'Open': float,
    'High': float,
    'Low': float,
    'Close': float,
    'Volume': float
})

# Агрегируем в 4-часовые бары:
# - Open   = первое значение
# - High   = максимум
# - Low    = минимум
# - Close  = последнее значение
# - Volume = сумма объёмов
df_4h = df.resample(
    '4H',
    label='right',   # время в индексе будет концом интервала (например, 04:00)
    closed='right'   # включаем правую границу в интервал
).agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

# Опционально: сбросить ненужные интервалы без данных
df_4h = df_4h.dropna(subset=['Open'])

# Сохраняем результат
df_4h.to_csv('AES_4h.csv')
