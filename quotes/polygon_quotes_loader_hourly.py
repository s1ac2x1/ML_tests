import requests
import pandas as pd
import time

# === НАСТРОЙКИ ===
API_KEY = "9j_C_F8fB29lulKXPysa3kSlLyuLSq_M"  # 🔑 вставь сюда свой API-ключ
symbol = "DB"
from_date = "2015-01-01"
to_date = "2025-01-01"
timespan = "hour"

# === СБОР ДАННЫХ С ПАГИНАЦИЕЙ ===
all_data = []
url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{timespan}/{from_date}/{to_date}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}"

while url:
    print(f"Запрашиваю: {url}")
    response = requests.get(url)
    if response.status_code != 200:
        print("Ошибка:", response.status_code, response.text)
        break

    data = response.json()

    # Проверка на наличие результатов
    results = data.get("results", [])
    all_data.extend(results)

    # Переход к следующей странице
    next_url = data.get("next_url")
    if next_url:
        url = next_url + f"&apiKey={API_KEY}"
        time.sleep(1.2)  # небольшая задержка для избежания rate limit
    else:
        break

# === ПРЕОБРАЗОВАНИЕ В DataFrame ===
df = pd.DataFrame(all_data)
if not df.empty:
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={
        't': 'datetime',
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume'
    })
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    df.to_csv(f"{symbol}_hourly_data.csv", index=False)
    print(f"✅ Сохранено {len(df)} строк в файл {symbol}_hourly_data.csv")
else:
    print("❌ Нет данных получено.")
