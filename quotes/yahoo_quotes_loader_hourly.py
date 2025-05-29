import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
import time  # для задержки между запросами

CHUNK_DAYS = 60  # максимальная длина одного запроса в днях
TOTAL_DAYS = 730  # сколько дней назад начинать

def setup_console_logger():
    """Настраивает логгер только для вывода в консоль"""
    logger = logging.getLogger("yahoo_logger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    return logger

def fetch_hourly_data(ticker: str, logger):
    """
    Загружает часовые котировки для тикера за последние TOTAL_DAYS дней,
    разбивая период на интервалы по CHUNK_DAYS дней.
    """
    now = datetime.now()
    start_total = now - timedelta(days=TOTAL_DAYS)
    end_total = now

    logger.info(f"🔍 Начинаем загрузку часовых данных для {ticker} с {start_total.date()} по {end_total.date()}")

    all_chunks = []
    chunk_start = start_total

    while chunk_start < end_total:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end_total)
        logger.info(f"📅 Запрос за период: {chunk_start.date()} – {chunk_end.date()}")

        try:
            df = yf.download(
                ticker,
                start=chunk_start.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                interval="1h",
                progress=False
            )
            if df.empty:
                logger.warning(f"⚠️ Нет данных за период {chunk_start.date()} – {chunk_end.date()}")
            else:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.index.name = 'Datetime'
                logger.info(f"✅ Получено {len(df)} записей")
                all_chunks.append(df)
        except Exception as e:
            logger.error(f"❌ Ошибка при запросе: {e}")

        # следующий чанка начинается сразу после конца предыдущего
        chunk_start = chunk_end
        time.sleep(1)  # пауза, чтобы не перегрузить сервер

    if all_chunks:
        result = pd.concat(all_chunks).sort_index()
        output_file = f"{ticker.upper()}_1h.csv"
        result.to_csv(output_file)
        logger.info(f"💾 Всего записей: {len(result)}. Сохранено в {output_file}")
    else:
        logger.warning("🚫 Нет данных для сохранения.")

def main():
    logger = setup_console_logger()

    tickers_file = "../tickers.txt"
    if not os.path.exists(tickers_file):
        logger.error(f"Файл {tickers_file} не найден.")
        return

    with open(tickers_file, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]

    if not tickers:
        logger.error("В файле tickers.txt нет тикеров.")
        return

    for idx, ticker in enumerate(tickers, 1):
        print(f"\n=== ▶ [{idx}/{len(tickers)}] Обработка тикера: {ticker} ===")
        fetch_hourly_data(ticker, logger)

if __name__ == "__main__":
    main()
