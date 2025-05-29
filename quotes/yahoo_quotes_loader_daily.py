import yfinance as yf
import pandas as pd
from datetime import datetime
import logging
import os
import time  # для задержки

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

def fetch_yahoo_data(ticker, logger, start_year=2014):
    all_data = []
    end_year = datetime.now().year
    interval_years = 3

    logger.info(f"🔍 Загрузка данных для {ticker} с {start_year} по {end_year}")

    for year in range(start_year, end_year, interval_years):
        start_date = f"{year}-01-01"
        end_date = f"{min(year + interval_years, end_year)}-01-01"
        logger.info(f"📅 Запрос за период: {start_date} - {end_date}")

        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                logger.warning(f"⚠️ Нет данных за период: {start_date} - {end_date}")
            else:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.index.name = 'Date'
                logger.info(f"✅ Получено {len(df)} записей")
                all_data.append(df)
        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке: {str(e)}")

        time.sleep(1)  # задержка между реквестами

    if all_data:
        final_df = pd.concat(all_data)
        output_file = f"{ticker.upper()}.csv"
        final_df.to_csv(output_file)
        logger.info(f"💾 Сохранено {len(final_df)} котировок в {output_file}")
    else:
        logger.warning("🚫 Нет данных для сохранения.")

def main():
    logger = setup_console_logger()

    tickers_file = "../tickers.txt"
    if not os.path.exists(tickers_file):
        logger.error(f"Файл {tickers_file} не найден.")
        return

    with open(tickers_file, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    if not tickers:
        logger.error("В файле tickers.txt нет тикеров.")
        return

    for i, ticker in enumerate(tickers, 1):
        print(f"\n=== ▶ [{i}/{len(tickers)}] Обработка тикера: {ticker} ===")
        fetch_yahoo_data(ticker, logger)

if __name__ == "__main__":
    main()
