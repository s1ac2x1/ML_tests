"""
bulk_polygon.py
---------------
Пакетная выгрузка часовых котировок Polygon.io
за период 2015-01-01 — 2025-01-01 с лимитом 20 RPS.

Зависимости:
    pip install aiohttp pandas tqdm certifi
"""

from __future__ import annotations

import asyncio
import ssl
from pathlib import Path

import aiohttp
import certifi
import pandas as pd
from aiohttp import ClientTimeout
from tqdm.asyncio import tqdm


# ===============  НАСТРОЙКИ  =================================================

API_KEY   = "9j_C_F8fB29lulKXPysa3kSlLyuLSq_M"   # 🔑 ваш ключ Polygon
FROM_DATE = "2015-01-01"
TO_DATE   = "2025-01-01"
TIMESPAN  = "hour"

MAX_REQUESTS_PER_SECOND = 20        # лимит подписки
CONCURRENT_SYMBOLS      = 50        # сколько тикеров обрабатываем одновременно
REQUEST_TIMEOUT         = 45        # сек. на любой GET (читать + ждать)

BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================


class RateLimiter:
    """≤ MAX_REQUESTS_PER_SECOND запросов в любую секунду."""
    def __init__(self, rps: int):
        self._rps = rps
        self._sem = asyncio.Semaphore(rps)
        asyncio.create_task(self._reset_loop())

    async def _reset_loop(self):
        while True:
            await asyncio.sleep(1)
            for _ in range(self._rps - self._sem._value):
                self._sem.release()

    async def __aenter__(self):
        await self._sem.acquire()

    async def __aexit__(self, exc_type, exc, tb):
        pass


async def fetch_json(session: aiohttp.ClientSession, url: str,
                     rl: RateLimiter, retries: int = 3) -> dict | None:
    """
    GET → JSON с тайм-аутом и повторами.
    Возврат None при окончательной неудаче.
    """
    for attempt in range(retries + 1):
        try:
            async with rl:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise aiohttp.ClientError(f"{resp.status}: {text[:120]}…")
                    return await resp.json()

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == retries:
                print(f"❌  {url} → {e}")
                return None
            await asyncio.sleep(2 ** attempt)       # 1 → 2 → 4 сек


async def download_symbol(symbol: str, session: aiohttp.ClientSession,
                          rl: RateLimiter) -> str:
    """Скачать все страницы одного тикера, сохранить в CSV, вернуть статус-строку."""
    base = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}"
            f"/range/1/{TIMESPAN}/{FROM_DATE}/{TO_DATE}"
            f"?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}")

    url, rows, pages = base, [], 0

    while url:
        data = await fetch_json(session, url, rl)
        if data is None:                           # безнадёжно
            return f"❌ {symbol}: ошибка запроса"

        rows.extend(data.get("results", []))
        pages += 1
        tqdm.write(f"{symbol}: страница {pages} готова")

        next_url = data.get("next_url")
        url = f"{next_url}&apiKey={API_KEY}" if next_url else None

    if not rows:
        return f"❌ {symbol}: нет данных"

    df = (pd.DataFrame(rows)
            .assign(datetime=lambda d: pd.to_datetime(d["t"], unit="ms"))
            .rename(columns={"o": "open", "h": "high", "l": "low",
                             "c": "close", "v": "volume"})
            .loc[:, ["datetime", "open", "high", "low", "close", "volume"]])

    outfile = OUTPUT_DIR / f"{symbol}.csv"
    df.to_csv(outfile, index=False)
    return f"✅ {symbol}: сохранено {len(df)} строк"


async def main() -> None:
    symbols_path = BASE_DIR / "symbols.txt"
    if not symbols_path.exists():
        print("Файл symbols.txt не найден.")
        return
    symbols = [s.strip() for s in symbols_path.read_text().splitlines() if s.strip()]
    if not symbols:
        print("Файл symbols.txt пуст.")
        return

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(
        ssl=ssl_ctx,
        limit=MAX_REQUESTS_PER_SECOND * 2            # соединений с запасом
    )
    timeout = ClientTimeout(total=REQUEST_TIMEOUT, connect=20)

    rl = RateLimiter(MAX_REQUESTS_PER_SECOND)

    async with aiohttp.ClientSession(timeout=timeout,
                                     connector=connector) as session:
        results: list[str] = []
        for i in range(0, len(symbols), CONCURRENT_SYMBOLS):
            chunk = symbols[i: i + CONCURRENT_SYMBOLS]
            tasks = [download_symbol(sym, session, rl) for sym in chunk]
            # tqdm.asyncio автоматически обновляет прогресс-бар
            results.extend(await tqdm.gather(*tasks, total=len(chunk)))

        print("\n--- Итоги ---")
        for line in results:
            print(line)


if __name__ == "__main__":
    asyncio.run(main())
