import sys
import json
import time
import datetime as dt
from zoneinfo import ZoneInfo           # Python 3.9+
import yfinance as yf
import pandas as pd

# ----------------------- parameters -----------------------------------------
TICKERS_FILE    = "sp500.txt"            # one ticker per line
START_DATE      = dt.date(2007, 1, 1)    # from Jan 1, 2007
CHUNK_YEARS     = 2                      # size of each slice in years
REQUEST_TIMEOUT = 15                     # seconds for HTTP timeout
DELAY_BETWEEN   = 3                      # seconds between tickers
PAUSE_BETWEEN_CHUNKS = 1                # seconds between chunk requests
OUTPUT_DIR      = ".."  # where to save "<TICKER>_quotes.txt"

TZ_NATIVE = ZoneInfo("US/Eastern")       # exchange local time
TZ_MY     = ZoneInfo("Europe/Berlin")    # your local time

# ------------------- helper: make date chunks -------------------------------
def make_date_chunks(start_date: dt.date, end_date: dt.date, years: int):
    """
    Split [start_date, end_date) into slices of `years` years each.
    Returns a list of (slice_start, slice_end) as date objects.
    """
    chunks = []
    cur = start_date
    while cur < end_date:
        try:
            nxt = dt.date(cur.year + years, cur.month, cur.day)
        except ValueError:
            # handle leap day for non-leap target year
            nxt = dt.date(cur.year + years, cur.month, cur.day - 1)
        if nxt > end_date:
            nxt = end_date
        chunks.append((cur, nxt))
        cur = nxt
    return chunks

# ------------------- compute today's end (include today) --------------------
today     = dt.date.today()
end_date  = today + dt.timedelta(days=1)  # so you include today's bar if available
date_chunks = make_date_chunks(START_DATE, end_date, CHUNK_YEARS)

# ----------------------- read tickers ---------------------------------------
with open(TICKERS_FILE, "r", encoding="utf-8") as f:
    tickers = [line.strip() for line in f if line.strip()]

if not tickers:
    print(f"No tickers found in {TICKERS_FILE}", file=sys.stderr)
    sys.exit(1)

# ----------------------- main loop ------------------------------------------
for ticker in tickers:
    print(f"\n→ Processing {ticker}")
    all_dfs = []

    for start, end in date_chunks:
        print(f"   ↳ Chunk: {start.isoformat()} → {end.isoformat()}", end=" … ")
        try:
            df_chunk = yf.Ticker(ticker).history(
                start=start.isoformat(),
                end=end.isoformat(),
                interval="1d",
                auto_adjust=False,
                timeout=REQUEST_TIMEOUT
            )
            if df_chunk is None or df_chunk.empty:
                print("no data")
            else:
                print(f"got {len(df_chunk)} rows")
                all_dfs.append(df_chunk)
        except Exception as e:
            print(f"error ({e})")
        time.sleep(PAUSE_BETWEEN_CHUNKS)

    if not all_dfs:
        print(f"   ⚠️  No data for {ticker}, skipping.")
        continue

    # concatenate, drop duplicates, sort
    df = pd.concat(all_dfs)
    df = df.reset_index().drop_duplicates(subset="Date").set_index("Date").sort_index()

    # build JSON records
    quotes = []
    for ts, row in df.iterrows():
        native_dt   = dt.datetime.combine(ts.date(), dt.time(9, 30), tzinfo=TZ_NATIVE)
        unix_ts     = int(native_dt.timestamp())
        native_str  = native_dt.isoformat(timespec="minutes") + "[US/Eastern]"
        my_str      = native_dt.astimezone(TZ_MY).isoformat(timespec="minutes") + "[Europe/Berlin]"

        quotes.append({
            "high"       : round(float(row["High"]), 2),
            "open"       : round(float(row["Open"]), 2),
            "close"      : round(float(row["Close"]), 2),
            "low"        : round(float(row["Low"]), 2),
            "volume"     : float(row["Volume"]),
            "timestamp"  : unix_ts,
            "nativeDate" : native_str,
            "myDate"     : my_str
        })

    # save out
    output_path = f"{OUTPUT_DIR}/{ticker}_quotes.txt"
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(quotes, out_f, ensure_ascii=False)
    print(f"   ✔ Saved {len(quotes)} bars to {output_path}")

    # polite delay before next ticker
    time.sleep(DELAY_BETWEEN)

print("\nAll done.")