"""
Script: fetch_stock_prices.py
Description: Reads a list of stock tickers from file.txt and fetches the latest market price for each using Yahoo Finance,
with a 3-second pause between each API request to avoid rate-limiting.
"""
import time
import sys
import yfinance as yf


def get_latest_price(ticker: str) -> float:
    """Fetch the latest market price for a given ticker symbol."""
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        price = info.get("regularMarketPrice")
        if price is None:
            hist = ticker_obj.history(period="1d")
            if not hist.empty:
                price = hist["Close"][0]
        return price
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}", file=sys.stderr)
        return None


def main():
    # Read tickers from file.txt (one ticker per line)
    try:
        with open('sp500.txt', 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
    except FileNotFoundError:
        print("file.txt not found. Please ensure the file exists in the current directory.")
        sys.exit(1)

    if not tickers:
        print("No tickers found in file.txt.")
        sys.exit(1)

    # Fetch and print prices with a pause between requests
    for ticker in tickers:
        price = get_latest_price(ticker)
        if price is not None:
            print(f"{ticker}: {price}")
        else:
            print(f"{ticker}: Price not available")
        # Pause to avoid hitting request limits
        time.sleep(1)


if __name__ == '__main__':
    main()