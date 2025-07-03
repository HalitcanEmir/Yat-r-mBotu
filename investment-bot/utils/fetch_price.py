# Price fetching utility implementation will go here 

import yfinance as yf
import pandas as pd
import os
import time
import sqlite3
from datetime import datetime

DB_PATH = "yatirim_botu/data/bot.db"

def fetch_multiple_stock_data(tickers, start_date=None, end_date=None):
    """
    Birden fazla NASDAQ hissesinin fiyat verisini ve göstergelerini döndürür.
    Caching: data/cache/{ticker}_{start}_{end}.parquet
    Dönüş: {ticker: DataFrame}
    """
    if end_date is None:
        end_date = datetime.today()
    if start_date is None:
        start_date = end_date - datetime.timedelta(days=365 * 10)
    data = {}
    cache_dir = os.path.join(os.path.dirname(__file__), '../data/cache')
    os.makedirs(cache_dir, exist_ok=True)
    for ticker in tickers:
        cache_file = os.path.join(cache_dir, f"{ticker}_{start_date.date()}_{end_date.date()}.parquet")
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                data[ticker] = df
                continue
            except Exception as e:
                print(f"Cache okunamadı, yeniden çekiliyor: {ticker} ({e})")
        # Retry mechanism for Yahoo API
        for attempt in range(3):
            try:
                df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
                if not isinstance(df, pd.DataFrame) or df.empty:
                    print(f"Veri çekilemedi: {ticker}")
                    break
                # Sütun isimlerini normalize et
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0].capitalize() for col in df.columns]
                else:
                    def normalize_col(col):
                        if isinstance(col, tuple):
                            return str(col[-1]).strip().capitalize()
                        return str(col).strip().capitalize()
                    df.columns = [normalize_col(col) for col in df.columns]
                df.to_parquet(cache_file)
                data[ticker] = df
                break
            except Exception as e:
                print(f"{ticker} için hata (deneme {attempt+1}): {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        else:
            print(f"{ticker} için veri alınamadı. Alternatif API TODO.")
            # TODO: Başarısız olursa alternatif API'den veri çek
    return data 

def create_db_schema(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT, action TEXT, price REAL, amount INTEGER, result TEXT, profit REAL, date TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS indicators (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT, indicator TEXT, value REAL, date TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT, total_value REAL, cash REAL, details TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS news (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT, news_text TEXT, analysis TEXT, sentiment REAL, created_at TEXT
    )''')
    conn.commit()
    conn.close()

def save_trade(ticker, action, price, amount, result, profit, date, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''INSERT INTO trades (ticker, action, price, amount, result, profit, date) VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (ticker, action, price, amount, result, profit, date))
    conn.commit()
    conn.close()

def save_indicator(ticker, indicator, value, date, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''INSERT INTO indicators (ticker, indicator, value, date) VALUES (?, ?, ?, ?)''',
              (ticker, indicator, value, date))
    conn.commit()
    conn.close()

def save_portfolio_snapshot(date, total_value, cash, details, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''INSERT INTO portfolio_snapshots (date, total_value, cash, details) VALUES (?, ?, ?, ?)''',
              (date, total_value, cash, details))
    conn.commit()
    conn.close() 