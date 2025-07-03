# Price fetching utility implementation will go here 

import yfinance as yf
import pandas as pd

def fetch_multiple_stock_data(tickers, years=10):
    """
    Birden fazla NASDAQ hissesinin fiyat verisini ve göstergelerini döndürür.
    Dönüş: {ticker: DataFrame}
    """
    import datetime
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=365 * years)
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
            if not isinstance(df, pd.DataFrame) or df.empty:
                print(f"Veri çekilemedi: {ticker}")
                continue
            # Sütun isimlerini normalize et
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].capitalize() for col in df.columns]
            else:
                def normalize_col(col):
                    if isinstance(col, tuple):
                        return str(col[-1]).strip().capitalize()
                    return str(col).strip().capitalize()
                df.columns = [normalize_col(col) for col in df.columns]
            data[ticker] = df
        except Exception as e:
            print(f"{ticker} için hata: {e}")
    return data 