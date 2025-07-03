"""
Hisse verisi çekme ve temel teknik göstergeleri hesaplama modülü.
"""
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

def get_stock_data(ticker: str, years: int = 10) -> pd.DataFrame:
    """
    Belirtilen NASDAQ hissesinin son yıllık verisini ve göstergelerini döner.
    """
    import datetime
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=365 * years)

    df = yf.download(ticker, start=start_date, end=end_date, interval="1d")

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"Veri çekilemedi veya {ticker} için veri bulunamadı.")

    # Sadece tekil sütun (Series) olarak 'Close' al
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    # Göstergeler:
    df['RSI'] = RSIIndicator(close, window=14).rsi()

    macd = MACD(close)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    df['MA50'] = close.rolling(window=50).mean()
    df['MA200'] = close.rolling(window=200).mean()

    # Temizlik
    df.dropna(inplace=True)

    return df 