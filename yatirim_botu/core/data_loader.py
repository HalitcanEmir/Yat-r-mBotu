"""
Hisse verisi çekme ve temel teknik göstergeleri hesaplama modülü.
"""
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from core import indicators

def get_stock_data(ticker: str, years: int = 10) -> pd.DataFrame:
    """
    Belirtilen NASDAQ hissesinin son yıllık verisini ve göstergelerini döner.
    """
    import datetime
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=365 * years)

    df = yf.download(ticker, start=start_date, end=end_date, interval="1d")

    # DEBUG: Veri yapısını göster
    print("df.columns:", df.columns)
    print(df.head())

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"Veri çekilemedi veya {ticker} için veri bulunamadı.")

    # Sütun isimlerini normalize et (MultiIndex ise ilk seviye, değilse önceki gibi)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].capitalize() for col in df.columns]
    else:
        def normalize_col(col):
            if isinstance(col, tuple):
                return str(col[-1]).strip().capitalize()
            return str(col).strip().capitalize()
        df.columns = [normalize_col(col) for col in df.columns]

    # Sadece tekil sütun (Series) olarak 'Close' al, yoksa 'Adj Close' dene
    close_col = 'Close' if 'Close' in df.columns else ('Adj close' if 'Adj close' in df.columns else None)
    if close_col is None:
        raise ValueError("DataFrame'de 'Close' veya 'Adj Close' sütunu yok!")
    close = df[close_col]
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    # Temel Göstergeler:
    df['Rsi'] = RSIIndicator(close, window=14).rsi()
    macd = MACD(close)
    df['Macd'] = macd.macd()
    df['Macd_signal'] = macd.macd_signal()
    df['Ma50'] = close.rolling(window=50).mean()
    df['Ma200'] = close.rolling(window=200).mean()

    # Gelişmiş Göstergeler:
    df = indicators.add_supertrend(df)
    df = indicators.add_keltner_channels(df)
    df = indicators.add_bollinger_bands(df)
    df = indicators.add_atr(df)
    df = indicators.add_obv(df)

    # Temizlik
    df.dropna(inplace=True)

    return df 