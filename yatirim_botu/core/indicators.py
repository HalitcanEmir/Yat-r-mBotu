# TODO: Teknik göstergeler (Keltner, SuperTrend, ATR, EMA, vb.) 

from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import ta
import numpy as np
import yfinance as yf
import pandas as pd

def add_supertrend(df, period=10, multiplier=3):
    """
    Supertrend hesaplar ve df'e sütun olarak ekler.
    """
    # Sütun kontrolü ve NaN temizliği
    for col in ['High', 'Low', 'Close']:
        if col not in df.columns:
            raise ValueError(f"DataFrame'de '{col}' sütunu yok!")
        if df[col].isnull().all().item():
            raise ValueError(f"DataFrame'de '{col}' sütunu tamamen boş!")
    df = df.dropna(subset=['High', 'Low', 'Close'])

    hl2 = (df['High'] + df['Low']) / 2
    atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=period).average_true_range()

    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    supertrend = [True]  # True = bullish, False = bearish

    for i in range(1, len(df)):
        if df['Close'].iloc[i] > upperband.iloc[i - 1]:
            supertrend.append(True)
        elif df['Close'].iloc[i] < lowerband.iloc[i - 1]:
            supertrend.append(False)
        else:
            supertrend.append(supertrend[i - 1])

    df['SuperTrend'] = supertrend
    return df

def add_keltner_channels(df, ema_window=20, atr_multiplier=2):
    """
    Keltner Channels üst, alt ve merkez çizgileri.
    """
    ema = df['Close'].ewm(span=ema_window, adjust=False).mean()
    atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=ema_window).average_true_range()

    df['KC_middle'] = ema
    df['KC_upper'] = ema + atr * atr_multiplier
    df['KC_lower'] = ema - atr * atr_multiplier

    return df

def add_bollinger_bands(df, window=20, std_dev=2):
    """
    Bollinger üst-alt bantlarını ekler.
    """
    rolling_mean = df['Close'].rolling(window).mean()
    rolling_std = df['Close'].rolling(window).std()

    df['BB_upper'] = rolling_mean + (rolling_std * std_dev)
    df['BB_lower'] = rolling_mean - (rolling_std * std_dev)

    return df

def add_atr(df, window=14):
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=window).average_true_range()
    return df

def add_obv(df):
    df['OBV'] = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    return df

def fetch_ema(symbol, interval="1d", ema_period=50):
    data = yf.download(symbol, period="6mo", interval=interval)
    # Sütun isimlerini normalize et (MultiIndex olasılığına karşı)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0].capitalize() for col in data.columns]
    else:
        data.columns = [str(col).strip().capitalize() for col in data.columns]
    data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False).mean()
    return data

def ema_trend_confirmation(symbol):
    daily = fetch_ema(symbol, interval="1d", ema_period=50)
    weekly = fetch_ema(symbol, interval="1wk", ema_period=50)

    daily_trend_up = daily['Close'].iloc[-1] > daily['EMA'].iloc[-1]
    weekly_trend_up = weekly['Close'].iloc[-1] > weekly['EMA'].iloc[-1]

    if daily_trend_up and weekly_trend_up:
        return 1  # Güçlü AL sinyali (hem kısa hem uzun vadeli yukarı)
    elif daily_trend_up and not weekly_trend_up:
        return 0  # Kısa vadeli yükseliş ama uzun vadeli değil
    elif not daily_trend_up and weekly_trend_up:
        return 0  # Kararsız durum
    else:
        return -1  # İkisi de düşüşte → Sat sinyali 