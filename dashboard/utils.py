import yfinance as yf
import pandas as pd
from .models import Portfolio, Trade

def get_bist_price(symbol):
    """
    BIST sembolü (ör: 'THYAO') için yfinance üzerinden anlık fiyatı döndürür.
    Hata olursa None döner.
    """
    try:
        yf_symbol = f"{symbol}.IS"
        ticker = yf.Ticker(yf_symbol)
        data = ticker.history(period="1d")
        if not data.empty:
            return float(data['Close'].iloc[-1])
        else:
            return None
    except Exception as e:
        print(f"Fiyat çekme hatası: {symbol} - {e}")
        return None

# BIST'in en büyük 20 şirketi (örnek semboller)
BIST_TOP20 = [
    'AKBNK', 'THYAO', 'SISE', 'KCHOL', 'BIMAS', 'EREGL', 'GARAN', 'TUPRS', 'ASELS', 'PETKM',
    'ARCLK', 'TCELL', 'VESTL', 'ENKAI', 'TOASO', 'FROTO', 'KOZAL', 'YKBNK', 'ISCTR', 'SAHOL'
]

def get_rsi(prices, period=14):
    """
    Basit RSI hesaplama (pandas Series ile)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# BIST dışı hisseleri portföyden ve işlem geçmişinden silen fonksiyon
def clean_non_bist_portfolio_and_trades():
    Portfolio.objects.exclude(symbol__in=BIST_TOP20).delete()
    Trade.objects.exclude(symbol__in=BIST_TOP20).delete()
    print("BIST dışı hisseler portföyden ve işlem geçmişinden silindi.") 