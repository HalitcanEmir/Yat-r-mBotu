import yfinance as yf

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