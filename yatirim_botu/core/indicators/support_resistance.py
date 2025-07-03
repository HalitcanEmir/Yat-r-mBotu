import yfinance as yf
import pandas as pd
import numpy as np

def get_support_resistance_levels(symbol):
    data = yf.download(symbol, period="6mo", interval="1d")
    # Sütun isimlerini normalize et (MultiIndex olasılığına karşı)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0].capitalize() for col in data.columns]
    else:
        data.columns = [str(col).strip().capitalize() for col in data.columns]
    data['min'] = data['Low'].rolling(window=5, center=True).min()
    data['max'] = data['High'].rolling(window=5, center=True).max()

    local_mins = data[(data['Low'] == data['min'])].tail(10)
    local_maxs = data[(data['High'] == data['max'])].tail(10)

    supports = local_mins['Low'].values
    resistances = local_maxs['High'].values

    return supports, resistances, data['Close'].iloc[-1]

def analyze_support_resistance(symbol):
    supports, resistances, current_price = get_support_resistance_levels(symbol)

    nearest_support = max([s for s in supports if s < current_price], default=None)
    nearest_resistance = min([r for r in resistances if r > current_price], default=None)

    if nearest_support and current_price < nearest_support * 0.98:
        return -1  # destek kırıldı → SAT
    elif nearest_resistance and current_price > nearest_resistance * 1.02:
        return 1   # direnç geçildi → AL
    else:
        return 0   # destek/direnç bölgesinde → kararsız 