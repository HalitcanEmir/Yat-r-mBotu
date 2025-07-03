import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from indicators.rsi import calculate_rsi
from indicators.macd import calculate_macd
from strategy.decision_engine import calculate_decision_score, calculate_sell_score, alim_karari_ver, alim_karari_ver_agirlikli
from portfolio.portfolio_manager import simulate_buy_sell, uygula_alim_karari
from strategy.learning import update_weights_after_result, log_learning_data, analyze_learning_log_and_update_weights, gosterge_agirliklari_ogren
from utils.fetch_price import fetch_multiple_stock_data
from ta.momentum import RSIIndicator
from ta.trend import MACD

print("AAPL verisi çekiliyor...")
df = yf.download("AAPL", start="2020-01-01", end="2024-12-31")
if df is None or df.empty:
    print("Veri çekilemedi veya boş. Çıkılıyor.")
    exit()
print(df.head())
print("Veri satırı sayısı:", len(df))

df["RSI"] = calculate_rsi(df["Close"])
df["MACD"], df["Signal"] = calculate_macd(df["Close"])
df["MA200"] = df["Close"].rolling(window=200).mean()

# Mock macro data for demonstration (in real use, pull from macro_signals)
df["M2_GROWTH"] = 0.01  # Yıllık para arzı büyüme oranı (örnek)
df["FED_RATE"] = 5.0    # Güncel faiz (örnek)
df["PREV_FED_RATE"] = 4.75  # Önceki faiz (örnek)

portfolio_value = []
cash = 5000
stock = 0
buy_price = 0
trade_log = []

print("Simülasyon başlıyor, toplam gün:", len(df))

# Simülasyon öncesi ağırlıkları öğren
agirliklar = gosterge_agirliklari_ogren()

# Haftalık simülasyon: her 5 günde bir işlem yap
for i in range(200, len(df), 5):
    row = df.iloc[i]
    indicators = {
        "RSI": float(row["RSI"]),
        "MACD": float(row["MACD"] - row["Signal"])
    }
    score = calculate_decision_score(indicators, {"RSI": 1, "MACD": 1}, news_score=0)
    price = float(row["Close"])
    date = row.name
    ma200 = float(row["MA200"])
    ma50 = float(df.iloc[i]["Close"]) if i >= 50 else price  # fallback
    m2_growth = float(row["M2_GROWTH"])
    fed_rate = float(row["FED_RATE"])
    prev_fed_rate = float(row["PREV_FED_RATE"])

    # Teknik kırılım ve destek/direnç (örnek, gerçek fonksiyonlarla değiştirilebilir)
    direnc_kirildi = price > ma200 * 1.01
    destekten_sekti = price < ma200 * 0.99
    para_arzi_artiyor = m2_growth > 0
    faiz_dusuyor = fed_rate < prev_fed_rate

    # Ağırlıklı alım skoru ve kararı
    alim_skoru_agirlikli = alim_karari_ver_agirlikli(
        float(row["RSI"]), float(row["MACD"]), float(row["Signal"]), ma50, ma200, agirliklar
    )
    alim_var = alim_skoru_agirlikli > 0.5  # Eşik: 0.5 (isteğe göre ayarlanabilir)

    # Satış skoru (profesyonel mantık)
    sell_score = calculate_sell_score(
        float(row["RSI"]), float(row["MACD"]), float(row["Signal"]), price, ma200, m2_growth, fed_rate, prev_fed_rate
    )
    current_loss = (buy_price - price) / buy_price if stock > 0 else 0

    print(f"{date.date()} | Score: {score:.2f} | SellScore: {sell_score} | AlimSkoruAgirlikli: {alim_skoru_agirlikli:.2f} | Agirliklar: {agirliklar} | Stock: {stock} | Cash: {cash}")

    # AL Kararı (ağırlıklı, öğrenen)
    if alim_var and cash >= price:
        cash, stock, miktar = uygula_alim_karari(alim_var, alim_skoru_agirlikli, cash, price, int(stock))
        if miktar > 0:
            buy_price = float(price)
            trade_log.append({"date": date, "action": "BUY", "price": price, "amount": miktar, "alim_skoru_agirlikli": alim_skoru_agirlikli, "agirliklar": agirliklar})
            print(f"{date.date()} ALIM: {price} x {miktar} (AlimSkoruAgirlikli: {alim_skoru_agirlikli:.2f})")
            # Learning log: 5 gün sonra fiyatı kaydet
            if i + 5 < len(df):
                sonraki_fiyat = df.iloc[i + 5]["Close"]
                ma_durum = int(ma50 > ma200)
                log_learning_data(date.date(), "ALIM", price, row["RSI"], row["MACD"] - row["Signal"], ma_durum, sonraki_fiyat)

    # SAT Kararı (profesyonel mantık)
    elif (sell_score >= 3 or (sell_score == 2 and current_loss > 0) or current_loss >= 0.10) and stock > 0:
        cash += price * stock
        result = "gain" if price > buy_price else "loss"
        update_weights_after_result(result, indicators_used=["RSI", "MACD", "MA200"])
        trade_log.append({"date": date, "action": "SELL", "price": price, "result": result, "sell_score": sell_score, "amount": stock})
        print(f"{date.date()} SATIŞ: {price} x {stock} → {'KAR' if result == 'gain' else 'ZARAR'} (SellScore: {sell_score})")
        stock = 0

    total_value = cash + stock * price
    portfolio_value.append(total_value)

if portfolio_value:
    plt.figure(figsize=(10,5))
    plt.plot(portfolio_value)
    plt.title("Portföy Değeri (AAPL Simülasyonu)")
    plt.xlabel("Gün")
    plt.ylabel("USD")
    plt.grid()
    plt.show(block=True)
else:
    print("Hiç portföy verisi oluşmadı!")

print("Simülasyon bitti.")

# Öğrenme logunu analiz et ve ağırlıkları güncelle
analyze_learning_log_and_update_weights()

# Çoklu NASDAQ hisse listesi
nasdaq_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'INTC', 'AMD', 'NFLX'
]

# Tüm hisseler için veri çek
all_data = fetch_multiple_stock_data(nasdaq_tickers, years=4)

# Her hisse için göstergeleri hesapla (örnek: sadece MA50/MA200, RSI, MACD)
def add_indicators(df):
    close = df['Close']
    df['RSI'] = RSIIndicator(close, window=14).rsi()
    macd = MACD(close)
    df['MACD'] = macd.macd()
    df['Signal'] = macd.macd_signal()
    df['MA50'] = close.rolling(window=50).mean()
    df['MA200'] = close.rolling(window=200).mean()
    df.dropna(inplace=True)
    return df

for ticker in all_data:
    all_data[ticker] = add_indicators(all_data[ticker])

# Portföy ve nakit yönetimi
cash = 100000
portfolio = {ticker: {'stock': 0, 'buy_price': 0} for ticker in nasdaq_tickers}
trade_log = []
portfolio_value = []

# Haftalık simülasyon: her 5 günde bir
max_len = min([len(all_data[ticker]) for ticker in all_data])
agirliklar = gosterge_agirliklari_ogren()

for i in range(200, max_len, 5):
    total_value = cash
    # --- 1. Her hisse için skorları hesapla ---
    ticker_scores = []
    for ticker in nasdaq_tickers:
        df = all_data[ticker]
        row = df.iloc[i]
        price = float(row['Close'])
        ma50 = float(row['MA50'])
        ma200 = float(row['MA200'])
        rsi = float(row['RSI'])
        macd = float(row['MACD'])
        signal = float(row['Signal'])
        stock = portfolio[ticker]['stock']
        buy_price = portfolio[ticker]['buy_price']

        # Ağırlıklı alım skoru
        alim_skoru_agirlikli = alim_karari_ver_agirlikli(rsi, macd, signal, ma50, ma200, agirliklar)
        ticker_scores.append((ticker, alim_skoru_agirlikli, price, ma50, ma200, rsi, macd, signal, stock, buy_price, row))

    # --- 2. Sadece en yüksek skoru alan 2 hisseyi seç ve al ---
    ticker_scores.sort(key=lambda x: x[1], reverse=True)
    top_to_buy = [t for t in ticker_scores if t[1] > 0.3][:2]

    for ticker, alim_skoru_agirlikli, price, ma50, ma200, rsi, macd, signal, stock, buy_price, row in ticker_scores:
        # Satış skoru
        sell_score = calculate_sell_score(rsi, macd, signal, price, ma200, 0, 0, 0)
        current_loss = (buy_price - price) / buy_price if stock > 0 else 0

        # Detaylı log
        print(f"{row.name.date()} {ticker} | AlimSkoru: {alim_skoru_agirlikli:.2f} | SatScore: {sell_score} | Stock: {stock} | Cash: {cash}")

        # AL Kararı: Sadece top_to_buy listesinde ve nakit varsa
        if (ticker, alim_skoru_agirlikli, price, ma50, ma200, rsi, macd, signal, stock, buy_price, row) in top_to_buy and cash >= price:
            cash, stock, miktar = uygula_alim_karari(True, alim_skoru_agirlikli, cash, price, int(portfolio[ticker]['stock']))
            if miktar > 0:
                buy_price = float(price)
                portfolio[ticker]['stock'] = int(stock)
                portfolio[ticker]['buy_price'] = float(buy_price)
                trade_log.append({'date': row.name, 'ticker': ticker, 'action': 'BUY', 'price': price, 'amount': int(miktar), 'alim_skoru_agirlikli': alim_skoru_agirlikli, 'agirliklar': agirliklar})
                print(f"{row.name.date()} {ticker} ALIM: {price} x {miktar} (AlimSkoruAgirlikli: {alim_skoru_agirlikli:.2f})")
                # Learning log: 5 gün sonra fiyatı kaydet
                if i + 5 < len(df):
                    sonraki_fiyat = df.iloc[i + 5]['Close']
                    ma_durum = int(ma50 > ma200)
                    log_learning_data(row.name.date(), f"ALIM-{ticker}", price, rsi, macd - signal, ma_durum, sonraki_fiyat)
        # SAT Kararı (daha esnek)
        elif (sell_score >= 2 or current_loss >= 0.05) and stock > 0:
            cash += price * stock
            result = "gain" if price > buy_price else "loss"
            update_weights_after_result(result, indicators_used=["RSI", "MACD", "MA200"])
            trade_log.append({'date': row.name, 'ticker': ticker, 'action': 'SELL', 'price': price, 'result': result, 'sell_score': sell_score, 'amount': int(stock)})
            print(f"{row.name.date()} {ticker} SATIŞ: {price} x {stock} → {'KAR' if result == 'gain' else 'ZARAR'} (SellScore: {sell_score})")
            portfolio[ticker]['stock'] = int(0)
            portfolio[ticker]['buy_price'] = float(0.0)

        total_value += stock * price
    portfolio_value.append(total_value)

    # --- 3. Her hafta ağırlıkları güncelle ---
    agirliklar = gosterge_agirliklari_ogren()

if portfolio_value:
    plt.figure(figsize=(10,5))
    plt.plot(portfolio_value)
    plt.title("Portföy Değeri (Çoklu NASDAQ Simülasyonu)")
    plt.xlabel("Gün")
    plt.ylabel("USD")
    plt.grid()
    plt.show(block=True)
else:
    print("Hiç portföy verisi oluşmadı!")

print("Simülasyon bitti.")

# Öğrenme logunu analiz et ve ağırlıkları güncelle
analyze_learning_log_and_update_weights() 