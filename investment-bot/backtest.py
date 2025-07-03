import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from indicators.rsi import calculate_rsi
from indicators.macd import calculate_macd
from strategy.decision_engine import calculate_decision_score, calculate_sell_score
from portfolio.portfolio_manager import simulate_buy_sell
from strategy.learning import update_weights_after_result

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

for i in range(200, len(df)):
    row = df.iloc[i]
    indicators = {
        "RSI": float(row["RSI"]),
        "MACD": float(row["MACD"] - row["Signal"])
    }
    score = calculate_decision_score(indicators, {"RSI": 1, "MACD": 1}, news_score=0)
    price = float(row["Close"])
    date = row.name
    ma200 = float(row["MA200"])
    m2_growth = float(row["M2_GROWTH"])
    fed_rate = float(row["FED_RATE"])
    prev_fed_rate = float(row["PREV_FED_RATE"])

    # Satış skoru (profesyonel mantık)
    sell_score = calculate_sell_score(
        float(row["RSI"]), float(row["MACD"]), float(row["Signal"]), price, ma200, m2_growth, fed_rate, prev_fed_rate
    )
    current_loss = (buy_price - price) / buy_price if stock > 0 else 0

    print(f"{date.date()} | Score: {score:.2f} | SellScore: {sell_score} | Stock: {stock} | Cash: {cash}")

    # AL Kararı (örnek, basit)
    if score > 0.7 and cash >= price:
        stock += 1
        cash -= price
        buy_price = price
        trade_log.append({"date": date, "action": "BUY", "price": price})
        print(f"{date.date()} ALINDI: {price}")

    # SAT Kararı (profesyonel mantık)
    elif (sell_score >= 3 or (sell_score == 2 and current_loss > 0) or current_loss >= 0.10) and stock > 0:
        cash += price * stock
        result = "gain" if price > buy_price else "loss"
        update_weights_after_result(result, indicators_used=["RSI", "MACD", "MA200"])
        trade_log.append({"date": date, "action": "SELL", "price": price, "result": result, "sell_score": sell_score})
        print(f"{date.date()} SATILDI: {price} → {'KAR' if result == 'gain' else 'ZARAR'} (SellScore: {sell_score})")
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