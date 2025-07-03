import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from multiprocessing import Pool

from indicators.rsi import calculate_rsi
from indicators.macd import calculate_macd
from strategy.decision_engine import calculate_decision_score, calculate_sell_score, alim_karari_ver, alim_karari_ver_agirlikli, macro_score, price_volume_signal, karar_skora_cevir, vix_score, putcall_score, adline_score, m2_score, pmi_score, hy_spread_score, yield_curve_score, fund_flows_score, advanced_karar_skora_cevir, calculate_atr_stop_levels
from portfolio.portfolio_manager import simulate_buy_sell, uygula_alim_karari, auto_rebalance_portfolio, plot_portfolio_risk
from strategy.learning import update_weights_after_result, log_learning_data, analyze_learning_log_and_update_weights, gosterge_agirliklari_ogren
from utils.fetch_price import fetch_multiple_stock_data, create_db_schema, save_trade, save_indicator, save_portfolio_snapshot
from ta.momentum import RSIIndicator
from ta.trend import MACD
from strategy.score_weights import update_global_weights_from_log, rolling_weight_update
from yatirim_botu.core.indicators import add_atr

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

# Simülasyon başında veritabanı şemasını oluştur
create_db_schema()

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
    # Sadece son 20 adımı (yaklaşık 1 ay) göster
    plot_range = 20 if len(portfolio_value) > 20 else len(portfolio_value)
    plt.plot(range(len(portfolio_value)-plot_range, len(portfolio_value)), portfolio_value[-plot_range:])
    plt.title("Portföy Değeri (Son 1 Ay)")
    plt.xlabel("Gün")
    plt.ylabel("USD")
    plt.grid()
    plt.show(block=True)
    # Son 1 ayın işlemlerini özetle
    print("\n--- Son 1 Ayın İşlem Özeti ---")
    for log in trade_log[-20:]:
        print(log)
else:
    print("Hiç portföy verisi oluşmadı!")

print("Simülasyon bitti.")

# Öğrenme logunu analiz et ve ağırlıkları güncelle
analyze_learning_log_and_update_weights()

# Çoklu NASDAQ hisse listesi
nasdaq_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'INTC', 'AMD', 'NFLX'
]

# Kullanıcıdan veya koddan tarih aralığı belirle
sim_start = datetime.datetime(2015, 1, 1)
sim_end = datetime.datetime(2018, 12, 31)

# Tüm hisseler için veri çek (tarih aralığına göre)
all_data = fetch_multiple_stock_data(nasdaq_tickers, start_date=sim_start, end_date=sim_end)
# Tüm DataFrame'leri tekrar kırpmaya gerek yok, çünkü fetch fonksiyonu zaten bu aralığı getiriyor.

# ATR'yi her hisseye ekle
for ticker in all_data:
    all_data[ticker] = add_atr(all_data[ticker])

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
portfolio = {ticker: {'stock': 0, 'buy_price': 0.0} for ticker in nasdaq_tickers}
trade_log = []
portfolio_value = []
agirliklar = {
    "RSI": 0.1, "MACD": 0.1, "MA": 0.08, "SuperTrend": 0.08, "Keltner": 0.05, "Bollinger": 0.05, "Fibo": 0.05, "PriceVol": 0.08, "Macro": 0.08,
    "VIX": 0.05, "PutCall": 0.05, "ADLine": 0.05, "M2": 0.05, "PMI": 0.05, "HYSpread": 0.05, "YieldCurve": 0.05, "FundFlows": 0.05
}

# Ortak tarih aralığını bul
common_index = all_data[nasdaq_tickers[0]].index
for ticker in nasdaq_tickers[1:]:
    common_index = common_index.intersection(all_data[ticker].index)
# Sadece 2015-2018 arası tarihleri al
common_index = common_index[(common_index >= '2015-01-01') & (common_index <= '2018-12-31')]
common_index = common_index[200::5]  # 200 sonrası, 5 günde bir

portfolio_value = []
date_list = []

indicator_columns = ['RSI', 'MACD', 'MA']  # Gerekirse diğer ana göstergeleri de ekleyin

for idx, date in enumerate(common_index):
    total_value = cash
    ticker_scores = []
    for ticker in nasdaq_tickers:
        df = all_data[ticker]
        if date not in df.index:
            continue
        row = df.loc[date]
        price = float(row['Close'])
        ma50 = float(row['MA50'])
        ma200 = float(row['MA200'])
        rsi = float(row['RSI'])
        macd = float(row['MACD'])
        signal = float(row['Signal'])
        stock = portfolio[ticker]['stock']
        buy_price = portfolio[ticker]['buy_price']
        bollinger_high = df['Close'].rolling(window=20).mean().loc[date] + 2 * df['Close'].rolling(window=20).std().loc[date]
        bollinger_low = df['Close'].rolling(window=20).mean().loc[date] - 2 * df['Close'].rolling(window=20).std().loc[date]
        volume = float(row['Volume'])
        prev_idx = df.index.get_loc(date) - 1
        prev_volume = float(df['Volume'].iloc[prev_idx]) if prev_idx >= 0 else volume
        skor = advanced_karar_skora_cevir(
            rsi, macd, signal, ma50, ma200, price, bollinger_high, bollinger_low, volume, prev_volume, agirliklar
        )
        ticker_scores.append((ticker, skor, price, ma50, ma200, rsi, macd, signal, stock, buy_price, row))
    ticker_scores.sort(key=lambda x: x[1], reverse=True)
    # Dinamik eşik: önce 0.3, yoksa 0.15, yine yoksa en yüksek skorlu hisse
    top_to_buy = [t for t in ticker_scores if t[1] > 0.3][:2]
    threshold_used = 0.3
    if not top_to_buy:
        top_to_buy = [t for t in ticker_scores if t[1] > 0.15][:2]
        threshold_used = 0.15
    if not top_to_buy and ticker_scores:
        top_to_buy = [ticker_scores[0]]
        threshold_used = 'max'
    print(f"Hafta {idx}: Kullanılan eşik: {threshold_used}, Seçilen hisseler: {[t[0] for t in top_to_buy]}")
    for ticker, skor, price, ma50, ma200, rsi, macd, signal, stock, buy_price, row in ticker_scores:
        atr = row['ATR'] if 'ATR' in row else 0
        alim_skoru, confidence = alim_karari_ver_agirlikli(rsi, macd, signal, ma50, ma200, agirliklar)
        # Göstergeleri kaydet
        save_indicator(ticker, 'RSI', rsi, date)
        save_indicator(ticker, 'MACD', macd, date)
        save_indicator(ticker, 'MA50', ma50, date)
        save_indicator(ticker, 'MA200', ma200, date)
        # ... diğer göstergeler eklenebilir
        if (ticker, skor, price, ma50, ma200, rsi, macd, signal, stock, buy_price, row) in top_to_buy and cash >= price and confidence > 0.5:
            miktar = 1
            if 0.5 < confidence < 0.7:
                miktar = max(1, int(0.5))
            cash, stock, miktar = uygula_alim_karari(True, alim_skoru, cash, price, int(portfolio[ticker]['stock']))
            if miktar > 0:
                buy_price = float(price)
                portfolio[ticker]['stock'] = int(stock)
                portfolio[ticker]['buy_price'] = float(buy_price)
                stop_loss, take_profit = calculate_atr_stop_levels(buy_price, atr)
                trade_log_entry = {'date': row.name, 'ticker': ticker, 'action': 'BUY', 'price': price, 'amount': int(miktar), 'alim_skoru_agirlikli': alim_skoru, 'confidence': confidence, 'stop_loss': stop_loss, 'take_profit': take_profit, 'agirliklar': agirliklar.copy(), 'indicator': 'RSI', 'profit': 0}
                trade_log.append(trade_log_entry)
                # Trade'i veritabanına kaydet
                save_trade(ticker, 'BUY', price, int(miktar), '', 0, str(row.name))
                print(f"{row.name.date()} {ticker} ALIM: {price} x {miktar} (Conf: {confidence}) SL: {stop_loss:.2f} TP: {take_profit:.2f}")
                if idx + 1 < len(df):
                    sonraki_fiyat = df.iloc[idx + 1]['Close']
                    ma_durum = int(ma50 > ma200)
                    log_learning_data(row.name.date(), f"ALIM-{ticker}", price, rsi, macd - signal, ma_durum, sonraki_fiyat)
        elif (sell_score >= 2 or current_loss >= 0.05) and stock > 0:
            cash += price * stock
            result = "gain" if price > buy_price else "loss"
            update_weights_after_result(result, indicators_used=["RSI", "MACD", "MA200"])
            profit = (price - buy_price) * stock
            trade_log.append({'date': row.name, 'ticker': ticker, 'action': 'SELL', 'price': price, 'result': result, 'sell_score': sell_score, 'amount': int(stock), 'indicator': 'RSI', 'profit': profit})
            print(f"{row.name.date()} {ticker} SATIŞ: {price} x {stock} → {'KAR' if result == 'gain' else 'ZARAR'}")
            portfolio[ticker]['stock'] = int(0)
            portfolio[ticker]['buy_price'] = float(0.0)
    date_list.append(date)
    portfolio_value.append(total_value)
    # Her 30 işlemde bir klasik ağırlık güncelleme
    if len(trade_log) > 0 and len(trade_log) % 30 == 0:
        log_df = pd.DataFrame(trade_log)
        agirliklar = update_global_weights_from_log(log_df, agirliklar)
        print(f"Ağırlıklar güncellendi: {agirliklar}")
    # Her 60 işlemde bir makine öğrenmesi ile ağırlık güncelleme
    if len(trade_log) > 0 and len(trade_log) % 60 == 0:
        log_df = pd.DataFrame(trade_log)
        # Başarı sütunu yoksa ekle (örnek: kar > 0 ise başarılı)
        if 'success' not in log_df.columns:
            log_df['success'] = log_df['profit'] > 0
        new_weights = rolling_weight_update(log_df, indicator_columns, window=60)
        if new_weights:
            agirliklar.update(new_weights)
            print(f"Makine öğrenmesi ile ağırlıklar güncellendi: {agirliklar}")
    # Her 20 adımda (haftada bir) otomatik rebalancing uygula
    if idx > 0 and idx % 20 == 0:
        # Fiyat ve performans verilerini hazırla
        price_data = {t: all_data[t].loc[date]['Close'] for t in nasdaq_tickers if date in all_data[t].index}
        # Basit performans: ilk fiyata göre getiri
        performance_data = {t: (all_data[t].loc[date]['Close'] / all_data[t]['Close'].iloc[0]) - 1 for t in nasdaq_tickers if date in all_data[t].index}
        portfolio = auto_rebalance_portfolio(portfolio, price_data, performance_data)
        print(f"Rebalancing sonrası portföy: { {k: v['stock'] for k, v in portfolio.items()} }")
    # Portföy snapshot'ı kaydet
    total_value = cash + sum(portfolio[t]['stock'] * all_data[t].loc[date]['Close'] for t in nasdaq_tickers if date in all_data[t].index)
    save_portfolio_snapshot(str(date), total_value, cash, str({k: v['stock'] for k, v in portfolio.items()}))

# Simülasyon sonunda risk dağılımı görselleştir
price_data = {t: all_data[t]['Close'].iloc[-1] for t in nasdaq_tickers}
plot_portfolio_risk(portfolio, price_data)

# Tüm simülasyonun grafiği (tarih ekseninde)
if portfolio_value:
    plt.figure(figsize=(12,6))
    plt.plot(date_list, portfolio_value)
    plt.title(f"Portföy Değeri ({sim_start.date()} - {sim_end.date()}) | Final: {portfolio_value[-1]:.2f} USD")
    plt.xlabel("Tarih")
    plt.ylabel("USD")
    plt.grid()
    plt.tight_layout()
    plt.show(block=True)
    print(f"\nSimülasyon Tarih Aralığı: {sim_start.date()} - {sim_end.date()}")
    print(f"Final Portföy Değeri: {portfolio_value[-1]:.2f} USD")
else:
    print("Hiç portföy verisi oluşmadı!")

def single_stock_backtest(ticker):
    df = all_data[ticker]
    # Tüm gösterge ve portföy hazırlıklarını burada yap
    # (Kısa örnek: gerçek kodda tüm simülasyon mantığını buraya taşıyın)
    # ...
    # Simülasyon döngüsü (örnek)
    cash = 100000
    stock = 0
    buy_price = 0
    trade_log = []
    portfolio_value = []
    for i in range(200, len(df), 5):
        row = df.iloc[i]
        price = float(row['Close'])
        # ... (karar motoru, al/sat, portföy güncelleme)
        total_value = cash + stock * price
        portfolio_value.append(total_value)
    # Sonuçları döndür
    return {
        'ticker': ticker,
        'final_value': portfolio_value[-1] if portfolio_value else 0,
        'trade_log': trade_log,
        'portfolio_value': portfolio_value
    }

def parallel_backtest(stock_list):
    with Pool(processes=4) as pool:
        results = pool.map(single_stock_backtest, stock_list)
    return results

# --- Ana simülasyonun sonunda ---
if __name__ == "__main__":
    results = parallel_backtest(nasdaq_tickers)
    for res in results:
        print(f"{res['ticker']} final portföy değeri: {res['final_value']}") 