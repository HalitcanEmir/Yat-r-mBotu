import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from indicators.rsi import calculate_rsi
from indicators.macd import calculate_macd
from strategy.decision_engine import calculate_decision_score, calculate_sell_score, alim_karari_ver, alim_karari_ver_agirlikli
from portfolio.portfolio_manager import uygula_alim_karari, plot_portfolio_risk
from strategy.learning import update_weights_after_result, log_learning_data, analyze_learning_log_and_update_weights, gosterge_agirliklari_ogren
from utils.fetch_price import fetch_multiple_stock_data, create_db_schema, save_trade, save_indicator, save_portfolio_snapshot
from ta.momentum import RSIIndicator
from ta.trend import MACD
from strategy.score_weights import update_global_weights_from_log, rolling_weight_update
from yatirim_botu.core.indicators import add_atr
import numpy as np
import json

# PyPortfolioOpt ile ilgili optimize_portfolio_with_mpt fonksiyonunu ve kullanılmayan tüm kodlarını kaldırıyorum.
# optimize_portfolio_with_mpt fonksiyonu ve PyPortfolioOpt ile ilgili kodlar kaldırıldı.

print("AAPL verisi çekiliyor...")
df = yf.download("AAPL", start="2020-01-01", end="2024-12-31")
if df is None or df.empty:
    print("Veri çekilemedi veya boş. Çıkılıyor.")
    exit()
# Sütunlar MultiIndex ise tek seviyeye indir
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
print("df.columns:", df.columns)
if df is None or df.empty:
    print("Veri çekilemedi veya boş. Çıkılıyor.")
    exit()
# Göstergeleri eklemeden önce indexi sıfırla
print("df.head() (göstergeler eklenmeden önce):\n", df.head(10))
df = df.reset_index()
print("df.tail() (göstergeler eklendikten sonra):\n", df.tail(10))
print("Veri satırı sayısı:", len(df))

# Eğer df["Close"] bir DataFrame ise, ilk sütunu al
if isinstance(df["Close"], pd.DataFrame):
    close = df["Close"].iloc[:, 0]
else:
    close = df["Close"]

df["RSI"] = calculate_rsi(close)
print("RSI örnek:", type(df["RSI"][:5]), df["RSI"][:5])
df["MACD"], df["Signal"] = calculate_macd(close)
print("MACD örnek:", type(df["MACD"][:5]), df["MACD"][:5])
print("Signal örnek:", type(df["Signal"][:5]), df["Signal"][:5])
df["MA200"] = df["Close"].rolling(window=200).mean()
print("MA200 örnek:", type(df["MA200"][:5]), df["MA200"][:5])

# Mock macro data for demonstration (in real use, pull from macro_signals)
df["M2_GROWTH"] = 0.01  # Yıllık para arzı büyüme oranı (örnek)
df["FED_RATE"] = 5.0    # Güncel faiz (örnek)
df["PREV_FED_RATE"] = 4.75  # Önceki faiz (örnek)

# Göstergelerin dolu olduğu satırları filtrele
indicator_cols = ["RSI", "MACD", "Signal", "MA200"]
df = df.dropna(subset=indicator_cols).reset_index()

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

# Simülasyon başında LSTM modeli eğit (veya yükle)
try:
    from ml_models import train_lstm_pytorch_price_forecast
    aapl_df = df.copy()
    lstm_model, _, _ = train_lstm_pytorch_price_forecast(aapl_df, plot=False)
except Exception:
    lstm_model = None

# Simülasyon başında Isolation Forest modeli eğit (veya yükle)
try:
    from ml_models import detect_anomalies_isolation_forest
    aapl_df = df.copy()
    isoforest_model, _ = detect_anomalies_isolation_forest(aapl_df, plot=False)
except Exception:
    isoforest_model = None

# Haftalık simülasyon: her 5 günde bir işlem yap
for i in range(0, len(df), 5):
    row = df.iloc[i]
    # Göstergeleri güvenli şekilde float olarak çek, NaN ise atla
    try:
        rsi = float(row["RSI"]) if not isinstance(row["RSI"], pd.Series) else float(row["RSI"].iloc[0])
        macd = float(row["MACD"]) if not isinstance(row["MACD"], pd.Series) else float(row["MACD"].iloc[0])
        signal = float(row["Signal"]) if not isinstance(row["Signal"], pd.Series) else float(row["Signal"].iloc[0])
        close_price = float(row["Close"]) if not isinstance(row["Close"], pd.Series) else float(row["Close"].iloc[0])
        ma200 = float(row["MA200"]) if not isinstance(row["MA200"], pd.Series) else float(row["MA200"].iloc[0])
        m2_growth = float(row["M2_GROWTH"]) if not isinstance(row["M2_GROWTH"], pd.Series) else float(row["M2_GROWTH"].iloc[0])
        fed_rate = float(row["FED_RATE"]) if not isinstance(row["FED_RATE"], pd.Series) else float(row["FED_RATE"].iloc[0])
        prev_fed_rate = float(row["PREV_FED_RATE"]) if not isinstance(row["PREV_FED_RATE"], pd.Series) else float(row["PREV_FED_RATE"].iloc[0])
    except Exception as e:
        print(f"[SKIP] {row.name}: Göstergeler NaN veya hatalı: {e}")
        continue
    # Eğer göstergeler NaN ise satırı atla
    if any(pd.isna(x) for x in [rsi, macd, signal, close_price, ma200, m2_growth, fed_rate, prev_fed_rate]):
        print(f"[SKIP] {row.name}: Göstergeler NaN")
        continue
    indicators = {
        "RSI": rsi,
        "MACD": macd,
        "Signal": signal,
        "Close": close_price,
        "MA200": ma200,
        "M2_GROWTH": m2_growth,
        "FED_RATE": fed_rate,
        "PREV_FED_RATE": prev_fed_rate
    }
    score = calculate_decision_score(indicators, {"RSI": 1, "MACD": 1}, news_score=0)
    price = close_price
    date = row['Date']
    ma50 = float(df.iloc[i]["Close"]) if i >= 50 else price
    # Teknik kırılım ve destek/direnç (örnek, gerçek fonksiyonlarla değiştirilebilir)
    direnc_kirildi = price > ma200 * 1.01
    destekten_sekti = price < ma200 * 0.99
    para_arzi_artiyor = m2_growth > 0
    faiz_dusuyor = fed_rate < prev_fed_rate
    # Strateji fonksiyonunu çağır ve kararını al
    if selected_strategy == 'trend':
        strategy_decision = trend_strategy_decision(ma50, ma200, price, macd)
    elif selected_strategy == 'mean_reversion':
        # Bollinger bandı yoksa dummy değer ver
        bollinger_low = row.get('BollingerLow', ma50 * 0.97)
        bollinger_high = row.get('BollingerHigh', ma50 * 1.03)
        strategy_decision = mean_reversion_strategy_decision(rsi, price, bollinger_low, bollinger_high)
    elif selected_strategy == 'volatility':
        atr = row.get('ATR', 0)
        vix = row.get('VIX', 0)
        prev_price = df.iloc[i-1]['Close'] if i > 0 else price
        strategy_decision = volatility_strategy_decision(atr, vix, price, prev_price)
    else:
        prev_price = df.iloc[i-1]['Close'] if i > 0 else price
        prev_volume = df.iloc[i-1]['Volume'] if i > 0 else row.get('Volume', 0)
        strategy_decision = momentum_strategy_decision(price, prev_price, row.get('Volume', 0), prev_volume)
    # Ağırlıklı skor ve confidence
    skor, confidence = alim_karari_ver_agirlikli(rsi, macd, signal, ma50, ma200, agirliklar)
    date_str = date.date() if hasattr(date, 'date') else str(date)
    print(f"DEBUG | Tarih: {date_str}, RSI: {rsi}, MACD: {macd}, Signal: {signal}, MA50: {ma50}, MA200: {ma200}, Skor: {skor}, Confidence: {confidence}, Agirliklar: {agirliklar}")
    alim_var = skor > 0.3  # Eşik: 0.3 (daha fazla işlem için düşürüldü)
    # Satış skoru (profesyonel mantık)
    sell_score = calculate_sell_score(rsi, macd, signal, price, ma200, m2_growth, fed_rate, prev_fed_rate)
    current_loss = (buy_price - price) / buy_price if stock > 0 else 0
    print(f"{date_str} | Score: {score:.2f} | SellScore: {sell_score} | AlimSkoruAgirlikli: {skor:.2f} | Agirliklar: {agirliklar} | Stock: {stock} | Cash: {cash}")
    # Skorları hesapla
    t_skor = teknik_skor(row)
    m_skor = makro_skor(row)
    f_skor = temel_skor(row)
    genel_skor = t_skor + m_skor + f_skor
    # Gelişmiş Deep Learning fiyat tahmini (LSTM + GRU ensemble)
    dl_preds = {}
    dl_confidences = {}
    dl_effect = 0
    dl_ensemble = None
    scaling_factor = 0.15  # Etkiyi büyütmek/küçültmek için
    # LSTM tahmini
    if 'lstm_model' in globals() and lstm_model is not None:
        features = [row.get('Close', 0), row.get('RSI', 0), row.get('MACD', 0), row.get('MA50', 0), row.get('MA200', 0), row.get('Volume', 0)]
        lstm_pred = get_lstm_prediction(lstm_model, features)
        lstm_conf = min(1, max(0, abs(lstm_pred)))  # Basit confidence: mutlak değer
        dl_preds['lstm'] = lstm_pred
        dl_confidences['lstm'] = lstm_conf
    # GRU tahmini (örnek, varsa)
    if 'gru_model' in globals() and gru_model is not None:
        features = [row.get('Close', 0), row.get('RSI', 0), row.get('MACD', 0), row.get('MA50', 0), row.get('MA200', 0), row.get('Volume', 0)]
        gru_pred = get_lstm_prediction(gru_model, features)  # get_lstm_prediction fonksiyonu GRU için de kullanılabilir
        gru_conf = min(1, max(0, abs(gru_pred)))
        dl_preds['gru'] = gru_pred
        dl_confidences['gru'] = gru_conf
    # Ensemble (ortalama)
    if dl_preds:
        preds = list(dl_preds.values())
        confs = list(dl_confidences.values())
        # Ağırlıklı ortalama (confidence ile)
        if sum(confs) > 0:
            dl_ensemble = sum(p * c for p, c in zip(preds, confs)) / sum(confs)
        else:
            dl_ensemble = sum(preds) / len(preds)
        # Dinamik etki: tahminin büyüklüğü ve güvenine göre
        avg_conf = sum(confs) / len(confs) if confs else 1
        dl_effect = float(np.clip(dl_ensemble * scaling_factor * avg_conf, -0.2, 0.2))
        genel_skor += dl_effect
    # Gelişmiş anomali tespiti ve risk yönetimi
    anomaly = False
    anomaly_score = 0
    risk_modu = 'normal'
    risk_aksiyonu = 'standart'
    risk_etki = 0
    pozisyon_carpani = 1.0
    if 'isoforest_model' in globals() and isoforest_model is not None:
        anomaly_features = [row.get('Close', 0), row.get('RSI', 0), row.get('MACD', 0), row.get('MA50', 0), row.get('MA200', 0), row.get('Volume', 0)]
        # Isolation Forest ile anomaly_score al
        try:
            anomaly_score = isoforest_model.decision_function([anomaly_features])[0]
        except Exception:
            anomaly_score = 0
        # Skor düşükse anomali var
        if anomaly_score < -0.15:
            anomaly = True
            risk_modu = 'panik'
            risk_aksiyonu = 'al-sat durdur, pozisyonu küçült'
            risk_etki = -0.25
            pozisyon_carpani = 0.2
        elif anomaly_score < 0.05:
            anomaly = True
            risk_modu = 'defansif'
            risk_aksiyonu = 'eşikleri yükselt, pozisyonu küçült'
            risk_etki = -0.12
            pozisyon_carpani = 0.5
        else:
            anomaly = False
            risk_modu = 'normal'
            risk_aksiyonu = 'standart'
            risk_etki = 0
            pozisyon_carpani = 1.0
        genel_skor += risk_etki
    # Piyasa rejimi ve eşik belirleme
    rejim = belirle_piyasa_rejimi(row.get('ATR'), row.get('VIX'), row.get('MA50'), row.get('MA200'))
    esik = rejime_gore_esik(rejim)
    # Strateji seçimi
    if rejim.startswith('trend'):
        selected_strategy = 'trend'
    elif rejim == 'sideways':
        selected_strategy = 'mean_reversion'
    elif rejim == 'volatility':
        selected_strategy = 'volatility'
    else:
        selected_strategy = 'momentum'
    # AL Kararı (ağırlıklı, öğrenen)
    if alim_var and cash >= price:
        miktar_raw = 1
        miktar = max(1, int(miktar_raw * pozisyon_carpani))
        cash, stock, miktar = uygula_alim_karari(alim_var, skor, cash, price, int(stock) + miktar - 1)
        if miktar > 0:
            buy_price = float(price)
            trade_log.append({'date': row.name, 'ticker': 'AAPL', 'action': 'BUY', 'price': price, 'amount': miktar, 'strategy': selected_strategy, 'strategy_decision': strategy_decision, 'confidence': confidence, 'rejim': rejim, 'risk_modu': risk_modu, 'anomaly': anomaly, 'anomaly_score': anomaly_score, 'risk_aksiyonu': risk_aksiyonu, 'risk_etki': risk_etki, 'pozisyon_carpani': pozisyon_carpani, 'genel_skor': skor, 'genel_skor_dl': genel_skor, 'dl_preds': dl_preds, 'dl_confidences': dl_confidences, 'dl_ensemble': dl_ensemble, 'dl_effect': dl_effect, 'success': None, 'weights': agirliklar.copy(), 'indicator': 'GENEL', 'RSI': rsi, 'MACD': macd, 'MA50': ma50, 'MA200': ma200, 'profit': 0, 'teknik_skor': t_skor, 'makro_skor': m_skor, 'temel_skor': f_skor, 'genel_skor_katmanli': genel_skor})
            print(f"{date_str} ALIM: {price} x {miktar} (AlimSkoruAgirlikli: {skor:.2f})")
            # Learning log: 5 gün sonra fiyatı kaydet
            if i + 5 < len(df):
                sonraki_fiyat = float(df.iloc[i + 5]["Close"])
                ma_durum = int(ma50 > ma200)
                log_learning_data(date_str, "ALIM", price, rsi, macd - signal, ma_durum, sonraki_fiyat)
    # SAT Kararı (profesyonel mantık)
    elif (sell_score >= 3 or (sell_score == 2 and current_loss > 0) or current_loss >= 0.10) and stock > 0:
        cash += price * stock
        result = "gain" if price > buy_price else "loss"
        update_weights_after_result(result, indicators_used=["RSI", "MACD", "MA200"])
        profit = (price - buy_price) * stock
        trade_log.append({'date': row.name, 'ticker': 'AAPL', 'action': 'SELL', 'price': price, 'amount': stock, 'strategy': selected_strategy, 'strategy_decision': strategy_decision, 'confidence': confidence, 'rejim': rejim, 'risk_modu': risk_modu, 'anomaly': anomaly, 'anomaly_score': anomaly_score, 'risk_aksiyonu': risk_aksiyonu, 'risk_etki': risk_etki, 'pozisyon_carpani': pozisyon_carpani, 'genel_skor': skor, 'genel_skor_dl': genel_skor, 'dl_preds': dl_preds, 'dl_confidences': dl_confidences, 'dl_ensemble': dl_ensemble, 'dl_effect': dl_effect, 'success': None, 'weights': agirliklar.copy(), 'indicator': 'GENEL', 'RSI': rsi, 'MACD': macd, 'MA50': ma50, 'MA200': ma200, 'profit': profit, 'teknik_skor': t_skor, 'makro_skor': m_skor, 'temel_skor': f_skor, 'genel_skor_katmanli': genel_skor})
        print(f"{date_str} SATIŞ: {price} x {stock} → {'KAR' if result == 'gain' else 'ZARAR'} (SellScore: {sell_score})")
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

# Trade logunun ilk 5 satırını kontrol için ekrana yazdır
print("\n--- Trade Log İlk 5 Satır ---")
for log in trade_log[:5]:
    print(log)

# Öğrenme logunu analiz et ve ağırlıkları güncelle
analyze_learning_log_and_update_weights()

# Çoklu NASDAQ hisse listesi (en büyük 50, örnek)
nasdaq_tickers = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'AVGO', 'GOOGL', 'TSLA', 'COST', 'PEP',
    'ADBE', 'CSCO', 'AMD', 'NFLX', 'QCOM', 'TXN', 'AMAT', 'INTC', 'TMUS', 'AMGN',
    'BKNG', 'LRCX', 'REGN', 'ISRG', 'VRTX', 'ADI', 'MU', 'MDLZ', 'INTU', 'GILD',
    'PDD', 'MAR', 'KDP', 'ATVI', 'CSX', 'MELI', 'CTAS', 'IDXX', 'KLAC', 'ASML',
    'PANW', 'CDNS', 'MRVL', 'AEP', 'EXC', 'SNPS', 'ORLY', 'NXPI', 'ROST', 'SBUX'
]

# Kullanıcıdan veya koddan tarih aralığı belirle
import datetime
sim_end = datetime.datetime.now()
sim_start = sim_end - datetime.timedelta(days=3*365)  # Son 3 yıl

# Tüm hisseler için veri çek (tarih aralığına göre)
all_data = fetch_multiple_stock_data(nasdaq_tickers, start_date=sim_start, end_date=sim_end)
# Eksik veya boş veri gelen ticker'ları çıkar
nasdaq_tickers = [t for t in nasdaq_tickers if t in all_data and all_data[t] is not None and not all_data[t].empty]

# Çoklu hisse portföyü için başlangıç portföyünü oluştur
portfolio = {t: {'stock': 0, 'buy_price': 0.0} for t in nasdaq_tickers}
cash = 100000  # Başlangıç nakdi

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

# Ortak tarih aralığını bul
common_index = all_data[nasdaq_tickers[0]].index
for ticker in nasdaq_tickers[1:]:
    common_index = common_index.intersection(all_data[ticker].index)
# Son 3 yıl ve 2 haftada bir (14 günde bir)
common_index = common_index[(common_index >= str(sim_start.date())) & (common_index <= str(sim_end.date()))]
common_index = common_index[::14]  # 2 haftada bir

portfolio_value = []
date_list = []

indicator_columns = ['RSI', 'MACD', 'MA50', 'MA200']  # Gerekirse diğer ana göstergeleri de ekleyin

# --- ML tabanlı strateji seçici altyapısı ---
from ml_models import train_strategy_selector, predict_strategy

# Strateji etiketleme fonksiyonu (örnek kurallar)
def label_strategy(row):
    if 'ATR' in row and row['ATR'] > 5:
        return 'volatility'
    elif 'MA50' in row and 'MA200' in row and row['MA50'] > row['MA200']:
        return 'trend'
    else:
        return 'mean'

# Eğitim verisi oluştur (AAPL üzerinden örnek)
train_df = all_data['AAPL'].dropna(subset=['ATR', 'MA50', 'MA200', 'RSI', 'MACD', 'Close', 'Volume']).copy()
train_df['strategy_label'] = train_df.apply(label_strategy, axis=1)
feature_cols = ['ATR', 'MA50', 'MA200', 'RSI', 'MACD', 'Close', 'Volume']
X_train = train_df[feature_cols]
y_train = train_df['strategy_label']
strategy_selector_model = train_strategy_selector(X_train, y_train)

# --- Gelişmiş stratejiye göre alım-satım karar fonksiyonları ---
def trend_strategy_decision(ma50, ma200, price, macd, stop_loss=0.03, take_profit=0.06):
    # Trend: MA50 > MA200 ve fiyat MA50 üzerinde ise al, tersi ise sat
    # MACD pozitifse alım sinyali güçlenir
    signal_strength = ma50 - ma200
    if ma50 > ma200 and price > ma50 and macd > 0:
        return {'action': 'buy', 'size': min(1, signal_strength/ma200), 'stop_loss': price*(1-stop_loss), 'take_profit': price*(1+take_profit)}
    elif ma50 < ma200 and price < ma50 and macd < 0:
        return {'action': 'sell', 'size': min(1, abs(signal_strength)/ma200)}
    else:
        return {'action': 'hold'}

def mean_reversion_strategy_decision(rsi, price, bollinger_low, bollinger_high):
    # Mean Reversion: RSI < 30 ve fiyat Bollinger alt bandına yakınsa al, RSI > 70 ve fiyat üst banda yakınsa sat
    if rsi < 30 and price <= bollinger_low * 1.02:
        size = min(1, (30 - rsi)/30)
        return {'action': 'buy', 'size': size}
    elif rsi > 70 and price >= bollinger_high * 0.98:
        size = min(1, (rsi - 70)/30)
        return {'action': 'sell', 'size': size}
    else:
        return {'action': 'hold'}

def volatility_strategy_decision(atr, vix=None, price=None, prev_price=None, atr_threshold=5):
    # Volatility: ATR ve VIX yüksekse küçük pozisyon, düşükse büyük pozisyon, ani fiyat sıçramasında hold
    if prev_price is not None and price is not None and abs(price - prev_price)/prev_price > 0.03:
        return {'action': 'hold'}
    if vix is not None and vix > 20 and atr > atr_threshold:
        return {'action': 'hold'}
    elif atr > atr_threshold:
        return {'action': 'buy', 'size': 0.2}
    else:
        return {'action': 'buy', 'size': 1.0}

def momentum_strategy_decision(price, prev_price, volume, prev_volume):
    # Momentum: Son 10 gün getiri pozitif ve hacim artıyorsa al, tersi ise sat
    if prev_price is not None and price > prev_price and volume > prev_volume:
        return {'action': 'buy', 'size': 1.0}
    elif prev_price is not None and price < prev_price and volume < prev_volume:
        return {'action': 'sell', 'size': 1.0}
    else:
        return {'action': 'hold'}

# --- Çok Katmanlı Skor Fonksiyonları ---
def teknik_skor(row):
    skorlar = []
    if 'RSI' in row: skorlar.append(max(0, min(1, (row['RSI']-30)/40)))  # 30-70 normalize
    if 'MACD' in row and 'Signal' in row: skorlar.append(1 if row['MACD'] > row['Signal'] else 0)
    if 'MA50' in row and 'MA200' in row: skorlar.append(1 if row['MA50'] > row['MA200'] else 0)
    # Bollinger Bands örneği (varsa)
    if 'Close' in row and 'BollingerHigh' in row and 'BollingerLow' in row:
        rel = (row['Close'] - row['BollingerLow']) / (row['BollingerHigh'] - row['BollingerLow'] + 1e-6)
        skorlar.append(rel)
    return sum(skorlar) / len(skorlar) if skorlar else 0.5

def makro_skor(row):
    skorlar = []
    if 'FED_RATE' in row and 'PREV_FED_RATE' in row:
        skorlar.append(1 if row['FED_RATE'] < row['PREV_FED_RATE'] else 0)
    if 'M2_GROWTH' in row: skorlar.append(max(0, min(1, row['M2_GROWTH']*10)))
    if 'VIX' in row: skorlar.append(1 - min(1, row['VIX']/40))
    return sum(skorlar) / len(skorlar) if skorlar else 0.5

def temel_skor(row):
    skorlar = []
    if 'PE' in row: skorlar.append(1 - min(1, row['PE']/40))
    if 'EPS' in row: skorlar.append(max(0, min(1, row['EPS']/10)))
    if 'DEBT' in row: skorlar.append(1 - min(1, row['DEBT']/100))
    if 'CASHFLOW' in row: skorlar.append(max(0, min(1, row['CASHFLOW']/10)))
    return sum(skorlar) / len(skorlar) if skorlar else 0.5

# --- Ağırlıklar ---
W1, W2, W3 = 1/3, 1/3, 1/3  # Başlangıçta eşit

# --- Piyasa Rejimi ve Dinamik Strateji Seçimi Fonksiyonları ---
def belirle_piyasa_rejimi(atr, vix, ma50, ma200, makro_sinyal=None):
    if atr is not None and vix is not None:
        if atr > 5 or vix > 20:
            return 'volatility'
    if ma50 > ma200 * 1.01:
        return 'trend_up'
    elif ma50 < ma200 * 0.99:
        return 'trend_down'
    else:
        return 'sideways'

def rejime_gore_esik(rejim):
    if rejim == 'trend_up':
        return {'buy': 0.7, 'sell': 0.3}
    elif rejim == 'trend_down':
        return {'buy': 0.3, 'sell': 0.7}
    elif rejim == 'volatility':
        return {'buy': 0.85, 'sell': 0.15}
    else:  # sideways
        return {'buy': 0.5, 'sell': 0.5}

# --- Deep Learning Fiyat Tahmini Fonksiyonu (örnek, LSTM) ---
def get_lstm_prediction(model, feature_row):
    # feature_row: DataFrame satırı veya array, uygun şekilde ölçeklenmeli
    # Burada örnek olarak modelin predict fonksiyonu çağrılır
    try:
        pred = model.predict([feature_row])[0]
        return pred
    except Exception:
        return 0

# --- Anomali Tespiti Fonksiyonu (Isolation Forest) ---
def is_anomaly_isoforest(model, feature_row):
    try:
        pred = model.predict([feature_row])[0]
        return pred == -1
    except Exception:
        return False

# --- Merkezi Karar ve Log Fonksiyonu ---
def karar_ve_islem(row, genel_skor, rejim, esik, selected_strategy, price, cash, stock, buy_price, agirliklar, portfolio, ticker, trade_log):
    action = 'hold'
    miktar = 0
    işlem_başarılı = None
    # Eşiklere göre karar
    if genel_skor >= esik['buy'] and cash >= price:
        action = 'buy'
        miktar = 1
        cash -= price * miktar
        portfolio[ticker]['stock'] += miktar
        portfolio[ticker]['buy_price'] = price
        işlem_başarılı = True
        trade_log.append({'date': row.name, 'ticker': ticker, 'action': 'BUY', 'price': price, 'amount': miktar, 'strategy': selected_strategy, 'strategy_decision': strategy_decision, 'confidence': confidence, 'rejim': rejim, 'risk_modu': risk_modu, 'anomaly': anomaly, 'anomaly_score': anomaly_score, 'risk_aksiyonu': risk_aksiyonu, 'risk_etki': risk_etki, 'pozisyon_carpani': pozisyon_carpani, 'genel_skor': genel_skor, 'genel_skor_dl': genel_skor, 'dl_preds': dl_preds, 'dl_confidences': dl_confidences, 'dl_ensemble': dl_ensemble, 'dl_effect': dl_effect, 'success': işlem_başarılı, 'weights': agirliklar.copy(), 'indicator': 'GENEL', 'RSI': rsi, 'MACD': macd, 'MA50': ma50, 'MA200': ma200, 'profit': 0, 'teknik_skor': t_skor, 'makro_skor': m_skor, 'temel_skor': f_skor, 'genel_skor_katmanli': genel_skor})
    elif genel_skor <= esik['sell'] and stock > 0:
        action = 'sell'
        miktar = stock
        cash += price * miktar
        portfolio[ticker]['stock'] -= miktar
        işlem_başarılı = price > buy_price
        trade_log.append({'date': row.name, 'ticker': ticker, 'action': 'SELL', 'price': price, 'amount': miktar, 'strategy': selected_strategy, 'strategy_decision': strategy_decision, 'confidence': confidence, 'rejim': rejim, 'risk_modu': risk_modu, 'anomaly': anomaly, 'anomaly_score': anomaly_score, 'risk_aksiyonu': risk_aksiyonu, 'risk_etki': risk_etki, 'pozisyon_carpani': pozisyon_carpani, 'genel_skor': genel_skor, 'genel_skor_dl': genel_skor, 'dl_preds': dl_preds, 'dl_confidences': dl_confidences, 'dl_ensemble': dl_ensemble, 'dl_effect': dl_effect, 'success': işlem_başarılı, 'weights': agirliklar.copy(), 'indicator': 'GENEL', 'RSI': rsi, 'MACD': macd, 'MA50': ma50, 'MA200': ma200, 'profit': profit, 'teknik_skor': t_skor, 'makro_skor': m_skor, 'temel_skor': f_skor, 'genel_skor_katmanli': genel_skor})
    else:
        action = 'hold'
    return cash, portfolio, trade_log, action, miktar

# Risk logunu başlat
risk_log = []

for idx, date in enumerate(common_index):
    total_value = cash
    ticker_scores = []
    for ticker in nasdaq_tickers:
        df = all_data[ticker]
        if date not in df.index:
            continue
        row = df.loc[date]
        # Bollinger için ek sütunlar
        bollinger_high = df['Close'].rolling(window=20).mean().loc[date] + 2 * df['Close'].rolling(window=20).std().loc[date]
        bollinger_low = df['Close'].rolling(window=20).mean().loc[date] - 2 * df['Close'].rolling(window=20).std().loc[date]
        row['BollingerHigh'] = bollinger_high
        row['BollingerLow'] = bollinger_low
        # Katman skorları
        t_skor = teknik_skor(row)
        m_skor = makro_skor(row)
        f_skor = temel_skor(row)
        genel_skor = W1 * t_skor + W2 * m_skor + W3 * f_skor
        row['GENEL_SKOR'] = genel_skor
        price = float(row['Close'].iloc[0]) if hasattr(row['Close'], 'iloc') else float(row['Close'])
        ma50 = float(row['MA50'].iloc[0]) if hasattr(row['MA50'], 'iloc') else float(row['MA50'])
        ma200 = float(row['MA200'].iloc[0]) if hasattr(row['MA200'], 'iloc') else float(row['MA200'])
        rsi = float(row['RSI'].iloc[0]) if hasattr(row['RSI'], 'iloc') else float(row['RSI'])
        macd = float(row['MACD'].iloc[0]) if hasattr(row['MACD'], 'iloc') else float(row['MACD'])
        signal = float(row['Signal'].iloc[0]) if hasattr(row['Signal'], 'iloc') else float(row['Signal'])
        stock = portfolio[ticker]['stock']
        buy_price = portfolio[ticker]['buy_price']
        atr = row['ATR'] if 'ATR' in row else 0
        volume = float(row['Volume'].iloc[0]) if hasattr(row['Volume'], 'iloc') else float(row['Volume'])
        # Piyasa rejimi ve eşik belirleme
        rejim = belirle_piyasa_rejimi(row.get('ATR'), row.get('VIX'), row.get('MA50'), row.get('MA200'))
        esik = rejime_gore_esik(rejim)
        # Strateji seçimi
        if rejim.startswith('trend'):
            selected_strategy = 'trend'
        elif rejim == 'sideways':
            selected_strategy = 'mean_reversion'
        elif rejim == 'volatility':
            selected_strategy = 'volatility'
        else:
            selected_strategy = 'momentum'
        # Anomali tespiti ve risk modu
        anomaly = False
        anomaly_score = 0
        risk_modu = 'normal'
        risk_aksiyonu = 'standart'
        risk_etki = 0
        pozisyon_carpani = 1.0
        if 'isoforest_model' in globals() and isoforest_model is not None:
            anomaly_features = [row.get('Close', 0), row.get('RSI', 0), row.get('MACD', 0), row.get('MA50', 0), row.get('MA200', 0), row.get('Volume', 0)]
            # Isolation Forest ile anomaly_score al
            try:
                anomaly_score = isoforest_model.decision_function([anomaly_features])[0]
            except Exception:
                anomaly_score = 0
            # Skor düşükse anomali var
            if anomaly_score < -0.15:
                anomaly = True
                risk_modu = 'panik'
                risk_aksiyonu = 'al-sat durdur, pozisyonu küçült'
                risk_etki = -0.25
                pozisyon_carpani = 0.2
            elif anomaly_score < 0.05:
                anomaly = True
                risk_modu = 'defansif'
                risk_aksiyonu = 'eşikleri yükselt, pozisyonu küçült'
                risk_etki = -0.12
                pozisyon_carpani = 0.5
            else:
                anomaly = False
                risk_modu = 'normal'
                risk_aksiyonu = 'standart'
                risk_etki = 0
                pozisyon_carpani = 1.0
            genel_skor += risk_etki
        # Gelişmiş Deep Learning fiyat tahmini (LSTM + GRU ensemble)
        dl_preds = {}
        dl_confidences = {}
        dl_effect = 0
        dl_ensemble = None
        scaling_factor = 0.15  # Etkiyi büyütmek/küçültmek için
        # LSTM tahmini
        if 'lstm_model' in globals() and lstm_model is not None:
            features = [row.get('Close', 0), row.get('RSI', 0), row.get('MACD', 0), row.get('MA50', 0), row.get('MA200', 0), row.get('Volume', 0)]
            lstm_pred = get_lstm_prediction(lstm_model, features)
            lstm_conf = min(1, max(0, abs(lstm_pred)))  # Basit confidence: mutlak değer
            dl_preds['lstm'] = lstm_pred
            dl_confidences['lstm'] = lstm_conf
        # GRU tahmini (örnek, varsa)
        if 'gru_model' in globals() and gru_model is not None:
            features = [row.get('Close', 0), row.get('RSI', 0), row.get('MACD', 0), row.get('MA50', 0), row.get('MA200', 0), row.get('Volume', 0)]
            gru_pred = get_lstm_prediction(gru_model, features)  # get_lstm_prediction fonksiyonu GRU için de kullanılabilir
            gru_conf = min(1, max(0, abs(gru_pred)))
            dl_preds['gru'] = gru_pred
            dl_confidences['gru'] = gru_conf
        # Ensemble (ortalama)
        if dl_preds:
            preds = list(dl_preds.values())
            confs = list(dl_confidences.values())
            # Ağırlıklı ortalama (confidence ile)
            if sum(confs) > 0:
                dl_ensemble = sum(p * c for p, c in zip(preds, confs)) / sum(confs)
            else:
                dl_ensemble = sum(preds) / len(preds)
            # Dinamik etki: tahminin büyüklüğü ve güvenine göre
            avg_conf = sum(confs) / len(confs) if confs else 1
            dl_effect = float(np.clip(dl_ensemble * scaling_factor * avg_conf, -0.2, 0.2))
            genel_skor += dl_effect
        # Merkezi karar ve işlem fonksiyonu ile yönet
        cash, portfolio, trade_log, action, miktar = karar_ve_islem(row, row['GENEL_SKOR'], rejim, esik, selected_strategy, price, cash, portfolio[ticker]['stock'], portfolio[ticker]['buy_price'], agirliklar, portfolio, ticker, trade_log)
        print(f"{date.date()} {ticker} | Strateji: {selected_strategy} | Rejim: {rejim} | Skor: {row['GENEL_SKOR']:.2f} | Karar: {action} | Miktar: {miktar} | Stock: {portfolio[ticker]['stock']} | Cash: {cash}")
        ticker_scores.append((ticker, genel_skor, price, ma50, ma200, rsi, macd, signal, stock, buy_price, row))
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
        alim_skoru, confidence = alim_karari_ver_agirlikli(rsi, macd, signal, ma50, ma200, agirliklar)
        # Göstergeleri kaydet
        save_indicator(ticker, 'RSI', rsi, str(date))
        save_indicator(ticker, 'MACD', macd, str(date))
        save_indicator(ticker, 'MA50', ma50, str(date))
        save_indicator(ticker, 'MA200', ma200, str(date))
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
                # Trade'i veritabanına kaydet
                save_trade(ticker, 'BUY', price, int(miktar), '', 0, str(date))
                print(f"{date.date()} {ticker} ALIM: {price} x {miktar} (Conf: {confidence})")
                if idx + 1 < len(df):
                    sonraki_fiyat = df.iloc[idx + 1]['Close'].iloc[0] if hasattr(df.iloc[idx + 1]['Close'], 'iloc') else float(df.iloc[idx + 1]['Close'])
                    ma_durum = int(ma50 > ma200)
                    log_learning_data(date.date(), f"ALIM-{ticker}", price, rsi, macd - signal, ma_durum, sonraki_fiyat)
        elif (sell_score >= 2 or current_loss >= 0.05) and stock > 0:
            cash += price * stock
            result = "gain" if price > buy_price else "loss"
            update_weights_after_result(result, indicators_used=["RSI", "MACD", "MA200"])
            profit = (price - buy_price) * stock
            trade_log.append({'date': date, 'ticker': ticker, 'action': 'SELL', 'price': price, 'result': result, 'sell_score': sell_score, 'amount': int(stock), 'indicator': 'GENEL', 'profit': profit, 'RSI': rsi, 'MACD': macd, 'MA50': ma50, 'MA200': ma200})
            print(f"{date.date()} {ticker} SATIŞ: {price} x {stock} → {'KAR' if result == 'gain' else 'ZARAR'}")
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
            log_df['success'] = (log_df['profit'] > 0).astype(int)
        else:
            log_df['success'] = log_df['success'].astype(int)
        new_weights = rolling_weight_update(log_df, indicator_columns, window=60)
        if new_weights:
            agirliklar.update(new_weights)
            print(f"Makine öğrenmesi ile ağırlıklar güncellendi: {agirliklar}")
    # --- Periyodik Monte Carlo Risk Analizi ve Otomatik Portföy Yeniden Dengeleme ---
    if idx > 0 and idx % 50 == 0:
        try:
            from ml_models import monte_carlo_portfolio_simulation
            price_df = pd.DataFrame({t: all_data[t]['Close'] for t in nasdaq_tickers})
            total_value = sum(portfolio[t]['stock'] * all_data[t]['Close'].loc[date] for t in nasdaq_tickers)
            weights = {t: (portfolio[t]['stock'] * all_data[t]['Close'].loc[date]) / total_value if total_value > 0 else 0 for t in nasdaq_tickers}
            print(f"[Risk] {date.date()} Monte Carlo risk analizi başlatılıyor...")
            # Simülasyonu çalıştır, risk metriklerini al
            risk_results = monte_carlo_portfolio_simulation(price_df, weights, n_days=30, n_sim=1000, initial_value=total_value, plot=False, return_metrics=True)
            var = risk_results.get('VaR', 0)
            cvar = risk_results.get('CVaR', 0)
            mdd = risk_results.get('MDD', 0)
            sharpe = risk_results.get('Sharpe', 0)
            sortino = risk_results.get('Sortino', 0)
            risk_action = 'none'
            # Risk eşikleri
            if var < -0.05 or cvar < -0.07 or mdd < -0.10:
                # Risk yüksek, portföyü defansif moda al
                for t in portfolio:
                    portfolio[t]['stock'] = int(portfolio[t]['stock'] * 0.5)
                risk_action = 'defansif: pozisyonlar %50 azaltıldı'
            elif sharpe < 0.5 or sortino < 0.7:
                # Getiri/riski düşük, portföyü yeniden dengele
                for t in portfolio:
                    portfolio[t]['stock'] = int(portfolio[t]['stock'] * 0.7)
                risk_action = 'denge: pozisyonlar %30 azaltıldı'
            else:
                risk_action = 'standart: değişiklik yok'
            # Risk loguna kaydet
            risk_log.append({'date': str(date), 'VaR': var, 'CVaR': cvar, 'MDD': mdd, 'Sharpe': sharpe, 'Sortino': sortino, 'action': risk_action, 'weights': weights.copy(), 'portfolio': {k: v['stock'] for k, v in portfolio.items()}})
            print(f"[Risk] {date.date()} Risk metrikleri: VaR={var:.3f}, CVaR={cvar:.3f}, MDD={mdd:.3f}, Sharpe={sharpe:.2f}, Sortino={sortino:.2f}, Aksiyon: {risk_action}")
        except Exception as e:
            print(f"[Risk] Monte Carlo simülasyonu çalıştırılamadı: {e}")
    # Trade loguna risk aksiyonu özetini ekle (her işlemde)
    if 'risk_action' in locals():
        if len(trade_log) > 0:
            trade_log[-1]['risk_action'] = risk_action

# Simülasyon sonunda risk logunu kaydet
with open('risk_log.json', 'w', encoding='utf-8') as f:
    json.dump(risk_log, f, ensure_ascii=False, indent=2)

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

# --- Simülasyon Sonu Özet ve Rapor ---
print("\n=== SİMÜLASYON ÖZETİ ===")
# Portföy performansı
final_values = {ticker: portfolio[ticker]['stock'] * all_data[ticker]['Close'].iloc[-1] for ticker in nasdaq_tickers}
total_final_value = sum(final_values.values()) + cash
initial_value = 100000  # Başlangıç nakit
getiri = (total_final_value - initial_value) / initial_value * 100
print(f"Başlangıç Değeri: {initial_value:.2f} USD")
print(f"Final Portföy Değeri (Cash dahil): {total_final_value:.2f} USD")
print(f"Toplam Getiri: %{getiri:.2f}")
# En iyi/kötü işlemler
import pandas as pd
log_df = pd.DataFrame(trade_log)
if not log_df.empty:
    log_df['kar'] = log_df.apply(lambda row: (row['price'] - portfolio[row['ticker']]['buy_price']) * row['amount'] if row['action']=='SELL' else 0, axis=1)
    en_iyi = log_df.loc[log_df['kar'].idxmax()] if (log_df['kar'] != 0).any() else None
    en_kotu = log_df.loc[log_df['kar'].idxmin()] if (log_df['kar'] != 0).any() else None
    if en_iyi is not None:
        print(f"En İyi İşlem: {en_iyi['date']} {en_iyi['ticker']} {en_iyi['action']} Kar: {en_iyi['kar']:.2f}")
    if en_kotu is not None:
        print(f"En Kötü İşlem: {en_kotu['date']} {en_kotu['ticker']} {en_kotu['action']} Kar: {en_kotu['kar']:.2f}")
# Strateji başarı oranları
if 'strategy' in log_df.columns:
    for strat in log_df['strategy'].unique():
        strat_df = log_df[log_df['strategy'] == strat]
        if not strat_df.empty and 'success' in strat_df.columns:
            oran = strat_df['success'].mean() * 100
            print(f"Strateji: {strat} | Başarı Oranı: %{oran:.1f}")
# Son ağırlıklar
def print_weights(ag):
    print("Güncel Ağırlıklar:")
    for k, v in ag.items():
        print(f"  {k}: {v:.3f}")
print_weights(agirliklar)
print("=== ML & Risk Analizleri ===")
print("(Detaylı analizler ve grafikler yukarıda gösterildi)")

# --- Ana simülasyonun sonunda ---
if __name__ == "__main__":
    # Sadece tek bir ana döngü çalışsın, paralel backtest kaldırıldı
    # Simülasyon sonunda toplam hisse değerlerini özetle
    final_values = {ticker: portfolio[ticker]['stock'] * all_data[ticker]['Close'].iloc[-1] for ticker in nasdaq_tickers}
    total_final_value = sum(final_values.values()) + cash
    print("\n--- Hisselerin Son Değerleri ---")
    for ticker, value in final_values.items():
        print(f"{ticker}: {value:.2f} USD")
    print(f"Toplam Portföy Değeri (Cash dahil): {total_final_value:.2f} USD")

    # --- ML Pipeline Entegrasyonu: AAPL verisiyle regresyon, LSTM ve anomali tespiti örneği ---
    try:
        from ml_models import train_price_regression, train_lstm_pytorch_price_forecast, detect_anomalies_isolation_forest, detect_anomalies_autoencoder
        print("\n[ML] AAPL verisiyle regresyon modeli eğitiliyor...")
        aapl_df = all_data['AAPL'].copy()
        # Regresyon (Random Forest)
        model, X_test, y_test, y_pred = train_price_regression(aapl_df, model_type='rf', plot=True)
        print("[ML] Regresyon modeli başarıyla çalıştı.")
        # PyTorch LSTM (Deep Learning)
        print("[ML] AAPL verisiyle PyTorch LSTM fiyat tahmini başlatılıyor...")
        lstm_model, y_test_inv, y_pred_inv = train_lstm_pytorch_price_forecast(aapl_df, plot=True)
        if lstm_model is not None:
            print("[ML] PyTorch LSTM fiyat tahmini başarıyla tamamlandı.")
        else:
            print("[ML] PyTorch LSTM pipeline çalıştırılamadı (PyTorch eksik veya veri yetersiz).")
        # Isolation Forest ile anomali tespiti
        print("[ML] AAPL verisiyle Isolation Forest anomali tespiti başlatılıyor...")
        _, anomalies = detect_anomalies_isolation_forest(aapl_df, plot=True)
        print(f"[ML] Tespit edilen anomali sayısı (Isolation Forest): {len(anomalies)}")
        # Autoencoder ile anomali tespiti
        print("[ML] AAPL verisiyle Autoencoder anomali tespiti başlatılıyor...")
        _, anomalies_ae = detect_anomalies_autoencoder(aapl_df, plot=True)
        print(f"[ML] Tespit edilen anomali sayısı (Autoencoder): {len(anomalies_ae)}")
    except Exception as e:
        print(f"[ML] Regresyon/LSTM/Anomali pipeline çalıştırılamadı: {e}")

    # --- Monte Carlo Risk Analizi ---
    try:
        from ml_models import monte_carlo_portfolio_simulation
        print("\n[Risk] Monte Carlo portföy simülasyonu başlatılıyor...")
        # Son fiyat verisi ve portföy ağırlıkları
        price_df = pd.DataFrame({t: all_data[t]['Close'] for t in nasdaq_tickers})
        # Portföydeki son ağırlıklar (hisse adedi/toplam portföy değeri)
        total_value = sum(portfolio[t]['stock'] * all_data[t]['Close'].iloc[-1] for t in nasdaq_tickers)
        weights = {t: (portfolio[t]['stock'] * all_data[t]['Close'].iloc[-1]) / total_value if total_value > 0 else 0 for t in nasdaq_tickers}
        monte_carlo_portfolio_simulation(price_df, weights, n_days=30, n_sim=1000, initial_value=total_value, plot=True)
    except Exception as e:
        print(f"[Risk] Monte Carlo simülasyonu çalıştırılamadı: {e}") 