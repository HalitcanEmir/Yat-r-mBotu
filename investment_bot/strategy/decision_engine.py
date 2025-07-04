# Decision engine implementation will go here 

def calculate_decision_score(indicators, weights, news_score=0):
    score = 0
    for k, v in indicators.items():
        score += v * weights.get(k, 1)
    score += news_score * weights.get("News", 1)
    return score 

def calculate_sell_score(rsi, macd, signal, price, ma200, m2_growth, fed_rate, prev_fed_rate):
    """
    Çoklu göstergeye dayalı satış skoru hesaplar.
    - rsi: float
    - macd: float
    - signal: float
    - price: float
    - ma200: float
    - m2_growth: float (yıllık para arzı büyüme oranı)
    - fed_rate: float (güncel faiz)
    - prev_fed_rate: float (önceki faiz)
    """
    score = 0
    if rsi > 70:
        score += 1
    if macd < signal:
        score += 1
    if price < ma200:
        score += 1
    if m2_growth < 0:  # Para arzı daralıyor
        score += 1
    if fed_rate > prev_fed_rate:  # Faiz artıyor
        score += 1
    return score 

def alim_karari_ver(price, rsi, macd, signal, ma50, ma200,
                    para_arzi_artiyor, faiz_dusuyor,
                    direnc_kirildi, destekten_sekti):
    """
    Profesyonel yatırımcı davranışını taklit eden alım kararı fonksiyonu.
    Göstergeler ve piyasa sinyallerine göre uygun alım fırsatlarını yakalar.
    """
    alim_skoru = 0

    # RSI aşırı satım
    if rsi < 30:
        alim_skoru += 1

    # MACD kesişimi
    if macd > signal:
        alim_skoru += 1

    # Golden Cross
    if ma50 > ma200:
        alim_skoru += 1

    # Teknik formasyonlar
    if direnc_kirildi:
        alim_skoru += 1
    if destekten_sekti:
        alim_skoru += 1

    # Makro ortam
    if para_arzi_artiyor:
        alim_skoru += 1
    if faiz_dusuyor:
        alim_skoru += 1

    # Alım kararı – dinamik eşik
    if alim_skoru >= 4:
        return True
    elif alim_skoru >= 3 and para_arzi_artiyor:
        return True
    else:
        return False 

def calculate_atr_stop_levels(entry_price, atr, atr_multiplier=2, profit_multiplier=3):
    """
    ATR bazlı dinamik stop-loss ve take-profit seviyeleri hesaplar.
    """
    stop_loss = entry_price - (atr * atr_multiplier)
    take_profit = entry_price + (atr * profit_multiplier)
    return stop_loss, take_profit

def confidence_score(indicator_signals):
    """
    Göstergelerin aynı yönde olması veya skorun yüksekliğine göre 0-1 arası güven skoru döner.
    indicator_signals: list of -1, 0, 1 (her gösterge için sinyal)
    """
    if not indicator_signals:
        return 0
    # Çoğunluk aynı yönde mi?
    pos = sum(1 for s in indicator_signals if s > 0)
    neg = sum(1 for s in indicator_signals if s < 0)
    total = len(indicator_signals)
    confidence = max(pos, neg) / total
    return round(confidence, 2)

def alim_karari_ver_agirlikli(rsi, macd, signal, ma50, ma200, agirliklar):
    """
    Göstergelerin ağırlıklarına göre alım skoru ve güven skoru hesaplar. Toplam skor en fazla 1 olur.
    Dönüş: (skor, confidence)
    """
    skor = 0
    signals = []
    if rsi < 30:
        skor += agirliklar.get("RSI", 0)
        signals.append(1)
    else:
        signals.append(0)
    if macd > signal:
        skor += agirliklar.get("MACD", 0)
        signals.append(1)
    else:
        signals.append(0)
    if ma50 > ma200:
        skor += agirliklar.get("MA", 0)
        signals.append(1)
    else:
        signals.append(0)
    conf = confidence_score(signals)
    return skor, conf

def macro_score(fed_rate, m2_growth, vix, unemployment):
    """
    Makroekonomik göstergelerden skor üretir. 0-1 arası normalize skor döner.
    """
    score = 0
    if fed_rate < 3:  # Düşük faiz pozitif
        score += 0.25
    if m2_growth > 0:  # Para arzı artıyor
        score += 0.25
    if vix < 20:  # Düşük VIX pozitif
        score += 0.25
    if unemployment < 5:  # Düşük işsizlik pozitif
        score += 0.25
    return round(score, 2)

def price_volume_signal(close, volume):
    """
    Fiyat-hacim uyumu sinyali üretir. 1: pozitif, -1: negatif, 0: nötr.
    """
    if len(close) < 2 or len(volume) < 2:
        return 0
    price_up = close[-1] > close[-2]
    vol_up = volume[-1] > volume[-2]
    if price_up and vol_up:
        return 1
    elif not price_up and vol_up:
        return -1
    else:
        return 0

def vix_score(vix):
    """VIX düşükse pozitif, yüksekse negatif skor döner."""
    if vix < 15:
        return 1
    elif vix < 20:
        return 0.5
    elif vix < 30:
        return 0
    else:
        return -1

def putcall_score(putcall):
    """Put/Call Ratio: 0.6 altı aşırı iyimserlik (-), 1.2 üstü aşırı korku (+)"""
    if putcall < 0.6:
        return -1
    elif putcall > 1.2:
        return 1
    else:
        return 0

def adline_score(adline_slope):
    """A/D Line yukarı (pozitif slope) ise pozitif, aşağı ise negatif."""
    if adline_slope > 0:
        return 1
    elif adline_slope < 0:
        return -1
    else:
        return 0

def m2_score(m2_growth):
    """M2 artıyorsa pozitif, düşüyorsa negatif."""
    if m2_growth > 0:
        return 1
    elif m2_growth < 0:
        return -1
    else:
        return 0

def pmi_score(pmi):
    """PMI 50 üstü pozitif, altı negatif."""
    if pmi > 50:
        return 1
    elif pmi < 50:
        return -1
    else:
        return 0

def hy_spread_score(spread):
    """High-yield spread artıyorsa negatif, düşüyorsa pozitif."""
    if spread > 5:
        return -1
    elif spread < 3:
        return 1
    else:
        return 0

def yield_curve_score(curve):
    """Yield curve tersse (inverted, negatif) negatif skor."""
    if curve < 0:
        return -1
    elif curve > 0.5:
        return 1
    else:
        return 0

def fund_flows_score(flows):
    """Fonlara para giriyorsa pozitif, çıkıyorsa negatif."""
    if flows > 0:
        return 1
    elif flows < 0:
        return -1
    else:
        return 0

def karar_skora_cevir(
    rsi, macd, signal, ma50, ma200, supertrend, keltner, bollinger, fibo, price_vol, macro,
    vix, putcall, adline, m2, pmi, hy_spread, yield_curve, fund_flows, agirliklar
):
    """
    Tüm teknik, makro ve öncü piyasa sinyallerini ağırlıklı olarak birleştirir.
    """
    skor = 0
    skor += agirliklar.get("RSI", 0.1) * (1 if rsi < 30 else 0)
    skor += agirliklar.get("MACD", 0.1) * (1 if macd > signal else 0)
    skor += agirliklar.get("MA", 0.1) * (1 if ma50 > ma200 else 0)
    skor += agirliklar.get("SuperTrend", 0.1) * (1 if supertrend else 0)
    skor += agirliklar.get("Keltner", 0.05) * (1 if keltner else 0)
    skor += agirliklar.get("Bollinger", 0.05) * (1 if bollinger else 0)
    skor += agirliklar.get("Fibo", 0.05) * (1 if fibo else 0)
    skor += agirliklar.get("PriceVol", 0.1) * price_vol
    skor += agirliklar.get("Macro", 0.1) * macro
    skor += agirliklar.get("VIX", 0.05) * vix
    skor += agirliklar.get("PutCall", 0.05) * putcall
    skor += agirliklar.get("ADLine", 0.05) * adline
    skor += agirliklar.get("M2", 0.05) * m2
    skor += agirliklar.get("PMI", 0.05) * pmi
    skor += agirliklar.get("HYSpread", 0.05) * hy_spread
    skor += agirliklar.get("YieldCurve", 0.05) * yield_curve
    skor += agirliklar.get("FundFlows", 0.05) * fund_flows
    return round(skor, 2) 

def advanced_karar_skora_cevir(
    rsi, macd, signal, ma50, ma200, price, bollinger_high, bollinger_low, volume, prev_volume, agirliklar
):
    """
    Gelişmiş, oransal ve dinamik alım skoru hesaplar.
    - RSI: 30 altı tam puan, 30-40 arası lineer azalan
    - MACD: MACD-Signal farkı büyüdükçe puan artar
    - MA: MA50-MA200 farkı normalize
    - Bollinger: Fiyat alt bandın altında ise pozitif, üst bandın üstünde ise negatif
    - Hacim: Hacim artışı pozitif
    """
    skor = 0
    # RSI
    if rsi < 30:
        skor += agirliklar.get("RSI", 0.1) * 1
    elif rsi < 40:
        skor += agirliklar.get("RSI", 0.1) * (40 - rsi) / 10
    # MACD
    macd_diff = macd - signal
    skor += agirliklar.get("MACD", 0.1) * min(max(macd_diff / 0.5, 0), 1)
    # MA
    ma_diff = ma50 - ma200
    if price > 0:
        skor += agirliklar.get("MA", 0.1) * min(max(ma_diff / (0.05 * price), 0), 1)
    # Bollinger
    if price < bollinger_low:
        skor += agirliklar.get("Bollinger", 0.05) * 1
    elif price > bollinger_high:
        skor -= agirliklar.get("Bollinger", 0.05) * 1
    # Hacim
    if prev_volume > 0:
        vol_change = (volume - prev_volume) / prev_volume
        skor += agirliklar.get("Volume", 0.05) * min(max(vol_change, 0), 1)
    return round(skor, 2) 

from strategy.score_weights import strong_signal_filter

def decide_with_prediction(tech_score, pred_6m, pred_1y, current_price, sector_score=None, buy_threshold=1.0, sell_threshold=-1.0, pred_weight=0.7, sector_thresh=0.7):
    """
    Fiyat tahmini, teknik skor ve sektör skoru ile alım/satım/hold kararı verir.
    """
    score = tech_score
    pred_score = 0
    # Tahminler mevcutsa, beklenen getiri oranına göre skor ekle
    if pred_6m is not None and current_price > 0:
        ret_6m = (float(pred_6m) - current_price) / current_price
        pred_score += ret_6m
    if pred_1y is not None and current_price > 0:
        ret_1y = (float(pred_1y) - current_price) / current_price
        pred_score += ret_1y
    # Ortalama al, 2 tahmin varsa etkisi daha yüksek
    if pred_score != 0:
        pred_score = pred_score / (2 if (pred_6m is not None and pred_1y is not None) else 1)
    total_score = score + pred_weight * pred_score
    # Sadece güçlü sinyal ve sektör skoru yüksekse al
    if sector_score is not None and strong_signal_filter(score, pred_score, sector_score, tech_thresh=buy_threshold, pred_thresh=0.05, sector_thresh=sector_thresh):
        return 'buy'
    elif total_score < sell_threshold:
        return 'sell'
    else:
        return 'hold' 