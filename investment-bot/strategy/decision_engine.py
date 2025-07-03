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

def alim_karari_ver_agirlikli(rsi, macd, signal, ma50, ma200, agirliklar):
    """
    Göstergelerin ağırlıklarına göre alım skoru hesaplar. Toplam skor en fazla 1 olur.
    agirliklar: {'RSI': float, 'MACD': float, 'MA': float}
    """
    skor = 0
    if rsi < 30:
        skor += agirliklar.get("RSI", 0)
    if macd > signal:
        skor += agirliklar.get("MACD", 0)
    if ma50 > ma200:
        skor += agirliklar.get("MA", 0)
    return skor
# Kullanım:
# agirliklar = gosterge_agirliklari_ogren()
# skor = alim_karari_ver_agirlikli(rsi, macd, signal, ma50, ma200, agirliklar) 