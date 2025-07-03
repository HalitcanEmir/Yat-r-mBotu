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