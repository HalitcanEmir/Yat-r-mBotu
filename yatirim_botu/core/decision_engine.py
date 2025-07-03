# TODO: Karar algoritması (al/sat/bekle) 

def make_decision(df, macro_score=0.0):
    """
    Verilen DataFrame ve makro skoruna göre AL/SAT/BEKLE kararı üretir.
    """
    latest = df.iloc[-1]
    score = 0

    # RSI: Aşırı alım-satım sinyali
    if latest['Rsi'] < 30:
        score += 1
    elif latest['Rsi'] > 70:
        score -= 1

    # MACD: Yükseliş/kesim
    if latest['Macd'] > latest['Macd_signal']:
        score += 1
    else:
        score -= 0.5

    # Fiyat MA50 üzerinde mi?
    if latest['Close'] > latest['Ma50']:
        score += 1
    else:
        score -= 0.5

    # SuperTrend sinyali pozitif mi?
    if latest.get("SuperTrend") == True:
        score += 1
    else:
        score -= 0.5

    # Keltner üst bandının altında mı? (alım fırsatı)
    if latest['Close'] < latest['KC_upper']:
        score += 0.5

    # Makro ekonomik katkı
    score += macro_score

    # Son kararı ver
    if score >= 2.5:
        return "BUY"
    elif score <= -1.5:
        return "SELL"
    else:
        return "HOLD"

def make_detailed_decision(df, macro_score=0.0):
    decision = make_decision(df, macro_score)
    return {
        "decision": decision,
        "confidence": round(macro_score, 2)
    }

def interpret_news_sentiment(news_text):
    """
    Gemini'den gelen haber analizini skora çevirir.
    """
    if news_text is None:
        return 0  # haber alınamadıysa nötr say
    text = news_text.lower()

    if "pozitif" in text or "olumlu" in text:
        return 0.5
    elif "negatif" in text or "olumsuz" in text:
        return -0.5
    else:
        return 0  # nötr

def calculate_decision_score(indicators, weights, news_score=None):
    """
    Tüm göstergeler ve haber analiziyle puan hesapla.
    """
    total = 0
    for key, value in indicators.items():
        weight = weights.get(key, 1)
        total += value * weight

    if news_score is not None:
        total += news_score * weights.get("News", 1)

    return total 