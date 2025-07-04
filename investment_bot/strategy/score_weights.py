# Score weights management implementation will go here 

import pandas as pd

def dynamic_indicator_weighting(log_df):
    """
    Trade log DataFrame'den hangi göstergenin daha başarılı olduğunu bulur ve ağırlıkları normalize eder.
    log_df: DataFrame, en azından 'indicator' ve 'profit' sütunları olmalı.
    Dönüş: {'RSI': 0.2, 'MACD': 0.3, ...}
    """
    indicator_success = log_df.groupby('indicator')['profit'].mean()
    if indicator_success.sum() == 0:
        # Hiç kâr yoksa eşit ağırlık ver
        indicator_weights = pd.Series(1, index=indicator_success.index)
    else:
        indicator_weights = indicator_success / indicator_success.sum()
    return indicator_weights.to_dict()

# Global ağırlık güncelleme fonksiyonu (örnek)
def update_global_weights_from_log(log_df, global_weights):
    new_weights = dynamic_indicator_weighting(log_df)
    for k, v in new_weights.items():
        if k in global_weights:
            global_weights[k] = v
    return global_weights 

def rolling_weight_update(log_df, indicator_columns, window=60):
    """
    Son window işlemi kullanarak RandomForest ile ağırlıkları optimize eder.
    log_df: trade log DataFrame
    indicator_columns: ['RSI', 'MACD', ...]
    window: son kaç işlemde bakılacak
    Dönüş: {'RSI': 0.2, 'MACD': 0.3, ...}
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        print("scikit-learn yüklü değil. 'pip install scikit-learn' ile yükleyin.")
        return None
    recent_results = log_df.tail(window)
    X = recent_results[indicator_columns]
    y = recent_results['success']
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    new_weights = model.feature_importances_
    # Normalize
    new_weights = new_weights / new_weights.sum()
    return dict(zip(indicator_columns, new_weights)) 

def strong_signal_filter(tech_score, pred_score, sector_score, tech_thresh: float = 1.0, pred_thresh: float = 0.05, sector_thresh: float = 0.7):
    """
    Hem teknik skor, hem tahmin, hem sektör skoru yüksekse True döner.
    """
    return (
        tech_score >= tech_thresh and
        pred_score >= pred_thresh and
        sector_score >= sector_thresh
    ) 

def compute_momentum_score(df, windows=[63, 126, 252]):
    """
    Son 3-6-12 ayda fiyatı en çok yükselen hisselere ek skor verir.
    df: fiyat verisi (DataFrame, 'Close' sütunu olmalı)
    windows: momentum pencereleri (gün cinsinden)
    """
    score = 0
    for w in windows:
        if len(df) > w:
            start = df['Close'].iloc[-w]
            end = df['Close'].iloc[-1]
            if start > 0:
                score += (end - start) / start
    return score / len(windows) if windows else 0

def top10_bias(ticker, top10_list, bias=0.2):
    """
    En büyük 10 şirkete ek skor/bias uygular.
    """
    return bias if ticker in top10_list else 0 