# TODO: Başarısız işlemlerden öğrenme 

import json
import os

def log_trade_result(record, file_path="yatirim_botu/data/history.json"):
    """
    Tek bir işlem kaydını hafızaya ekler.
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            history = json.load(f)
    else:
        history = []

    history.append(record)

    with open(file_path, "w") as f:
        json.dump(history, f, indent=4)

def update_weights(history_path="yatirim_botu/data/history.json"):
    """
    Tüm geçmiş işlemleri analiz ederek sinyal ağırlıklarını günceller.
    """
    if not os.path.exists(history_path):
        return {}
    with open(history_path, "r") as f:
        trades = json.load(f)

    signal_scores = {}
    signal_counts = {}

    for trade in trades:
        for signal, weight in trade["signals"].items():
            if signal not in signal_scores:
                signal_scores[signal] = 0
                signal_counts[signal] = 0
            signal_scores[signal] += weight * trade["result"]
            signal_counts[signal] += 1

    # Ortalama başarıyı hesapla
    signal_weights = {}
    for signal in signal_scores:
        avg = signal_scores[signal] / signal_counts[signal]
        signal_weights[signal] = round(avg, 3)

    return signal_weights 

def update_weights_after_result(result, indicators_used, weights_path="yatirim_botu/weights.json"):
    """
    Sonuç 'loss' ise göstergelerin ağırlığını azalt, 'win' ise artır.
    """
    if not os.path.exists(weights_path):
        # Varsayılan ağırlıklar
        weights = {
            "RSI": 1.0,
            "MACD": 1.0,
            "SuperTrend": 1.0,
            "MultiTimeframe": 1.5,
            "SupportResistance": 1.2,
            "News": 0.6
        }
    else:
        with open(weights_path, "r") as f:
            weights = json.load(f)

    adjustment = -0.1 if result == "loss" else 0.05

    for ind in indicators_used:
        if ind in weights:
            weights[ind] += adjustment
            weights[ind] = max(0.1, min(weights[ind], 3.0))  # Sınırla

    with open(weights_path, "w") as f:
        json.dump(weights, f, indent=2) 