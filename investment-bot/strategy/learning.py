# Learning from trade results implementation will go here 

import json
import os
import csv
import pandas as pd

def update_weights_after_result(result, indicators_used, weights_path="weights.json"):
    if not os.path.exists(weights_path):
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
            weights[ind] = max(0.1, min(weights[ind], 3.0))
    with open(weights_path, "w") as f:
        json.dump(weights, f, indent=2)

def log_learning_data(tarih, islem, fiyat, rsi, macd, ma_durum, sonraki_fiyat):
    """
    Alım/satım işlemi sonrası 5 gün sonra fiyat değişimini ve göstergeleri learning_log.csv dosyasına kaydeder.
    """
    sonuc = (sonraki_fiyat - fiyat) / fiyat  # % kazanç/kayıp
    with open("learning_log.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            tarih, islem, round(fiyat,2), round(rsi,2), round(macd,2), ma_durum, round(sonuc*100, 2)
        ])

def analyze_learning_log_and_update_weights(log_path="learning_log.csv", weights_path="../weights.json"):
    """
    learning_log.csv dosyasını analiz eder, göstergelerin başarı oranına göre ağırlıkları günceller.
    - RSI, MACD, MA50>MA200 için ayrı başarı oranı hesaplar.
    - Sonuçlara göre ağırlıkları weights.json dosyasına yazar.
    """
    if not os.path.exists(log_path):
        print(f"Log dosyası bulunamadı: {log_path}")
        return
    df = pd.read_csv(log_path, header=None, names=["Tarih", "Islem", "Fiyat", "RSI", "MACD", "MA_DURUM", "Sonuc"])
    # Tip dönüşümü
    df["RSI"] = pd.to_numeric(df["RSI"], errors="coerce")
    df["MACD"] = pd.to_numeric(df["MACD"], errors="coerce")
    df["MA_DURUM"] = pd.to_numeric(df["MA_DURUM"], errors="coerce")
    df["Sonuc"] = pd.to_numeric(df["Sonuc"], errors="coerce")
    # Sadece ALIM işlemlerini ve 5 gün sonrası getiriyi analiz et
    alim_df = df[df["Islem"].str.startswith("ALIM")]
    if alim_df.empty:
        print("Analiz için yeterli ALIM işlemi yok.")
        return
    # Başarı: Sonuc > 0 ise başarılı
    rsi_success = (alim_df.loc[alim_df["RSI"] < 30, "Sonuc"] > 0).mean() if (alim_df["RSI"] < 30).any() else 0.1
    macd_success = (alim_df.loc[alim_df["MACD"] > 0, "Sonuc"] > 0).mean() if (alim_df["MACD"] > 0).any() else 0.1
    ma_success = (alim_df.loc[alim_df["MA_DURUM"] == 1, "Sonuc"] > 0).mean() if (alim_df["MA_DURUM"] == 1).any() else 0.1
    # Minimum ağırlık 0.1 olsun
    scores = {
        "RSI": max(rsi_success, 0.1),
        "MACD": max(macd_success, 0.1),
        "MA50>MA200": max(ma_success, 0.1)
    }
    total = sum(scores.values())
    if total == 0:
        print("Başarı oranı yok, ağırlıklar güncellenmedi.")
        return
    weights = {k: round(v/total, 2) for k, v in scores.items()}
    # Eski ağırlıkları oku ve güncelle
    if os.path.exists(weights_path):
        with open(weights_path, "r") as f:
            old_weights = json.load(f)
    else:
        old_weights = {}
    old_weights.update(weights)
    with open(weights_path, "w") as f:
        json.dump(old_weights, f, indent=2)
    print(f"Ağırlıklar güncellendi: {weights}")

def gosterge_agirliklari_ogren(dosya="learning_log.csv"):
    """
    learning_log.csv dosyasını okuyup, RSI, MACD ve MA göstergelerinin başarı oranına göre normalize edilmiş ağırlıklar döndürür.
    Dosya yoksa varsayılan ağırlıklar döner.
    """
    if not os.path.exists(dosya):
        print(f"{dosya} bulunamadı, varsayılan ağırlıklar kullanılacak.")
        return {"RSI": 0.33, "MACD": 0.33, "MA": 0.34}
    df = pd.read_csv(dosya, header=None, names=["Tarih", "Islem", "Fiyat", "RSI", "MACD", "MA", "Sonuc"])
    # Tip dönüşümü
    df["RSI"] = pd.to_numeric(df["RSI"], errors="coerce")
    df["MACD"] = pd.to_numeric(df["MACD"], errors="coerce")
    df["MA"] = pd.to_numeric(df["MA"], errors="coerce")
    df["Sonuc"] = pd.to_numeric(df["Sonuc"], errors="coerce")
    df = df[df["Islem"] == "ALIM"]
    if df.empty:
        print("Yeterli ALIM işlemi yok.")
        return {"RSI": 0.33, "MACD": 0.33, "MA": 0.34}
    rsi_col = pd.Series(df["RSI"])
    macd_col = pd.Series(df["MACD"])
    ma_col = pd.Series(df["MA"])
    rsi_katki = pd.Series(df.loc[df["Sonuc"] > 0, "RSI"]).count() / rsi_col.count() if rsi_col.count() > 0 else 0.1
    macd_katki = pd.Series(df.loc[df["Sonuc"] > 0, "MACD"]).count() / macd_col.count() if macd_col.count() > 0 else 0.1
    ma_katki   = pd.Series(df.loc[df["Sonuc"] > 0, "MA"]).sum() / ma_col.count() if ma_col.count() > 0 else 0.1
    toplam = rsi_katki + macd_katki + ma_katki
    agirliklar = {
        "RSI": round(rsi_katki / toplam, 2) if toplam > 0 else 0.33,
        "MACD": round(macd_katki / toplam, 2) if toplam > 0 else 0.33,
        "MA": round(ma_katki / toplam, 2) if toplam > 0 else 0.34
    }
    print("🔍 Güncellenmiş Ağırlıklar:", agirliklar)
    return agirliklar 