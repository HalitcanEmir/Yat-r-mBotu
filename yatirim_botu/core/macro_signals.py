# TODO: Makro sinyaller (faiz, para arzı, VIX vs.) 

import pandas as pd
import os

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

def get_m2_growth(data_path="yatirim_botu/data/macro/m2.csv"):
    """
    M2 verisinden yıllık büyüme oranını hesaplar.
    """
    print("Trying to read:", os.path.abspath(data_path))
    df = pd.read_csv(data_path, parse_dates=["Date"])
    df = df.sort_values("Date")
    latest = df.iloc[-1]["Value"]
    one_year_ago = df[df["Date"] <= (df.iloc[-1]["Date"] - pd.Timedelta(days=365))].iloc[-1]["Value"]
    growth = (latest - one_year_ago) / one_year_ago
    return round(growth, 4)

def get_fed_rate(data_path="yatirim_botu/data/macro/fed_funds.csv"):
    """
    Fed'in politika faizini döner (son veri).
    """
    df = pd.read_csv(data_path, parse_dates=["Date"])
    df = df.sort_values("Date")
    return float(df.iloc[-1]["Value"])

def get_vix(data_path="yatirim_botu/data/macro/vix.csv"):
    """
    VIX endeksinin son değerini döner.
    """
    df = pd.read_csv(data_path, parse_dates=["Date"])
    df = df.sort_values("Date")
    return float(df.iloc[-1]["Value"])

def forecast_m2_prophet(data_path="yatirim_botu/data/macro/m2.csv", periods=90):
    """
    Facebook Prophet ile M2 para arzı için 3 ay (90 gün) ileriye dönük tahmin üretir.
    Dönüş: Tahmin DataFrame'i (tarih ve tahmini değerler)
    """
    if Prophet is None:
        print("Prophet yüklü değil. 'pip install prophet' ile yükleyin.")
        return None
    df = pd.read_csv(data_path, parse_dates=["Date"])
    df = df.sort_values("Date")
    prophet_df = df.rename(columns={"Date": "ds", "Value": "y"})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]].tail(periods)

def get_macro_score(future=False):
    """
    M2, Fed Rate ve VIX'e göre -1 ile +1 arasında makro sinyal üretir.
    Eğer future=True ise, M2'nin 3 ay sonrası tahminiyle skor üretir.
    """
    if future:
        forecast = forecast_m2_prophet()
        if forecast is not None:
            m2_growth = (forecast["yhat"].iloc[-1] - forecast["yhat"].iloc[0]) / forecast["yhat"].iloc[0]
        else:
            m2_growth = get_m2_growth()
    else:
        m2_growth = get_m2_growth()
    fed = get_fed_rate()
    vix = get_vix()

    score = 0

    if m2_growth > 0.08:
        score += 0.5
    elif m2_growth < 0.02:
        score -= 0.5

    if fed < 3.0:
        score += 0.3
    elif fed > 5.0:
        score -= 0.3

    if vix < 20:
        score += 0.2
    elif vix > 30:
        score -= 0.2

    return round(score, 2) 