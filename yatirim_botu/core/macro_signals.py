# TODO: Makro sinyaller (faiz, para arzı, VIX vs.) 

import pandas as pd
import os

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

def get_macro_score():
    """
    M2, Fed Rate ve VIX'e göre -1 ile +1 arasında makro sinyal üretir.
    """
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