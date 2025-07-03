# TODO: Performans analizi (Sharpe, Sortino, kar/zarar) 

import pandas as pd
import json
import numpy as np

def calculate_returns(history_path="yatirim_botu/data/history.json"):
    with open(history_path, "r") as f:
        history = json.load(f)

    df = pd.DataFrame(history)

    # Tarihe göre sırala
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    # Günlük getiriler
    df['return'] = df['result']

    # Günlük getiriden kümülatif getiri
    df['cumulative'] = (1 + df['return']).cumprod()

    return df[['date', 'return', 'cumulative']]

def calculate_sharpe(returns: pd.Series, risk_free_rate=0.01):
    """
    Getirilerin Sharpe oranını hesaplar.
    """
    excess_returns = returns - (risk_free_rate / 252)
    return round(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252), 2)

def max_drawdown(cumulative: pd.Series):
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak) / peak
    return round(drawdown.min() * 100, 2)  # yüzde olarak

def success_ratio(df: pd.DataFrame):
    """
    Pozitif sonuçlu işlemlerin oranı
    """
    success = df[df["return"] > 0]
    return round(len(success) / len(df) * 100, 2)

def generate_performance_report():
    df = calculate_returns()
    sharpe = calculate_sharpe(df['return'])
    drawdown = max_drawdown(df['cumulative'])
    success = success_ratio(df)

    return {
        "total_return": round(df['cumulative'].iloc[-1] - 1, 4),
        "sharpe_ratio": sharpe,
        "max_drawdown": drawdown,
        "success_rate": success
    } 