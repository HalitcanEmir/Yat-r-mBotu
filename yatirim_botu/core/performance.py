# TODO: Performans analizi (Sharpe, Sortino, kar/zarar) 

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

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

def calculate_sortino(returns: pd.Series, risk_free_rate=0.01):
    """
    Sortino oranı: sadece negatif sapmalarla risk ölçümü
    """
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-8
    excess_returns = returns - (risk_free_rate / 252)
    return round(np.mean(excess_returns) / downside_std * np.sqrt(252), 2)

def calculate_calmar(cumulative: pd.Series):
    """
    Calmar oranı: yıllık getiri / max drawdown
    """
    total_return = cumulative.iloc[-1] - 1
    years = len(cumulative) / 252
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    mdd = abs(max_drawdown(cumulative)) / 100
    return round(annual_return / mdd, 2) if mdd > 0 else np.nan

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

def plot_performance_report(df):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df['date'], df['cumulative'], label='Kümülatif Getiri', color='b')
    ax1.set_ylabel('Kümülatif Getiri')
    ax1.set_xlabel('Tarih')
    ax2 = ax1.twinx()
    drawdown = (df['cumulative'] - df['cumulative'].expanding(min_periods=1).max()) / df['cumulative'].expanding(min_periods=1).max()
    ax2.plot(df['date'], drawdown, label='Drawdown', color='r', alpha=0.3)
    ax2.set_ylabel('Drawdown')
    fig.suptitle('Performans ve Risk Zaman Serisi')
    fig.legend(loc='upper left')
    plt.show()

def generate_performance_report():
    df = calculate_returns()
    sharpe = calculate_sharpe(df['return'])
    sortino = calculate_sortino(df['return'])
    drawdown = max_drawdown(df['cumulative'])
    calmar = calculate_calmar(df['cumulative'])
    success = success_ratio(df)
    plot_performance_report(df)
    return {
        "total_return": round(df['cumulative'].iloc[-1] - 1, 4),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": drawdown,
        "success_rate": success
    }

def rolling_metrics(df, window=60, risk_free_rate=0.01):
    """
    Rolling Sharpe, Sortino, Calmar oranlarını hesaplar.
    """
    roll_sharpe = df['return'].rolling(window).apply(lambda x: calculate_sharpe(x, risk_free_rate), raw=False)
    roll_sortino = df['return'].rolling(window).apply(lambda x: calculate_sortino(x, risk_free_rate), raw=False)
    roll_calmar = df['cumulative'].rolling(window).apply(lambda x: calculate_calmar(x), raw=False)
    return roll_sharpe, roll_sortino, roll_calmar

def plot_benchmark_comparison(df, benchmark_df):
    plt.figure(figsize=(12,6))
    plt.plot(df['date'], df['cumulative'], label='Portföy')
    plt.plot(benchmark_df['date'], benchmark_df['cumulative'], label='Benchmark', linestyle='--')
    plt.title('Portföy vs Benchmark Kümülatif Getiri')
    plt.legend()
    plt.show()

def plot_drawdown_events(df):
    drawdown = (df['cumulative'] - df['cumulative'].expanding(min_periods=1).max()) / df['cumulative'].expanding(min_periods=1).max()
    plt.figure(figsize=(12,4))
    plt.plot(df['date'], drawdown, color='red', label='Drawdown')
    min_idx = drawdown.idxmin()
    plt.scatter(df['date'].iloc[min_idx], drawdown.iloc[min_idx], color='black', label='En Derin Drawdown')
    plt.title('Drawdown Olayları')
    plt.legend()
    plt.show()

def plot_risk_return_scatter(perf_dict):
    tickers = list(perf_dict.keys())
    returns = [v['mean_return'] for v in perf_dict.values()]
    risks = [v['std'] for v in perf_dict.values()]
    plt.figure(figsize=(8,6))
    plt.scatter(risks, returns)
    for i, t in enumerate(tickers):
        plt.annotate(t, (risks[i], returns[i]))
    plt.xlabel('Risk (Std)')
    plt.ylabel('Getiri (Ortalama)')
    plt.title('Risk-Getiri Dağılımı')
    plt.show()

def export_html_report(df, perf_report, filename='performance_report.html'):
    html = df.to_html(index=False)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('<h1>Performans Raporu</h1>')
        for k, v in perf_report.items():
            f.write(f'<b>{k}:</b> {v}<br>')
        f.write(html)
    print(f'HTML rapor kaydedildi: {filename}')

def generate_full_performance_report(benchmark_df=None, perf_dict=None):
    df = calculate_returns()
    perf_report = generate_performance_report()
    # Rolling metrikler
    roll_sharpe, roll_sortino, roll_calmar = rolling_metrics(df)
    plt.figure(figsize=(12,6))
    plt.plot(df['date'], roll_sharpe, label='Rolling Sharpe')
    plt.plot(df['date'], roll_sortino, label='Rolling Sortino')
    plt.plot(df['date'], roll_calmar, label='Rolling Calmar')
    plt.title('Rolling Risk Metrikleri')
    plt.legend()
    plt.show()
    # Benchmark karşılaştırma
    if benchmark_df is not None:
        plot_benchmark_comparison(df, benchmark_df)
    # Drawdown olayları
    plot_drawdown_events(df)
    # Risk-getiri dağılımı
    if perf_dict is not None:
        plot_risk_return_scatter(perf_dict)
    # HTML rapor
    export_html_report(df, perf_report)
    return perf_report 