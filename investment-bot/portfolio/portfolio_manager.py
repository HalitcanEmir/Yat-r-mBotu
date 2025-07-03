# Portfolio manager implementation will go here 

import matplotlib.pyplot as plt

def simulate_buy_sell(*args, **kwargs):
    pass  # Logic handled in backtest.py for now 

def uygula_alim_karari(alim_var, alim_skoru, cash, price, stock):
    """
    Alım kararı ve skoru ile portföyü günceller. Alım yapılırsa miktarı belirler ve nakit/lot günceller.
    Dönüş: yeni_cash, yeni_stock (int), miktar (int)
    """
    def al_satis_miktari_belirle(alim_skoru, cash, price):
        if alim_skoru >= 6:
            oran = 0.25
        elif alim_skoru >= 5:
            oran = 0.20
        elif alim_skoru >= 4:
            oran = 0.15
        elif alim_skoru >= 3:
            oran = 0.10
        else:
            oran = 0.05
        miktar = int((cash * oran) // price)
        return max(miktar, 0)

    miktar = 0
    if alim_var and cash >= price:
        miktar = al_satis_miktari_belirle(alim_skoru, cash, price)
        if miktar > 0:
            cash -= price * miktar
            stock += miktar
    return cash, int(stock), int(miktar)

def auto_rebalance_portfolio(portfolio, price_data, performance_data, threshold=0.1):
    """
    Portföydeki hisseleri performansa göre otomatik rebalance eder.
    threshold: en iyi ve en kötü performanslı hisseler arasında min. fark (oran)
    """
    # Performansa göre sıralama
    perf_sorted = sorted(performance_data.items(), key=lambda x: x[1], reverse=True)
    n = len(perf_sorted)
    if n < 2:
        return portfolio  # Rebalance için yeterli çeşit yok
    top, bottom = perf_sorted[0], perf_sorted[-1]
    # Eğer fark yeterince büyükse, en iyiye ekle, en kötüden azalt
    if (top[1] - bottom[1]) > threshold:
        # En kötüden azalt, en iyiye ekle
        worst, best = bottom[0], top[0]
        if portfolio[worst]['stock'] > 0:
            sat_miktar = max(1, int(portfolio[worst]['stock'] * 0.25))
            portfolio[worst]['stock'] -= sat_miktar
            cash = price_data[worst] * sat_miktar
            al_miktar = max(1, int(cash // price_data[best]))
            portfolio[best]['stock'] += al_miktar
    return portfolio

def plot_portfolio_risk(portfolio, price_data, risk_data=None):
    """
    Portföydeki risk dağılımını pasta veya bar grafikle gösterir.
    risk_data: dict, her hisse için risk (ör: volatilite, beta, vs.)
    """
    labels = list(portfolio.keys())
    values = [portfolio[t]['stock'] * price_data[t] for t in labels]
    if risk_data:
        risks = [risk_data.get(t, 1) for t in labels]
        values = [v * r for v, r in zip(values, risks)]
    plt.figure(figsize=(8, 5))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Portföy Risk Dağılımı')
    plt.show() 