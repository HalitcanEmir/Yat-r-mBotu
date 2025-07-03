# Portfolio manager implementation will go here 

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