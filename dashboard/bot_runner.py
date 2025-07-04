import threading
import time
from .models import Trade, Portfolio, Balance, PortfolioValueSnapshot
from .utils import BIST_TOP20, get_rsi, get_bist_price
import yfinance as yf
import pandas as pd
from django.utils import timezone
from investment_bot.strategy.decision_engine import decide_with_prediction
from investment_bot.strategy.score_weights import compute_momentum_score, top10_bias
from investment_bot.news.gemini_news import ask_gemini_sector_allocation
import json
import os

# Basit teknik skor örneği (daha gelişmişi için scoring_engine'den alınabilir)
def compute_technical_score(rsi, macd, signal, ma50, ma200):
    score = 0
    if rsi < 50:
        score += 1
    if macd > signal or abs(macd - signal) < 10:
        score += 1
    if ma50 >= ma200:
        score += 1
    return score

SCORES_PATH = 'dashboard/scores.json'

# Skorları kaydet
def save_scores(scores, prev_scores):
    with open(SCORES_PATH, 'w', encoding='utf-8') as f:
        json.dump({'scores': scores, 'prev_scores': prev_scores, 'timestamp': str(timezone.now())}, f, ensure_ascii=False)

# ML tahminlerini yükle
ML_PRED_PATH = 'investment_bot/ml_predictions.csv'
ml_predictions = {}
if os.path.exists(ML_PRED_PATH):
    df = pd.read_csv(ML_PRED_PATH)
    for symbol in df['ticker'].unique():
        symbol_df = df[df['ticker'] == symbol]
        # En güncel tahmini al
        latest = symbol_df.sort_values('date').iloc[-1]
        ml_predictions[symbol] = {
            'pred_6m': latest['pred_6m'],
            'pred_1y': latest['pred_1y']
        }

def run_trading_bot():
    print("Bot başlatıldı.")
    gemini_sector_scores = None
    step = 0
    prev_scores = {}
    while True:
        print("Bot döngüsü başladı.")
        # Her 6 ayda bir Gemini'den sektör önerisi al
        if step % 36 == 0:
            sectors = ["Finans", "Sanayi", "Hizmet", "Teknoloji", "Enerji"]
            gemini_sector_scores = ask_gemini_sector_allocation(timezone.now().date(), sectors)
            print(f"[GEMINI] {timezone.now().date()} için sektör önerisi: {gemini_sector_scores}")
        total_value = 0
        total_profit = 0
        balance = Balance.objects.first()
        if not balance:
            balance = Balance.objects.create(amount=100000.0)
        scores = {}
        # 1. Aşama: Skorları ve action'ları hesapla
        for symbol in BIST_TOP20:
            try:
                yf_symbol = f"{symbol}.IS"
                data = yf.Ticker(yf_symbol).history(period="60d")
                if data.empty or len(data['Close']) < 30:
                    continue
                prices = data['Close']
                rsi = get_rsi(prices).iloc[-1]
                macd = prices.ewm(span=12).mean().iloc[-1] - prices.ewm(span=26).mean().iloc[-1]
                signal = prices.ewm(span=9).mean().iloc[-1]
                ma50 = prices.rolling(window=50).mean().iloc[-1]
                ma200 = prices.rolling(window=200).mean().iloc[-1] if len(prices) >= 200 else ma50
                current_price = float(prices.iloc[-1])
                tech_score = compute_technical_score(rsi, macd, signal, ma50, ma200)
                sector_score = None
                if gemini_sector_scores:
                    sector_score = list(gemini_sector_scores.values())[0]['score'] / 100.0
                # ML tahminlerini al
                pred_6m = ml_predictions.get(symbol, {}).get('pred_6m', None)
                pred_1y = ml_predictions.get(symbol, {}).get('pred_1y', None)
                action = decide_with_prediction(
                    tech_score, pred_6m, pred_1y, current_price, sector_score=sector_score,
                    buy_threshold=0.3, sell_threshold=-1.0, pred_weight=0.7, sector_thresh=0.7
                )
                scores[symbol] = {
                    'tech_score': tech_score,
                    'sector_score': sector_score,
                    'total_score': tech_score + (sector_score or 0),
                    'action': action,
                    'current_price': current_price
                }
                # DEBUG: Teknik skorun neden sıfır olduğunu görmek için göstergeleri yazdır
                print(f"{symbol}: rsi={rsi}, macd={macd}, signal={signal}, ma50={ma50}, ma200={ma200}")
            except Exception as e:
                print(f"Bot hata: {symbol} - {e}")
        # 2. Aşama: En yüksek skorlu 1 hisseyi seç
        buy_candidates = [(symbol, scores[symbol]['total_score']) for symbol in scores if scores[symbol]['action'] == 'buy' and scores[symbol]['total_score'] > 1.2]
        buy_candidates = sorted(buy_candidates, key=lambda x: x[1], reverse=True)[:1]
        buy_symbols = set([s[0] for s in buy_candidates])
        # 3. Aşama: Alım/satım işlemlerini uygula
        for symbol in BIST_TOP20:
            if symbol not in scores:
                continue
            try:
                current_price = scores[symbol]['current_price']
                action = scores[symbol]['action']
                portfolio, _ = Portfolio.objects.get_or_create(symbol=symbol, defaults={'quantity': 0, 'avg_buy_price': 0.0})
                quantity = 0
                buy_amount = 0
                if symbol in buy_symbols and balance.amount > current_price:
                    ts = scores[symbol]['total_score']
                    if ts > 4.0:
                        buy_amount = balance.amount * 0.05  # %5
                    elif ts > 3.0:
                        buy_amount = balance.amount * 0.03  # %3
                    elif ts > 2.0:
                        buy_amount = balance.amount * 0.01  # %1
                    elif ts > 1.2:
                        buy_amount = balance.amount * 0.005  # %0.5
                    else:
                        buy_amount = 0
                    quantity = int(buy_amount // current_price)
                # DEBUG: Her hisse için karar ve skorları yazdır
                print(f"{symbol}: action={action}, tech_score={scores[symbol]['tech_score']}, pred_score={(float(ml_predictions.get(symbol, {}).get('pred_6m', 0)) - current_price) / current_price if ml_predictions.get(symbol, {}).get('pred_6m', None) is not None and current_price > 0 else 0}, sector_score={scores[symbol]['sector_score']}, total_score={scores[symbol]['total_score']}, quantity={quantity}, buy_amount={buy_amount}, current_price={current_price}, balance={balance.amount}")
                if quantity > 0:
                    new_total = portfolio.quantity + quantity
                    new_avg = ((portfolio.quantity * portfolio.avg_buy_price) + (quantity * current_price)) / new_total if new_total > 0 else current_price
                    portfolio.quantity = new_total
                    portfolio.avg_buy_price = new_avg
                    portfolio.save()
                    balance.amount -= current_price * quantity
                    balance.save()
                    Trade.objects.create(symbol=symbol, trade_type='BUY', quantity=quantity, price=current_price, is_bot=True)
                    print(f"Bot ALDI: {quantity} {symbol} @{current_price}")
                elif action == 'sell' and portfolio.quantity > 0:
                    profit_loss = (current_price - portfolio.avg_buy_price) * portfolio.quantity
                    balance.amount += current_price * portfolio.quantity
                    balance.save()
                    Trade.objects.create(symbol=symbol, trade_type='SELL', quantity=portfolio.quantity, price=current_price, profit_loss=profit_loss, is_bot=True)
                    print(f"Bot SATTI: {portfolio.quantity} {symbol} @{current_price} Kar/Zarar: {profit_loss}")
                    portfolio.delete()
                total_value += portfolio.quantity * current_price
                total_profit += (current_price - portfolio.avg_buy_price) * portfolio.quantity
            except Exception as e:
                print(f"Bot hata: {symbol} - {e}")
        PortfolioValueSnapshot.objects.create(date=timezone.now(), total_value=total_value + balance.amount, total_profit=total_profit)
        print(f"[SNAPSHOT] Portföy Değeri: {total_value + balance.amount} TL, Kar/Zarar: {total_profit} TL")
        # Skorları kaydet
        save_scores(scores, prev_scores)
        prev_scores = scores
        print("Bot 1 dakika uyuyor...")
        step += 1
        time.sleep(60)

def start_bot_in_thread():
    t = threading.Thread(target=run_trading_bot, daemon=True)
    t.start() 