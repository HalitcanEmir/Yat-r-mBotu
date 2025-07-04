from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import Trade, Portfolio, Balance, PortfolioValueSnapshot
from django.utils import timezone
import json
from django.contrib import messages
from .utils import get_bist_price
from django.views import View
import os

# Create your views here.

def dashboard_home(request):
    return render(request, 'dashboard/dashboard.html')

def api_portfolio(request):
    latest = PortfolioSnapshot.objects.order_by('-date').first()
    data = {
        'date': latest.date if latest else None,
        'total_value': latest.total_value if latest else 0,
        'cash': latest.cash if latest else 0,
        'details': latest.details if latest else '{}',
    }
    return JsonResponse(data)

def api_performance(request):
    trades = Trade.objects.all()
    total_profit = sum(t.profit_loss for t in trades)
    total_trades = trades.count()
    return JsonResponse({'total_profit': total_profit, 'total_trades': total_trades})

def api_risk_distribution(request):
    latest = PortfolioSnapshot.objects.order_by('-date').first()
    if latest and latest.details:
        try:
            details = json.loads(latest.details.replace("'", '"'))
        except Exception:
            details = {}
    else:
        details = {}
    data = [{'ticker': k, 'amount': v} for k, v in details.items()]
    return JsonResponse({'distribution': data})

def api_portfolio_timeseries(request):
    snapshots = PortfolioSnapshot.objects.order_by('-date')[:30][::-1]
    data = [{'date': s.date.strftime('%Y-%m-%d %H:%M'), 'total_value': s.total_value} for s in snapshots]
    return JsonResponse({'series': data})

def api_recent_trades(request):
    trades = Trade.objects.all().order_by('-date')[:10]
    data = [
        {
            'symbol': t.symbol,
            'trade_type': t.trade_type,
            'quantity': t.quantity,
            'price': t.price,
            'profit_loss': t.profit_loss,
            'date': t.date.strftime('%Y-%m-%d %H:%M'),
        }
        for t in trades
    ]
    return JsonResponse({'trades': data})

def api_all_trades(request):
    trades = Trade.objects.order_by('-date')[:100]
    data = [
        {
            'ticker': t.ticker,
            'action': t.action,
            'price': t.price,
            'amount': t.amount,
            'profit': t.profit,
            'date': t.date.strftime('%Y-%m-%d %H:%M'),
        } for t in trades
    ]
    return JsonResponse({'trades': data})

class BuySellView(View):
    template_name = 'dashboard/buy_sell.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        symbol = request.POST.get('symbol', '').upper()
        quantity = int(request.POST.get('quantity', 0))
        trade_type = request.POST.get('trade_type', 'BUY')
        price = get_bist_price(symbol)
        if not price:
            messages.error(request, f"{symbol} için fiyat alınamadı.")
            return render(request, self.template_name)
        # Bakiye ve portföy kontrolü
        balance = Balance.objects.first()
        if not balance:
            balance = Balance.objects.create(amount=100000.0)
        if trade_type == 'BUY':
            total_cost = price * quantity
            if balance.amount < total_cost:
                messages.error(request, "Yetersiz bakiye.")
                return render(request, self.template_name)
            # Portföyde var mı?
            portfolio, created = Portfolio.objects.get_or_create(symbol=symbol, defaults={'quantity': 0, 'avg_buy_price': 0.0})
            new_total = portfolio.quantity + quantity
            new_avg = ((portfolio.quantity * portfolio.avg_buy_price) + (quantity * price)) / new_total if new_total > 0 else price
            portfolio.quantity = new_total
            portfolio.avg_buy_price = new_avg
            portfolio.save()
            balance.amount -= total_cost
            balance.save()
            Trade.objects.create(symbol=symbol, trade_type='BUY', quantity=quantity, price=price, is_bot=False)
            messages.success(request, f"{quantity} adet {symbol} alındı.")
        elif trade_type == 'SELL':
            try:
                portfolio = Portfolio.objects.get(symbol=symbol)
            except Portfolio.DoesNotExist:
                messages.error(request, "Portföyde bu hisse yok.")
                return render(request, self.template_name)
            if portfolio.quantity < quantity:
                messages.error(request, "Yeterli hisse yok.")
                return render(request, self.template_name)
            profit_loss = (price - portfolio.avg_buy_price) * quantity
            portfolio.quantity -= quantity
            if portfolio.quantity == 0:
                portfolio.delete()
            else:
                portfolio.save()
            balance.amount += price * quantity
            balance.save()
            Trade.objects.create(symbol=symbol, trade_type='SELL', quantity=quantity, price=price, profit_loss=profit_loss, is_bot=False)
            messages.success(request, f"{quantity} adet {symbol} satıldı. Kar/Zarar: {profit_loss:.2f} TL")
        else:
            messages.error(request, "Geçersiz işlem tipi.")
        return redirect('buy_sell')

# Portföy görüntüleme
def portfolio_view(request):
    portfolio = Portfolio.objects.all()
    portfolio_data = []
    total_value = 0
    total_profit = 0
    for item in portfolio:
        current_price = get_bist_price(item.symbol)
        if current_price is None:
            continue  # Fiyatı çekilemeyen hisseleri atla
        value = item.quantity * current_price
        profit = (current_price - item.avg_buy_price) * item.quantity
        total_value += value
        total_profit += profit
        portfolio_data.append({
            'symbol': item.symbol,
            'quantity': item.quantity,
            'avg_buy_price': item.avg_buy_price,
            'current_price': current_price,
            'value': value,
            'profit': profit,
        })
    balance = Balance.objects.first()
    return render(request, 'dashboard/portfolio.html', {
        'portfolio': portfolio_data,
        'total_value': total_value,
        'total_profit': total_profit,
        'balance': balance.amount if balance else 0,
    })

# İşlem geçmişi
def trade_history_view(request):
    trades = Trade.objects.all().order_by('-date')
    return render(request, 'dashboard/trade_history.html', {'trades': trades})

# Ana sayfa (dashboard)
def home_view(request):
    # Portföy zaman serisi (son 30 snapshot)
    snapshots = PortfolioValueSnapshot.objects.order_by('-date')[:30][::-1]
    # Portföy tablosu
    portfolio = Portfolio.objects.all()
    portfolio_data = []
    for item in portfolio:
        current_price = get_bist_price(item.symbol)
        if current_price is None:
            continue  # Fiyatı çekilemeyen hisseleri atla
        value = item.quantity * current_price
        profit = (current_price - item.avg_buy_price) * item.quantity
        portfolio_data.append({
            'symbol': item.symbol,
            'quantity': item.quantity,
            'avg_buy_price': item.avg_buy_price,
            'current_price': current_price,
            'value': value,
            'profit': profit,
        })
    # Risk dağılımı (her hissenin portföydeki oranı)
    total_value = sum(item['value'] for item in portfolio_data)
    risk_data = [
        {'symbol': item['symbol'], 'value': item['value'], 'ratio': (item['value']/total_value if total_value > 0 else 0)}
        for item in portfolio_data
    ]
    # İşlem geçmişi (son 20 işlem)
    trades = Trade.objects.all().order_by('-date')[:20]
    # Botun düşündükleri: en beğenilen 3 hisse ve yeri değişen hisse
    top3_scores = []
    mover_score = None
    scores_path = os.path.join('dashboard', 'scores.json')
    if os.path.exists(scores_path):
        with open(scores_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            scores = data.get('scores', {})
            prev_scores = data.get('prev_scores', {})
            # En yüksek toplam skora sahip 3 hisse
            sorted_scores = sorted(scores.items(), key=lambda x: x[1].get('total_score', 0), reverse=True)
            top3_scores = [
                {'symbol': s[0], 'total_score': s[1]['total_score'], 'action': s[1]['action']}
                for s in sorted_scores[:3]
            ]
            # Sıralamada yeri değişen bir hisse bul
            if prev_scores:
                prev_sorted = [s[0] for s in sorted(prev_scores.items(), key=lambda x: x[1].get('total_score', 0), reverse=True)]
                curr_sorted = [s[0] for s in sorted_scores]
                for i, sym in enumerate(curr_sorted):
                    if sym in prev_sorted and prev_sorted.index(sym) != i:
                        mover_score = {
                            'symbol': sym,
                            'old_rank': prev_sorted.index(sym)+1,
                            'new_rank': i+1,
                            'total_score': scores[sym]['total_score'],
                            'action': scores[sym]['action']
                        }
                        break
    return render(request, 'dashboard/home.html', {
        'snapshots': snapshots,
        'portfolio': portfolio_data,
        'risk_data': risk_data,
        'trades': trades,
        'top3_scores': top3_scores,
        'mover_score': mover_score,
    })
