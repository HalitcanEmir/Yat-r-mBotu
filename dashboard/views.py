from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import Trade, Portfolio, Balance
from django.utils import timezone
import json
from django.contrib import messages
from .utils import get_bist_price
from django.views import View

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
    trades = Trade.objects.order_by('-date')[:100]
    total_profit = sum(t.profit for t in trades)
    win_trades = [t for t in trades if t.profit > 0]
    win_rate = len(win_trades) / len(trades) * 100 if trades else 0
    data = {
        'total_profit': total_profit,
        'win_rate': win_rate,
        'trade_count': len(trades),
    }
    return JsonResponse(data)

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
    trades = Trade.objects.order_by('-date')[:10]
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
