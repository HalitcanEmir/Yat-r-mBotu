from django.shortcuts import render
from django.http import JsonResponse
from .models import PortfolioSnapshot, Trade, Indicator
from django.utils import timezone
import json

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
