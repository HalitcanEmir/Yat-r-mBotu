from django.shortcuts import render
from django.http import JsonResponse
from .models import PortfolioSnapshot, Trade
from django.utils import timezone

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
