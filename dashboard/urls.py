from django.urls import path
from .views import BuySellView, portfolio_view, trade_history_view, api_recent_trades, api_performance, home_view

urlpatterns = [
    path('', home_view, name='dashboard_home'),
    path('buy-sell/', BuySellView.as_view(), name='buy_sell'),
    path('portfolio/', portfolio_view, name='portfolio'),
    path('trade-history/', trade_history_view, name='trade_history'),
    path('api/recent_trades/', api_recent_trades, name='api_recent_trades'),
    path('api/performance/', api_performance, name='api_performance'),
] 