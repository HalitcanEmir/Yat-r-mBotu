from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_home, name='dashboard_home'),
    path('api/portfolio/', views.api_portfolio, name='api_portfolio'),
    path('api/performance/', views.api_performance, name='api_performance'),
    path('api/risk/', views.api_risk_distribution, name='api_risk_distribution'),
    path('api/portfolio_series/', views.api_portfolio_timeseries, name='api_portfolio_timeseries'),
    path('api/recent_trades/', views.api_recent_trades, name='api_recent_trades'),
    path('api/all_trades/', views.api_all_trades, name='api_all_trades'),
] 