from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_home, name='dashboard_home'),
    path('api/portfolio/', views.api_portfolio, name='api_portfolio'),
    path('api/performance/', views.api_performance, name='api_performance'),
] 