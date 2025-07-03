from django.contrib import admin
from .models import Trade, Indicator, PortfolioSnapshot, News

admin.site.register(Trade)
admin.site.register(Indicator)
admin.site.register(PortfolioSnapshot)
admin.site.register(News)
