from django.contrib import admin
from .models import Trade, Portfolio, Balance, Stock

admin.site.register(Trade)
admin.site.register(Portfolio)
admin.site.register(Balance)
admin.site.register(Stock)
