from django.db import models
from django.utils import timezone

# Create your models here.

class Stock(models.Model):
    symbol = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return f"{self.symbol} ({self.name})"

class Trade(models.Model):
    TRADE_TYPE_CHOICES = (
        ("BUY", "Alım"),
        ("SELL", "Satım"),
        ("BOT", "Bot İşlemi"),
    )
    symbol = models.CharField(max_length=10)
    trade_type = models.CharField(max_length=4, choices=TRADE_TYPE_CHOICES)
    quantity = models.PositiveIntegerField()
    price = models.FloatField(default=0.0)
    date = models.DateTimeField(default=timezone.now)
    profit_loss = models.FloatField(default=0.0)
    is_bot = models.BooleanField(default=bool(False))

    def __str__(self):
        return f"{self.trade_type} {self.symbol} {self.quantity} adet @{self.price}"

class Portfolio(models.Model):
    symbol = models.CharField(max_length=10)
    quantity = models.PositiveIntegerField()
    avg_buy_price = models.FloatField(default=0.0)

    def __str__(self):
        return f"{self.symbol}: {self.quantity} adet, Ort. Alış: {self.avg_buy_price}"

class Balance(models.Model):
    amount = models.FloatField(default=100000.0)  # Başlangıç bakiyesi 100.000 TL
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Bakiye: {self.amount} TL"

class PortfolioSnapshot(models.Model):
    date = models.DateTimeField()
    total_value = models.FloatField()
    cash = models.FloatField()
    details = models.TextField()

    def __str__(self):
        return f"{self.date} - {self.total_value}"

class News(models.Model):
    query = models.CharField(max_length=128)
    news_text = models.TextField()
    analysis = models.TextField()
    sentiment = models.FloatField()
    created_at = models.DateTimeField()

    def __str__(self):
        return f"{self.query} ({self.created_at})"

class PortfolioValueSnapshot(models.Model):
    date = models.DateTimeField(default=timezone.now)
    total_value = models.FloatField()
    total_profit = models.FloatField()

    def __str__(self):
        return f"{self.date}: {self.total_value} TL, Kar/Zarar: {self.total_profit} TL"
