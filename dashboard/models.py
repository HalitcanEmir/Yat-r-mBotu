from django.db import models

# Create your models here.

class Trade(models.Model):
    ticker = models.CharField(max_length=16)
    action = models.CharField(max_length=8)
    price = models.FloatField()
    amount = models.IntegerField()
    result = models.CharField(max_length=16, blank=True, null=True)
    profit = models.FloatField(default=0)
    date = models.DateTimeField()

    def __str__(self):
        return f"{self.ticker} {self.action} {self.amount} @{self.price} ({self.date})"

class Indicator(models.Model):
    ticker = models.CharField(max_length=16)
    indicator = models.CharField(max_length=32)
    value = models.FloatField()
    date = models.DateTimeField()

    def __str__(self):
        return f"{self.ticker} {self.indicator} {self.value} ({self.date})"

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
