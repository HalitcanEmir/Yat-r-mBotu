# simulation_engine.py

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from indicators.rsi import calculate_rsi
from indicators.macd import calculate_macd

class DataManager:
    def __init__(self, config):
        """Veri yolları, hisse listesi, tarih aralığı gibi parametreleri alır."""
        self.config = config
        self.tickers = config.get('tickers', ['AAPL'])
        self.start_date = pd.to_datetime(config.get('start_date', '2020-01-01'))
        self.end_date = pd.to_datetime(config.get('end_date', '2023-12-31'))
        self.data = {}  # {ticker: DataFrame}
        self.macro_data = None
        self.dates = []

    def load_data(self):
        """Tüm fiyat, gösterge ve makro verilerini yükler."""
        for ticker in self.tickers:
            # Örnek: CSV veya yfinance'dan veri çekilebilir
            # Burada örnek olarak CSV'den yükleme
            data_path = self.config.get('data_path', f'data/{ticker}.csv')
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, parse_dates=['Date'])
            else:
                # Dummy veri üret (örnek)
                dates = pd.date_range(self.start_date, self.end_date)
                df = pd.DataFrame({
                    'Date': dates,
                    'Close': np.random.normal(100, 10, len(dates)),
                    'Open': np.random.normal(100, 10, len(dates)),
                    'High': np.random.normal(100, 10, len(dates)),
                    'Low': np.random.normal(100, 10, len(dates)),
                    'Volume': np.random.randint(1000000, 2000000, len(dates)),
                })
            df = df[(df['Date'] >= self.start_date) & (df['Date'] <= self.end_date)].reset_index(drop=True)
            self.data[ticker] = df
        # Makro veri yükleme (örnek)
        # self.macro_data = pd.read_csv('macro.csv')
        # Tarih listesini oluştur
        self.dates = list(self.data[self.tickers[0]]['Date'])

    def update_indicators(self):
        """Veri üzerinde teknik göstergeleri ve makro güncellemeleri uygular."""
        for ticker, df in self.data.items():
            close = df['Close']
            df['RSI'] = calculate_rsi(close)
            df['MACD'], df['Signal'] = calculate_macd(close)
            df['MA50'] = close.rolling(window=50).mean()
            df['MA200'] = close.rolling(window=200).mean()
            # ATR, VIX, makro göstergeler vs. eklenebilir
            # df['ATR'] = ...
            self.data[ticker] = df

    def get_dates(self):
        """Simülasyonun çalışacağı tarih listesini döndürür."""
        return self.dates

    def get_step_data(self, date):
        """Belirli bir tarih için fiyat, göstergeler ve makro verileri döndürür."""
        step_data = {}
        for ticker, df in self.data.items():
            row = df[df['Date'] == date]
            if not row.empty:
                step_data[ticker] = row.iloc[0].to_dict()
        # Makro veriler de eklenebilir
        # step_data['macro'] = ...
        return step_data

class StrategySelector:
    """
    Piyasa rejimini (trend, yatay, volatil, anomali) belirler ve uygun stratejiyi seçer.
    Göstergeler ve makro verilerle çalışır. Genişletilebilir yapıdadır.
    """
    def __init__(self, config):
        self.config = config
        # Rejim ve strateji eşleştirmeleri burada tanımlanabilir
        self.regime_strategies = {
            'trend': 'trend_following',
            'sideways': 'mean_reversion',
            'volatile': 'volatility_avoidance',
            'anomaly': 'anomaly_detection',
        }

    def determine_regime(self, step_data):
        """
        Göstergelere göre piyasa rejimini belirler.
        Basit örnek: MA ve RSI ile trend/yatay/volatil/anomali ayrımı.
        """
        # Çoklu varlık desteği için örnek olarak ilk varlık alınır
        ticker = list(step_data.keys())[0]
        data = step_data[ticker]
        ma50 = data.get('MA50', np.nan)
        ma200 = data.get('MA200', np.nan)
        rsi = data.get('RSI', np.nan)
        # Basit kurallar (örnek)
        if np.isnan(ma50) or np.isnan(ma200) or np.isnan(rsi):
            return 'sideways'  # Yeterli veri yoksa yatay kabul et
        if abs(ma50 - ma200) / ma200 > 0.03:
            if ma50 > ma200 and rsi > 55:
                return 'trend'
            elif ma50 < ma200 and rsi < 45:
                return 'trend'
        if 40 < rsi < 60:
            return 'sideways'
        if rsi > 80 or rsi < 20:
            return 'anomaly'
        # Volatilite için ATR veya VIX eklenebilir
        return 'volatile'

    def select_strategy(self, regime):
        """
        Rejime göre uygun strateji adını döndürür.
        """
        return self.regime_strategies.get(regime, 'mean_reversion')

class ScoringEngine:
    """
    Teknik, makro ve temel skorları hesaplar. Genişletilebilir, modüler yapıdadır.
    """
    def __init__(self, config):
        self.config = config

    def compute_scores(self, step_data, regime, strategy):
        """
        Teknik, makro ve temel skorları hesaplar. Şimdilik örnek teknik skor: RSI ve MACD ile.
        """
        scores = {}
        for ticker, data in step_data.items():
            rsi = data.get('RSI', np.nan)
            macd = data.get('MACD', np.nan)
            # Basit teknik skor örneği
            if np.isnan(rsi) or np.isnan(macd):
                tech_score = 0
            else:
                tech_score = (rsi - 50) / 50 + np.sign(macd)
            # Makro ve temel skorlar ileride eklenecek
            macro_score = 0
            fundamental_score = 0
            scores[ticker] = {
                'technical': tech_score,
                'macro': macro_score,
                'fundamental': fundamental_score,
            }
        return scores

class DecisionEngine:
    """
    Skorlar, rejim ve stratejiye göre buy/sell/hold kararı verir. Genişletilebilir yapıdadır.
    """
    def __init__(self, config):
        self.config = config

    def decide(self, scores, regime, strategy):
        """
        Skorlar ve rejime göre karar verir. Basit örnek: teknik skor > 0.5 ise buy, < -0.5 ise sell, aksi halde hold.
        """
        decisions = {}
        for ticker, score_dict in scores.items():
            tech_score = score_dict['technical']
            if tech_score > 0.5:
                action = 'buy'
            elif tech_score < -0.5:
                action = 'sell'
            else:
                action = 'hold'
            decisions[ticker] = action
        return decisions

class PortfolioManager:
    """
    Portföy bakiyesi, pozisyonlar ve güncellemeleri yönetir. Genişletilebilir yapıdadır.
    """
    def __init__(self, config):
        self.config = config
        self.cash = config.get('initial_cash', 100000)
        self.positions = {}  # {ticker: {'amount': float, 'avg_price': float}}
        self.history = []    # Portföy geçmişi (her adımda kaydedilebilir)

    def update(self, date, prices):
        """
        Portföy değerini ve geçmişini günceller.
        """
        total_value = self.cash
        for ticker, pos in self.positions.items():
            price = prices.get(ticker, {}).get('Close', 0)
            total_value += pos['amount'] * price
        self.history.append({'date': date, 'total_value': total_value, 'cash': self.cash, 'positions': self.positions.copy()})

    def get_portfolio_value(self, prices):
        value = self.cash
        for ticker, pos in self.positions.items():
            price = prices.get(ticker, {}).get('Close', 0)
            value += pos['amount'] * price
        return value

    def get_position(self, ticker):
        return self.positions.get(ticker, {'amount': 0, 'avg_price': 0})

    def set_position(self, ticker, amount, avg_price):
        self.positions[ticker] = {'amount': amount, 'avg_price': avg_price}

class TradeExecutor:
    """
    Kararları uygular, işlemleri portföye yansıtır. Komisyon, slippage, stop-loss vb. eklenebilir.
    """
    def __init__(self, config, portfolio_manager):
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.commission = config.get('commission', 0.001)  # %0.1 örnek

    def execute(self, decisions, step_data):
        """
        Buy/sell/hold kararlarını uygular, portföyü günceller.
        """
        for ticker, action in decisions.items():
            price = step_data.get(ticker, {}).get('Close', 0)
            pos = self.portfolio_manager.get_position(ticker)
            if action == 'buy' and self.portfolio_manager.cash > price:
                amount = self.config.get('trade_size', 1)
                cost = amount * price * (1 + self.commission)
                if self.portfolio_manager.cash >= cost:
                    new_amount = pos['amount'] + amount
                    new_avg = (pos['amount'] * pos['avg_price'] + amount * price) / new_amount if new_amount > 0 else price
                    self.portfolio_manager.cash -= cost
                    self.portfolio_manager.set_position(ticker, new_amount, new_avg)
            elif action == 'sell' and pos['amount'] > 0:
                amount = min(self.config.get('trade_size', 1), pos['amount'])
                proceeds = amount * price * (1 - self.commission)
                new_amount = pos['amount'] - amount
                self.portfolio_manager.cash += proceeds
                if new_amount > 0:
                    self.portfolio_manager.set_position(ticker, new_amount, pos['avg_price'])
                else:
                    self.portfolio_manager.set_position(ticker, 0, 0)
            # hold için işlem yok

class Logger:
    """
    Tüm önemli olayları, işlemleri ve portföy durumunu kaydeder. Genişletilebilir yapıdadır.
    """
    def __init__(self, config):
        self.config = config
        self.logs = []

    def log(self, date, event_type, details):
        self.logs.append({'date': date, 'event': event_type, 'details': details})

    def get_logs(self):
        return self.logs

class RiskManager:
    """
    Portföy risk metriklerini (MDD, Sharpe, Sortino, vs.) hesaplar.
    """
    def __init__(self, config):
        self.config = config

    def compute_risk_metrics(self, portfolio_history):
        df = pd.DataFrame(portfolio_history)
        if df.empty or 'total_value' not in df:
            return {}
        returns = df['total_value'].pct_change().dropna()
        mdd = self.max_drawdown(df['total_value'])
        sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252) if not returns.empty else 0
        sortino = self.sortino_ratio(returns) if not returns.empty else 0
        return {
            'MDD': mdd,
            'Sharpe': sharpe,
            'Sortino': sortino,
        }

    @staticmethod
    def max_drawdown(values):
        roll_max = np.maximum.accumulate(values)
        drawdown = (values - roll_max) / roll_max
        return drawdown.min()

    @staticmethod
    def sortino_ratio(returns):
        downside = returns[returns < 0]
        denom = downside.std() if not downside.empty else 1e-9
        return returns.mean() / denom * np.sqrt(252)

class ReportGenerator:
    """
    Simülasyon sonunda özet rapor ve temel görselleştirme/istatistik üretir.
    """
    def __init__(self, config):
        self.config = config

    def generate(self, portfolio_history, risk_metrics, logs):
        df = pd.DataFrame(portfolio_history)
        summary = {
            'final_value': df['total_value'].iloc[-1] if not df.empty else 0,
            'start_value': df['total_value'].iloc[0] if not df.empty else 0,
            'return_%': ((df['total_value'].iloc[-1] / df['total_value'].iloc[0] - 1) * 100) if len(df) > 1 else 0,
            'risk_metrics': risk_metrics,
            'num_trades': sum(1 for log in logs if log['event'] == 'trade'),
        }
        return summary

class SimulationEngine:
    def __init__(self, config):
        self.config = config
        self.data_manager = DataManager(config)
        self.strategy_selector = StrategySelector(config)
        self.scoring_engine = ScoringEngine(config)
        self.decision_engine = DecisionEngine(config)
        self.portfolio_manager = PortfolioManager(config)
        self.trade_executor = TradeExecutor(config, self.portfolio_manager)
        self.logger = Logger(config)
        self.risk_manager = RiskManager(config)
        self.report_generator = ReportGenerator(config)

    def run(self):
        """Simülasyonun ana döngüsü. Zaman çizelgesi üzerinde ilerler."""
        self.data_manager.load_data()
        self.data_manager.update_indicators()
        for date in self.data_manager.get_dates():
            data = self.data_manager.get_step_data(date)
            regime = self.strategy_selector.determine_regime(data)
            strategy = self.strategy_selector.select_strategy(regime)
            scores = self.scoring_engine.compute_scores(data, regime, strategy)
            decisions = self.decision_engine.decide(scores, regime, strategy)
            self.trade_executor.execute(decisions, data)
            self.portfolio_manager.update(date, data)
            # Log işlemleri ve portföy durumu
            self.logger.log(date, 'decisions', {'regime': regime, 'strategy': strategy, 'scores': scores, 'decisions': decisions})
            for ticker, action in decisions.items():
                if action in ['buy', 'sell']:
                    self.logger.log(date, 'trade', {'ticker': ticker, 'action': action, 'price': data[ticker]['Close'] if ticker in data else None})
            self.logger.log(date, 'portfolio', {'positions': self.portfolio_manager.positions.copy(), 'cash': self.portfolio_manager.cash})
        # Simülasyon sonunda risk ve rapor
        risk_metrics = self.risk_manager.compute_risk_metrics(self.portfolio_manager.history)
        report = self.report_generator.generate(self.portfolio_manager.history, risk_metrics, self.logger.get_logs())
        self.logger.log('END', 'report', report)
        return report 