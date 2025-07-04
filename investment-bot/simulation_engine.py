# simulation_engine.py

import pandas as pd
import numpy as np
import os
import yaml
import json
from datetime import datetime, timedelta
from indicators.rsi import calculate_rsi
from indicators.macd import calculate_macd
import yfinance as yf
from strategy.decision_engine import decide_with_prediction
from strategy.score_weights import compute_momentum_score, top10_bias
from news.gemini_news import ask_gemini_sector_allocation

# Yardımcı fonksiyonlar

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_csv(path):
    return pd.read_csv(path)

DEBUG = True
TEST_MACRO_DATA = {
    "2022-01-03": {"VIX": 18.5, "M2": 21000, "FED": 0.05, "PMI": 52.4},
    "2022-01-04": {"VIX": 19.2, "M2": 21050, "FED": 0.05, "PMI": 51.8},
    "2022-01-05": {"VIX": 20.1, "M2": 21100, "FED": 0.05, "PMI": 51.0},
}

class DataManager:
    def __init__(self, config, macro_data=None):
        """Veri yolları, hisse listesi, tarih aralığı gibi parametreleri alır."""
        self.config = config
        self.tickers = config.get('tickers', ['AAPL'])
        self.start_date = pd.to_datetime(config.get('start_date', '2020-01-01'))
        self.end_date = pd.to_datetime(config.get('end_date', '2023-12-31'))
        self.data = {}  # {ticker: DataFrame}
        self.macro_data = macro_data if macro_data is not None else (TEST_MACRO_DATA if DEBUG else None)
        self.dates = []
        self.sector_map = {}  # {ticker: sector}
        self._fetch_sectors()
        self.predictions = self._load_predictions()

    def _fetch_sectors(self):
        for ticker in self.tickers:
            try:
                info = yf.Ticker(ticker).info
                sector = info.get('sector', 'Unknown')
                self.sector_map[ticker] = sector
                if DEBUG:
                    print(f"[SECTOR] {ticker}: {sector}")
            except Exception as e:
                self.sector_map[ticker] = 'Unknown'
                print(f"[WARN] Could not fetch sector for {ticker}: {e}")

    def load_data(self):
        """Fiyat verisini dosyadan veya yfinance ile yükler, cache eder."""
        for ticker in self.tickers:
            data_path = self.config.get('data_path')
            cache_dir = 'data/raw/'
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = f"{cache_dir}{ticker}_{self.start_date.date()}_{self.end_date.date()}.csv"
            df = None
            if data_path and os.path.exists(data_path):
                df = pd.read_csv(data_path, parse_dates=['Date'])
                for base in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    matches = [c for c in df.columns if base in c]
                    if matches:
                        df = df.rename(columns={matches[0]: base})
                    if base in df.columns:
                        df[base] = pd.to_numeric(df[base], errors='coerce')
            elif os.path.exists(cache_file):
                df = pd.read_csv(cache_file, parse_dates=['Date'])
                for base in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    matches = [c for c in df.columns if base in c]
                    if matches:
                        df = df.rename(columns={matches[0]: base})
                    if base in df.columns:
                        df[base] = pd.to_numeric(df[base], errors='coerce')
            else:
                try:
                    print(f"[INFO] Fetching {ticker} from yfinance: {self.start_date.date()} to {self.end_date.date()}")
                    yf_df = yf.download(ticker, start=self.start_date.date(), end=self.end_date.date())
                    if yf_df is not None and not yf_df.empty:
                        print(f"[DEBUG] yfinance raw columns: {yf_df.columns}")
                        print(f"[DEBUG] yfinance raw dtypes: {yf_df.dtypes}")
                        print(f"[DEBUG] yfinance head:\n{yf_df.head()}")
                        # Eğer MultiIndex varsa düzleştir
                        if isinstance(yf_df.columns, pd.MultiIndex):
                            yf_df.columns = ['_'.join([str(i) for i in col if i]) for col in yf_df.columns.values]
                            print(f"[DEBUG] Flattened columns: {yf_df.columns}")
                    if yf_df is None or yf_df.empty:
                        print(f"[WARN] No data returned for {ticker} from yfinance. Skipping.")
                        continue
                    yf_df = yf_df.reset_index()
                    # Sütun isimlerini normalize et
                    col_map = {}
                    for base in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        matches = [c for c in yf_df.columns if base in c]
                        if matches:
                            col_map[matches[0]] = base
                    yf_df = yf_df.rename(columns=col_map)
                    print(f"[DEBUG] After column normalization: {yf_df.columns}")
                    print(f"[DEBUG] After normalization head:\n{yf_df.head()}")
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        if col in yf_df.columns:
                            yf_df[col] = pd.to_numeric(yf_df[col], errors='coerce')
                    yf_df.to_csv(cache_file, index=False)
                    df = yf_df
                except Exception as e:
                    print(f"[WARN] Could not fetch {ticker} from yfinance: {e}. Skipping.")
                    continue
            if df is None:
                print(f"[WARN] No data loaded for {ticker}. Skipping.")
                continue
            else:
                df = df[(df['Date'] >= self.start_date) & (df['Date'] <= self.end_date)].reset_index(drop=True)
                self.data[ticker] = df
        # Tarih listesini oluştur
        self.dates = list(self.data[self.tickers[0]]['Date'])

    def update_indicators(self):
        """Veri üzerinde teknik göstergeleri ve makro güncellemeleri uygular. Veri tipi ve shape kontrolleri ile güvenli."""
        for ticker, df in self.data.items():
            try:
                print(f"[DEBUG] DataFrame columns for {ticker}: {df.columns}")
                print(f"[DEBUG] DataFrame head for {ticker}:\n{df.head()}")
                close_col = 'Close'
                if close_col not in df.columns:
                    # 'Close' geçen ilk sütunu bul
                    close_candidates = [c for c in df.columns if 'Close' in c]
                    if close_candidates:
                        close_col = close_candidates[0]
                        print(f"[INFO] Using column '{close_col}' as close price for {ticker}.")
                    else:
                        print(f"[ERROR] No 'Close' column found for {ticker}. Skipping.")
                        continue
                close = df[close_col]
                # Tip ve shape kontrolü
                if not isinstance(close, (pd.Series, np.ndarray)):
                    print(f"[ERROR] {ticker}: 'Close' is not a Series or ndarray, got {type(close)}. Skipping.")
                    continue
                if isinstance(close, np.ndarray):
                    close = pd.Series(close)
                if len(close) == 0:
                    print(f"[ERROR] {ticker}: 'Close' series is empty. Skipping.")
                    continue
                # NaN kontrolü
                if close.isnull().all():
                    print(f"[ERROR] {ticker}: 'Close' series is all NaN. Skipping.")
                    continue
                # Göstergeler
                df['RSI'] = calculate_rsi(close)
                macd, signal = calculate_macd(close)
                df['MACD'] = macd
                df['Signal'] = signal
                df['MA50'] = close.rolling(window=50).mean()
                df['MA200'] = close.rolling(window=200).mean()
                # ATR, VIX, makro göstergeler vs. eklenebilir
                # df['ATR'] = ...
                self.data[ticker] = df
            except Exception as e:
                print(f"[FATAL] Indicator calculation failed for {ticker}: {e}")
                import traceback
                traceback.print_exc()
                print(f"[WARN] Skipping {ticker} for indicator calculation.")
                continue

    def get_dates(self):
        """Simülasyonun çalışacağı tarih listesini döndürür."""
        return self.dates

    def get_step_data(self, date):
        """Belirli bir tarih için fiyat, göstergeler ve makro verileri döndürür."""
        step_data = {}
        for ticker, df in self.data.items():
            row = df[df['Date'] == date]
            if not row.empty:
                d = row.iloc[0].to_dict()
                # Tahminleri ekle
                pred = self.get_prediction(ticker, date)
                d['pred_6m'] = pred['pred_6m']
                d['pred_1y'] = pred['pred_1y']
                step_data[ticker] = d
        # Makro veriler de eklenir
        macro = {}
        date_str = str(date)[:10]
        if self.macro_data and date_str in self.macro_data:
            macro = self.macro_data[date_str]
        step_data['macro'] = macro
        return step_data

    def compute_sector_returns(self, lookback_days=252):
        """Her sektörün son 1 yıl getirisine göre normalize skorunu döndürür."""
        sector_returns = {}
        for ticker, df in self.data.items():
            sector = self.sector_map.get(ticker, 'Unknown')
            if len(df) < lookback_days:
                continue
            start_price = df['Close'].iloc[-lookback_days] if len(df) > lookback_days else df['Close'].iloc[0]
            end_price = df['Close'].iloc[-1]
            if start_price > 0:
                ret = (end_price - start_price) / start_price
                if sector not in sector_returns:
                    sector_returns[sector] = []
                sector_returns[sector].append(ret)
        # Sektör bazında ortalama getiri
        sector_avg = {s: np.mean(v) for s, v in sector_returns.items() if v}
        if not sector_avg:
            return {}
        max_ret = max(sector_avg.values())
        # Normalize et (en iyi sektör 1.0, diğerleri oransal)
        sector_scores = {s: (v / max_ret if max_ret != 0 else 0) for s, v in sector_avg.items()}
        if DEBUG:
            print(f"[SECTOR SCORES] {sector_scores}")
        return sector_scores

    def _load_predictions(self):
        pred_path = self.config.get('ml_predictions_path')
        if pred_path and os.path.exists(pred_path):
            df = pd.read_csv(pred_path)
            # Beklenen kolonlar: ticker, date, pred_6m, pred_1y
            preds = {}
            for _, row in df.iterrows():
                ticker = row['ticker']
                date = row['date']
                pred_6m = row.get('pred_6m', None)
                pred_1y = row.get('pred_1y', None)
                if ticker not in preds:
                    preds[ticker] = {}
                preds[ticker][date] = {'pred_6m': pred_6m, 'pred_1y': pred_1y}
            return preds
        return {}

    def get_prediction(self, ticker, date):
        # Tahminler string tarih ile tutuluyor olabilir
        date_str = str(date)[:10]
        if ticker in self.predictions and date_str in self.predictions[ticker]:
            return self.predictions[ticker][date_str]
        return {'pred_6m': None, 'pred_1y': None}

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
    def __init__(self, config, strategy_config=None, logger=None, sector_map=None, sector_scores=None):
        self.config = config
        self.strategy_config = strategy_config or {}
        self.logger = logger
        self.sector_map = sector_map or {}
        self.sector_scores = sector_scores or {}
        self.data_manager = DataManager(config)

    def compute_scores(self, step_data, regime, strategy):
        """
        Teknik, makro ve temel skorları hesaplar. Eşikler strategy_config.yaml'dan alınır.
        Sektör skorunu da ekler.
        """
        scores = {}
        top10_list = self.config.get('top10_list', [])
        rsi_buy = self.strategy_config.get('strategy', {}).get('rsi_buy', 30)
        rsi_sell = self.strategy_config.get('strategy', {}).get('rsi_sell', 70)
        use_ma_trend = self.strategy_config.get('strategy', {}).get('ma_trend', True)
        use_atr_filter = self.strategy_config.get('strategy', {}).get('atr_volatility_filter', False)
        atr_threshold = self.strategy_config.get('strategy', {}).get('atr_threshold', 5.0)
        macro_weights = self.strategy_config.get('macro_weights', {"VIX": -0.2, "M2": 0.1, "PMI": 0.1})
        macro_keys = list(macro_weights.keys())
        macro_data = step_data.get('macro', {})
        sector_weight = self.strategy_config.get('sector_weight', 0.5)  # Sektör skorunun ağırlığı
        for ticker, data in step_data.items():
            if ticker == 'macro':
                continue
            rsi = data.get('RSI', np.nan)
            macd = data.get('MACD', np.nan)
            ma50 = data.get('MA50', np.nan)
            ma200 = data.get('MA200', np.nan)
            atr = data.get('ATR', np.nan)
            # RSI tabanlı skor
            tech_score = 0
            if not np.isnan(rsi):
                if rsi < rsi_buy:
                    tech_score += 1
                elif rsi > rsi_sell:
                    tech_score -= 1
            # MA trend filtresi
            if use_ma_trend and not (np.isnan(ma50) or np.isnan(ma200)):
                if ma50 > ma200:
                    tech_score += 0.5
                else:
                    tech_score -= 0.5
            # ATR volatilite filtresi
            if use_atr_filter and not np.isnan(atr):
                if atr > atr_threshold:
                    tech_score = 0
            # Makro skor
            macro_score = 0
            missing_macros = []
            for key in macro_keys:
                if key in macro_data:
                    macro_score += macro_data[key] * macro_weights[key]
                else:
                    missing_macros.append(key)
            if missing_macros and self.logger:
                self.logger.log(
                    data.get('Date', 'N/A'),
                    'WARN',
                    {f'macro_missing_{ticker}': f"Missing macro data for {missing_macros}, macro score set to 0."}
                )
                macro_score = 0
            # Temel skorlar ileride eklenecek
            fundamental_score = 0
            # Sektör skoru
            sector = self.sector_map.get(ticker, 'Unknown')
            sector_score = self.sector_scores.get(sector, 0)
            # Momentum skoru ekle
            df = self.data_manager.data.get(ticker, None)
            momentum_score = compute_momentum_score(df) if df is not None else 0
            tech_score += momentum_score
            # Top10 bias ekle
            tech_score += top10_bias(ticker, top10_list, bias=0.2)
            total_score = tech_score + macro_score + fundamental_score + sector_weight * sector_score
            scores[ticker] = {
                'technical': tech_score,
                'macro': macro_score,
                'fundamental': fundamental_score,
                'sector': sector_score,
                'total': total_score,
            }
        return scores

class DecisionEngine:
    """
    Skorlar, rejim ve stratejiye göre buy/sell/hold kararı verir. Genişletilebilir yapıdadır.
    """
    def __init__(self, config, strategy_config=None):
        self.config = config
        self.strategy_config = strategy_config or {}

    def decide(self, scores, regime, strategy, step_data=None, sector_scores=None):
        decisions = {}
        buy_threshold = self.strategy_config.get('ml', {}).get('buy_score_threshold', 1.0)
        sell_threshold = -buy_threshold
        pred_weight = self.strategy_config.get('ml', {}).get('pred_weight', 0.7)
        sector_thresh = 0.7
        # En iyi sektörleri bul
        best_sectors = []
        if sector_scores:
            max_score = max(sector_scores.values()) if sector_scores else 1.0
            best_sectors = [s for s, v in sector_scores.items() if v >= max_score * 0.9]  # En iyi %10
        for ticker, score_dict in scores.items():
            tech_score = score_dict['technical']
            sector = self.strategy_config.get('sector_map', {}).get(ticker)
            # Sektör skorunu bul
            sector_score = None
            if sector_scores:
                sector = step_data[ticker].get('sector', None) if step_data and ticker in step_data else None
                if sector:
                    sector_score = sector_scores.get(sector, None)
            # Tahminli karar
            if step_data and ticker in step_data:
                pred_6m = step_data[ticker].get('pred_6m', None)
                pred_1y = step_data[ticker].get('pred_1y', None)
                current_price = step_data[ticker].get('Close', None)
                # Sadece en iyi sektörlerde işlem aç
                if sector_score is not None and best_sectors and sector in best_sectors:
                    action = decide_with_prediction(tech_score, pred_6m, pred_1y, current_price, sector_score, buy_threshold, sell_threshold, pred_weight, sector_thresh)
                    decisions[ticker] = action
                    continue
            # Klasik teknik skor ile karar
            if tech_score > buy_threshold:
                action = 'buy'
            elif tech_score < sell_threshold:
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
    Kademeli alım ve kademeli satış destekler.
    """
    def __init__(self, config, portfolio_manager):
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.commission = config.get('commission', 0.001)  # %0.1 örnek

    def execute(self, decisions, step_data, scores=None):
        """
        Buy/sell/hold kararlarını uygular, portföyü günceller.
        Kademeli alım ve kademeli satış uygular.
        """
        for ticker, action in decisions.items():
            price = step_data.get(ticker, {}).get('Close', 0)
            pos = self.portfolio_manager.get_position(ticker)
            portfolio_value = self.portfolio_manager.get_portfolio_value(step_data)
            # Kademeli alım
            if action == 'buy' and self.portfolio_manager.cash > price:
                # Sadece çok uygun hisse için %10'a kadar alım
                total_score = 0
                if scores and ticker in scores and 'total' in scores[ticker]:
                    total_score = scores[ticker]['total']
                if total_score >= 3.5:
                    max_position_pct = 0.10
                else:
                    max_position_pct = 0.03
                max_amount = (portfolio_value * max_position_pct) / price
                amount = self.config.get('trade_size', 1)
                new_amount = min(pos['amount'] + amount, max_amount)
                buy_amount = new_amount - pos['amount']
                if buy_amount > 0:
                    cost = buy_amount * price * (1 + self.commission)
                    if self.portfolio_manager.cash >= cost:
                        avg_price = (pos['amount'] * pos['avg_price'] + buy_amount * price) / new_amount if new_amount > 0 else price
                        self.portfolio_manager.cash -= cost
                        self.portfolio_manager.set_position(ticker, new_amount, avg_price)
            # Kademeli satış
            elif action == 'sell' and pos['amount'] > 0:
                # Pozisyonun %50'sini sat (kademeli azaltma)
                sell_amount = max(1, int(pos['amount'] * 0.5))
                sell_amount = min(sell_amount, pos['amount'])
                proceeds = sell_amount * price * (1 - self.commission)
                new_amount = pos['amount'] - sell_amount
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
    Portföy risk metriklerini (MDD, Sharpe, Sortino, vs.) ve risk kurallarını yönetir.
    """
    def __init__(self, config, strategy_config=None):
        self.config = config
        self.strategy_config = strategy_config or {}

    def get_risk_limits(self):
        risk = self.strategy_config.get('risk', {})
        return {
            'max_position_pct': risk.get('max_position_pct', 0.10),
            'stop_loss_pct': risk.get('stop_loss_pct', 0.07),
            'take_profit_pct': risk.get('take_profit_pct', 0.15),
            'max_daily_loss_pct': risk.get('max_daily_loss_pct', 0.03),
            'max_open_positions': risk.get('max_open_positions', 3),
        }

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

def precheck_files(config, logger=None):
    status = {}
    # Zorunlu dosyalar
    required = {
        'strategy_config.yaml': config.get('strategy_config_path'),
    }
    # Opsiyonel dosyalar
    optional = {
        'macro.json': config.get('macro_data_path'),
        'ml_predictions.csv': config.get('ml_predictions_path'),
        'trades.csv': config.get('trades_path'),
        'portfolio.csv': config.get('portfolio_path'),
    }
    # Fiyat verisi zorunluysa burada kontrol edilebilir
    # ...
    # Zorunlu dosyalar
    for name, path in required.items():
        if not path or not os.path.exists(path):
            msg = f"[✖] {name} not found — simulation cannot start."
            if logger: logger.log('PRECHECK', 'ERROR', {'file': name, 'msg': msg})
            status[name] = False
        else:
            msg = f"[✔] {name} loaded"
            if logger: logger.log('PRECHECK', 'INFO', {'file': name, 'msg': msg})
            status[name] = True
    # Opsiyonel dosyalar
    for name, path in optional.items():
        if not path or not os.path.exists(path):
            msg = f"[✖] {name} not found — optional, will be skipped."
            if logger: logger.log('PRECHECK', 'INFO', {'file': name, 'msg': msg})
            status[name] = False
        else:
            msg = f"[✔] {name} loaded"
            if logger: logger.log('PRECHECK', 'INFO', {'file': name, 'msg': msg})
            status[name] = True
    return status

class SimulationEngine:
    def __init__(self, config):
        self.config = config
        self.data_manager = DataManager(config)
        self.strategy_selector = StrategySelector(config)
        self.logger = Logger(config)
        self.strategy_config = load_yaml(config.get('strategy_config', 'strategy_config.yaml')) if os.path.exists(config.get('strategy_config', 'strategy_config.yaml')) else {}
        self.risk_manager = RiskManager(config, self.strategy_config)
        self.portfolio_manager = PortfolioManager(config)
        self.trade_executor = TradeExecutor(config, self.portfolio_manager)
        self.report_generator = ReportGenerator(config)
        self.dates = []
        self.scoring_engine = None
        self.decision_engine = None
        self.sector_scores = {}

    def run(self):
        self.data_manager.load_data()
        self.data_manager.update_indicators()
        self.dates = self.data_manager.get_dates()
        self.sector_scores = self.data_manager.compute_sector_returns(lookback_days=252)
        self.scoring_engine = ScoringEngine(self.config, self.strategy_config, self.logger, self.data_manager.sector_map, self.sector_scores)
        self.decision_engine = DecisionEngine(self.config, self.strategy_config)
        gemini_sector_scores = None
        for i, date in enumerate(self.dates):
            # Her 6 ayda bir Gemini'den sektör önerisi al
            if i % 126 == 0:  # 252 trading day ~ 1 yıl, 126 ~ 6 ay
                sectors = list(self.sector_scores.keys())
                gemini_sector_scores = ask_gemini_sector_allocation(date, sectors)
                print(f"[GEMINI] {date.date()} için sektör önerisi: {gemini_sector_scores}")
            step_data = self.data_manager.get_step_data(date)
            regime = self.strategy_selector.determine_regime(step_data)
            strategy = self.strategy_selector.select_strategy(regime)
            scores = self.scoring_engine.compute_scores(step_data, regime, strategy)
            # Gemini sektör skorunu teknik skora ekle
            if gemini_sector_scores:
                for ticker in scores:
                    sector = self.data_manager.sector_map.get(ticker, None)
                    if sector and sector in gemini_sector_scores:
                        scores[ticker]['technical'] += gemini_sector_scores[sector]['score'] / 100.0  # normalize etki
            total_scores = {t: v['total'] for t, v in scores.items()}
            decisions = self.decision_engine.decide(scores, regime, strategy, step_data, self.sector_scores)
            self.trade_executor.execute(decisions, step_data, scores)
            self.portfolio_manager.update(date, step_data)
            self.logger.log(date, 'STEP', {'regime': regime, 'strategy': strategy, 'scores': scores, 'decisions': decisions})
        portfolio_history = self.portfolio_manager.history
        risk_metrics = self.risk_manager.compute_risk_metrics(portfolio_history)
        logs = self.logger.get_logs()
        report = self.report_generator.generate(portfolio_history, risk_metrics, logs)
        print(f"\n--- Sektör Skorları ---\n{self.sector_scores}\n")
        return report 