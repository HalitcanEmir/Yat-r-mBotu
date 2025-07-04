from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from investment_bot.simulation_engine import SimulationEngine
from investment_bot.main import config

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Bot engine'i başlat (veya mevcut bir engine'i kullan)
engine = SimulationEngine(config)

@app.get('/api/portfolio')
def get_portfolio():
    # Son portföy durumu (nakit, pozisyonlar, toplam değer)
    if engine.portfolio_manager.history:
        return engine.portfolio_manager.history[-1]
    return {}

@app.get('/api/trades')
def get_trades():
    # Tüm trade logları (al/sat işlemleri)
    return [log for log in engine.logger.get_logs() if log['event'] == 'STEP']

@app.get('/api/prices')
def get_prices():
    # Son fiyatlar
    return {ticker: float(df['Close'].iloc[-1]) for ticker, df in engine.data_manager.data.items() if not df.empty}

@app.get('/api/summary')
def get_summary():
    report = engine.report_generator.generate(
        engine.portfolio_manager.history,
        engine.risk_manager.compute_risk_metrics(engine.portfolio_manager.history),
        engine.logger.get_logs()
    )
    return report 