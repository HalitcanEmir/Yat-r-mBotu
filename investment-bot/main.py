from simulation_engine import SimulationEngine

# Main bot loop will be implemented here
if __name__ == "__main__":
    config = {
        'tickers': ['AAPL'],
        'start_date': '2022-01-01',
        'end_date': '2022-03-31',
        'initial_cash': 100000,
        'trade_size': 10,
        'commission': 0.001,
        # 'data_path': 'data/AAPL.csv',  # Gerçek veri için eklenebilir
    }
    engine = SimulationEngine(config)
    report = engine.run()
    print("Simülasyon Sonuç Raporu:")
    for k, v in report.items():
        print(f"{k}: {v}") 