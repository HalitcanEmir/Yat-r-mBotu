from simulation_engine import SimulationEngine

# Main bot loop will be implemented here
if __name__ == "__main__":
    config = {
        'tickers': [
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "PEP",
            "COST", "ADBE", "CSCO", "TMUS", "AMD", "TXN", "QCOM", "AMGN", "INTC", "HON",
            "SBUX", "AMAT", "INTU", "BKNG", "ISRG", "ADI", "MDLZ", "LRCX", "REGN", "GILD",
            "FISV", "VRTX", "ADP", "MU", "PDD", "MAR", "KDP", "MELI", "CDNS", "CSX",
            "CTAS", "AEP", "ADSK", "MNST", "ORLY", "KLAC", "PCAR", "IDXX", "ROST", "PAYX",
            "MRVL", "XEL", "WBD", "ODFL", "FAST", "CRWD", "BIIB", "DDOG", "TEAM", "SGEN",
            "PANW", "LCID", "ZS", "SPLK", "DLTR", "VRSK", "CHTR", "ALGN", "CPRT", "EXC",
            "CGEN", "FTNT", "EA", "SNPS", "CTSH", "ANSS", "SWKS", "BIDU", "ASML", "NXPI",
            "TTD", "DOCU", "OKTA", "MCHP", "WDAY", "SIRI", "ILMN", "VRSN", "BMRN", "INCY",
            "LULU", "JD", "NTES", "SGEN", "SIRI", "ZM", "PDD", "MRNA", "AAL", "UAL"
        ],
        'start_date': '2021-01-01',
        'end_date': '2024-01-01',
        'initial_cash': 100000,
        'trade_size': 10,
        'commission': 0.001,
        # Dış veri/config dosyaları:
        'trades_path': 'trades.csv',
        'portfolio_path': 'portfolio.csv',
        'strategy_config_path': 'investment-bot/strategy_config.yaml',
        'macro_data_path': 'macro.json',
        'ml_predictions_path': 'ml_predictions.csv',
        'top10_list': ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "PEP", "COST"],
        # 'data_path': 'data/AAPL.csv',  # Gerçek veri için eklenebilir
    }
    engine = SimulationEngine(config)
    report = engine.run()
    print("Simülasyon Sonuç Raporu:")
    for k, v in report.items():
        print(f"{k}: {v}") 