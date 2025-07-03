from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from core.portfolio_manager import buy_stock, sell_stock, get_portfolio
import os

app = FastAPI()

@app.get("/api/portfolio")
def api_get_portfolio():
    return JSONResponse(get_portfolio())

@app.post("/api/buy")
async def api_buy(request: Request):
    data = await request.json()
    symbol = data["symbol"]
    price = data["price"]
    quantity = data["quantity"]
    buy_stock(symbol, price, quantity)
    return {"status": "ok"}

@app.post("/api/sell")
async def api_sell(request: Request):
    data = await request.json()
    symbol = data["symbol"]
    quantity = data["quantity"]
    sell_stock(symbol, quantity)
    return {"status": "ok"}

@app.get("/")
def serve_dashboard():
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "templates", "portfolio_dashboard.html")
    with open(dashboard_path, encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html) 