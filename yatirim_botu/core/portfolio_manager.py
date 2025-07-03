# TODO: Portföy takibi ve işlem uygulama 

portfolio = {}

def buy_stock(symbol, price, quantity):
    if symbol not in portfolio:
        portfolio[symbol] = {"quantity": quantity, "buy_price": price}
    else:
        total_qty = portfolio[symbol]["quantity"] + quantity
        avg_price = (
            portfolio[symbol]["quantity"] * portfolio[symbol]["buy_price"]
            + quantity * price
        ) / total_qty
        portfolio[symbol]["quantity"] = total_qty
        portfolio[symbol]["buy_price"] = avg_price

def sell_stock(symbol, quantity):
    if symbol in portfolio:
        portfolio[symbol]["quantity"] -= quantity
        if portfolio[symbol]["quantity"] <= 0:
            del portfolio[symbol]

def get_portfolio():
    # Placeholder: Güncel fiyatları API'den çekmek gerekir
    # Şimdilik buy_price'ı current_price olarak döndürüyoruz
    result = {}
    for symbol, pos in portfolio.items():
        result[symbol] = {
            "quantity": pos["quantity"],
            "buy_price": pos["buy_price"],
            "current_price": pos.get("current_price", pos["buy_price"]),
        }
    return result 