# TODO: Botun ana başlatıcısı burada olacak 

from core.data_loader import get_stock_data

if __name__ == "__main__":
    df = get_stock_data("AAPL")
    print(df.tail()) 