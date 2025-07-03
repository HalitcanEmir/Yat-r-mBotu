# TODO: Botun ana başlatıcısı burada olacak 

from core.data_loader import get_stock_data
from core.macro_signals import get_macro_score
from core.decision_engine import make_detailed_decision

if __name__ == "__main__":
    df = get_stock_data("AAPL")
    macro_score = get_macro_score()
    result = make_detailed_decision(df, macro_score)
    print("Karar:", result["decision"], "| Güven:", result["confidence"]) 