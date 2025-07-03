# Learning from trade results implementation will go here 

import json
import os

def update_weights_after_result(result, indicators_used, weights_path="weights.json"):
    if not os.path.exists(weights_path):
        weights = {
            "RSI": 1.0,
            "MACD": 1.0,
            "SuperTrend": 1.0,
            "MultiTimeframe": 1.5,
            "SupportResistance": 1.2,
            "News": 0.6
        }
    else:
        with open(weights_path, "r") as f:
            weights = json.load(f)
    adjustment = -0.1 if result == "loss" else 0.05
    for ind in indicators_used:
        if ind in weights:
            weights[ind] += adjustment
            weights[ind] = max(0.1, min(weights[ind], 3.0))
    with open(weights_path, "w") as f:
        json.dump(weights, f, indent=2) 