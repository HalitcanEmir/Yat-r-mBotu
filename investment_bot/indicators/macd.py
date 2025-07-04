# MACD indicator implementation will go here 

import numpy as np
import pandas as pd

def ema(arr, span):
    alpha = 2 / (span + 1)
    ema_arr = np.zeros_like(arr)
    ema_arr[0] = arr[0]
    for i in range(1, len(arr)):
        ema_arr[i] = alpha * arr[i] + (1 - alpha) * ema_arr[i-1]
    return ema_arr

def calculate_macd(close, fast=12, slow=26, signal=9):
    close = np.asarray(close)
    exp1 = ema(close, fast)
    exp2 = ema(close, slow)
    macd = exp1 - exp2
    signal_line = ema(macd, signal)
    # Return as pandas Series for index compatibility
    return pd.Series(macd, index=range(len(macd))), pd.Series(signal_line, index=range(len(signal_line))) 