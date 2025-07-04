# RSI indicator implementation will go here 

import numpy as np
import pandas as pd

def calculate_rsi(close, window=14):
    close = np.asarray(close)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Rolling mean with convolution
    kernel = np.ones(window) / window
    avg_gain = np.convolve(gain, kernel, mode='same')
    avg_loss = np.convolve(loss, kernel, mode='same')
    # Avoid division by zero
    rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    # Return as pandas Series for index compatibility
    return pd.Series(rsi, index=range(len(rsi))) 