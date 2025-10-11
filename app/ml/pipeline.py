
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class SeriesWindow:
    def __init__(self, lookback=60, horizon=1):
        self.lookback = lookback
        self.horizon = horizon

    def make(self, series: np.ndarray):
        X, y = [], []
        for i in range(self.lookback, len(series) - self.horizon + 1):
            X.append(series[i-self.lookback:i])
            y.append(series[i+self.horizon-1])
        return np.array(X), np.array(y)

def train_val_split(df, ratio=0.8):
    n = len(df)
    cut = int(n * ratio)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def scale_fit_transform(train, val):
    scaler = MinMaxScaler()
    train_s = scaler.fit_transform(train)
    val_s = scaler.transform(val)
    return scaler, train_s, val_s
