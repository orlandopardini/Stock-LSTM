
from datetime import timedelta
import numpy as np
import pandas as pd

def rolling_mae(real: pd.Series, pred: pd.Series, window=20):
    diff = (real - pred).abs()
    return diff.rolling(window).mean().iloc[-1]

def needs_retrain(recent_mae: float, baseline_mae: float, threshold=1.25) -> bool:
    if baseline_mae is None:
        return False
    return recent_mae > threshold * baseline_mae
