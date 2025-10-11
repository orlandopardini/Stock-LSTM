# app/ml/constants.py
import os

DEFAULT_LOOKBACK = int(os.getenv("LOOKBACK", 60))
DEFAULT_HORIZON  = int(os.getenv("HORIZON", 1))
MODELS_DIR       = os.getenv("MODELS_DIR", "models")
