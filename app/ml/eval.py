import numpy as np
import pandas as pd
from .constants import DEFAULT_LOOKBACK, DEFAULT_HORIZON

def rolling_backtest_1step(model, scaler, close, lookback=60, window=180):
    """
    Usa SEMPRE o scaler fornecido (do modelo vencedor).
    Retorna DataFrame com index de datas e colunas y_true / y_pred (escala real).
    """
    import numpy as np, pandas as pd

    # faixa final (window dias) com contexto de lookback
    if window > len(close): window = len(close)
    start_idx = max(0, len(close) - window - lookback)
    seg = close.iloc[start_idx:].copy()

    s = seg.values.reshape(-1,1)
    s_scaled = scaler.transform(s)

    preds = []
    idxs  = []
    # anda 1 passo por vez no trecho final (após o lookback)
    for t in range(lookback, len(seg)):
        X = s_scaled[t-lookback:t].reshape(1,lookback,1)
        yhat_s = model.predict(X, verbose=0)[0][0]
        yhat = scaler.inverse_transform([[yhat_s]])[0][0]
        preds.append(yhat)
        idxs.append(seg.index[t])

    y_true = seg.iloc[lookback:].values.astype(float)
    df = pd.DataFrame({"y_true": y_true, "y_pred": np.array(preds)}, index=pd.to_datetime(idxs))
    return df


def metrics_from_series(df_pred):
    """
    Aceita colunas com vários nomes:
      y_true | real | true | target
      y_pred | pred | previsto | forecast
    Retorna dict: mae, rmse, mape, r2, accuracy (0..1)
    """
    cols = {c.lower(): c for c in df_pred.columns}
    ycol = None
    for k in ("y_true", "real", "true", "target"):
        if k in cols:
            ycol = cols[k]; break
    pcol = None
    for k in ("y_pred", "pred", "previsto", "forecast"):
        if k in cols:
            pcol = cols[k]; break
    if ycol is None or pcol is None:
        raise ValueError(f"metrics_from_series: colunas não encontradas. "
                         f"Tem: {list(df_pred.columns)} | Esperado: y_true/real e y_pred/pred")
    y    = df_pred[ycol].astype(float).to_numpy()
    yhat = df_pred[pcol].astype(float).to_numpy()
    mae  = float(np.mean(np.abs(y - yhat)))
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    mape = float(np.mean(np.abs((y - yhat) / (y + 1e-9)))) * 100.0
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-9
    r2 = float(1.0 - ss_res / ss_tot)
    prev = np.r_[y[0], y[:-1]]
    acc = float(np.mean(np.sign(y - prev) == np.sign(yhat - prev)))
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2, "accuracy": acc}
