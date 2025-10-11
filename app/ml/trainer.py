import os
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
import os, json, glob, joblib, numpy as np, pandas as pd
from .model_zoo import build_model, MODEL_NAMES
from .constants import DEFAULT_LOOKBACK, DEFAULT_HORIZON, MODELS_DIR
from .eval import metrics_from_series, rolling_backtest_1step
from ..models import db, PrecoDiario, ModelRegistry  # importa seu SQLAlchemy
import time
from sqlalchemy.exc import OperationalError
from sklearn.preprocessing import MinMaxScaler
from ..utils.timing import Stopwatch
from ..monitoring import RETRAIN_COUNT, RETRAIN_DURATION


DEFAULT_LOOKBACK = int(os.getenv("LOOKBACK", 60))
DEFAULT_HORIZON = int(os.getenv("HORIZON", 1))
MODELS_DIR = os.getenv("MODELS_DIR", "models")

# --- PATCH: utilitários de split/escala/janelas ---

def _add_registry_with_retry(reg, tries=6):
    for i in range(tries):
        try:
            db.session.add(reg); db.session.commit(); return
        except OperationalError as e:
            if "database is locked" not in str(e).lower():
                db.session.rollback(); raise
            db.session.rollback(); time.sleep(0.5 * (i + 1))
    db.session.add(reg); db.session.commit()

def _update_winner_with_retry(ticker, winner_version, tries=6):
    for i in range(tries):
        try:
            ModelRegistry.query.filter_by(ticker=ticker, is_winner=True).update({"is_winner": False})
            db.session.query(ModelRegistry).filter(ModelRegistry.version == winner_version).update({"is_winner": True})
            db.session.commit(); return
        except OperationalError as e:
            if "database is locked" not in str(e).lower():
                db.session.rollback(); raise
            db.session.rollback(); time.sleep(0.5 * (i + 1))
    ModelRegistry.query.filter_by(ticker=ticker, is_winner=True).update({"is_winner": False})
    db.session.query(ModelRegistry).filter(ModelRegistry.version == winner_version).update({"is_winner": True})
    db.session.commit()
    try:
        from tensorflow.keras import backend as K
        K.clear_session()
    except Exception:
        pass


def _prepare_series(ticker):
    qs = (PrecoDiario.query.filter_by(ticker=ticker).order_by(PrecoDiario.date.asc()).all())
    if not qs: raise ValueError("no data")
    close = pd.Series([r.close for r in qs], index=pd.to_datetime([r.date for r in qs]))
    close = close.dropna()
    return close

def _train_val_split(close: pd.Series, val_ratio: float = 0.2):
    n = len(close); n_tr = max(10, int((1 - val_ratio) * n))
    return close.iloc[:n_tr].copy(), close.iloc[n_tr:].copy()

def _make_supervised(arr_scaled: np.ndarray, lookback: int, horizon: int):
    X, y = [], []
    for t in range(lookback, len(arr_scaled) - (horizon - 1)):
        X.append(arr_scaled[t - lookback:t])
        y.append(arr_scaled[t + (horizon - 1)])
    if not X: return np.empty((0, lookback, 1)), np.empty((0, 1))
    return np.stack(X), np.stack(y)

def _metrics_from_arrays(y_true, y_pred):
    import numpy as np
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9)))) * 100.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-9
    r2 = float(1.0 - ss_res / ss_tot)
    prev = np.r_[y_true[0], y_true[:-1]]
    acc = float(np.mean(np.sign(y_true - prev) == np.sign(y_pred - prev)))
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2, "accuracy": acc}

def train_all_models_fast(ticker: str,
                          lookback=DEFAULT_LOOKBACK,
                          horizon=DEFAULT_HORIZON,
                          epochs=1,           # <- 1 passada
                          batch_size=32,
                          reuse_if_exists=True):
    """
    1) Se já há vencedor e reuse_if_exists=True → só retorna o vencedor (sem treinar).
    2) Senão, treina os 5 modelos UMA vez cada, avalia na validação e elege o melhor.
    """
    sw = Stopwatch()  # <— cronômetro
    if reuse_if_exists:
        rec = (ModelRegistry.query.filter_by(ticker=ticker, is_winner=True)
               .order_by(ModelRegistry.registered_at.desc()).first())
        if rec:
            elapsed = sw.stop()
            return {
                "results": [{
                    "model_id": rec.model_id,
                    "model_name": rec.model_name,
                    "metrics": {"mae": rec.mae, "rmse": rec.rmse, "mape": rec.mape, "r2": rec.r2, "accuracy": rec.accuracy},
                    "version": rec.version,
                    "path_model": rec.path_model,
                    "path_scaler": rec.path_scaler,
                    "registered_at": rec.registered_at.isoformat() if rec.registered_at else None
                }],
                "winner": {
                    "model_id": rec.model_id, "model_name": rec.model_name,
                    "metrics": {"mae": rec.mae, "rmse": rec.rmse, "mape": rec.mape, "r2": rec.r2, "accuracy": rec.accuracy},
                    "version": rec.version
                },
                "reused": True,
                "walltime_sec": max(0.1, elapsed)  # <— nunca 0.0
            }
    # Dados
    close = _prepare_series(ticker)
    train, val = _train_val_split(close, val_ratio=0.2)
    scaler = MinMaxScaler()
    s_tr = scaler.fit_transform(train.values.reshape(-1, 1))
    # para validação, concatenamos a cauda do treino para compor janelas
    s_va = scaler.transform(val.values.reshape(-1, 1)) if len(val) else np.empty((0,1))
    Xtr, ytr = _make_supervised(s_tr, lookback, horizon)
    Xva, yva = _make_supervised(np.concatenate([s_tr[-lookback:], s_va], axis=0) if len(val) else s_tr, lookback, horizon)
    if Xtr.size == 0 or Xva.size == 0:
        raise ValueError("série insuficiente para lookback/horizon escolhidos")

    # Para métricas em escala real
    def inv(a): return scaler.inverse_transform(a.reshape(-1,1)).reshape(-1)

    results = []
    for model_id in range(1, 6):
        model = build_model(model_id, input_shape=(lookback,1))
        model.fit(Xtr, ytr, epochs=epochs, batch_size=batch_size, verbose=0)  # 1 passada
        yva_pred = model.predict(Xva, verbose=0).reshape(-1)

        # métricas em escala original
        mets = _metrics_from_arrays(inv(yva.reshape(-1)), inv(yva_pred.reshape(-1)))

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        version = f"{ticker}_{model_id}_{ts}"
        path_model  = os.path.join(MODELS_DIR, f"{version}.keras")
        path_scaler = os.path.join(MODELS_DIR, f"{version}.scaler")
        model.save(path_model); joblib.dump(scaler, path_scaler)

        reg = ModelRegistry(
            ticker=ticker, model_id=model_id, model_name=MODEL_NAMES[model_id],
            version=version, path_model=path_model, path_scaler=path_scaler,
            mae=mets["mae"], rmse=mets["rmse"], mape=mets["mape"], r2=mets["r2"], accuracy=mets["accuracy"],
            params=json.dumps({"lookback": lookback, "horizon": horizon, "epochs": epochs, "batch_size": batch_size}),
            is_winner=False
        )
        _add_registry_with_retry(reg)

        results.append({
            "model_id": model_id, "model_name": MODEL_NAMES[model_id],
            "metrics": mets, "version": version,
            "path_model": path_model, "path_scaler": path_scaler,
            "registered_at": datetime.utcnow().isoformat()+"Z"
        })

    # vencedor por RMSE (desempate por MAE)
    results_sorted = sorted(results, key=lambda r: (r["metrics"]["rmse"], r["metrics"]["mae"]))
    winner = results_sorted[0]
    _update_winner_with_retry(ticker, winner["version"])
    try:
        from tensorflow.keras import backend as K
        K.clear_session()
    except Exception:
        pass
    elapsed = sw.stop()
    return {
        "results": results,
        "winner": winner,
        "reused": False,
        "walltime_sec": max(0.1, elapsed)}  # <— garante minimo 0.1s



def load_best_model(ticker: str):
    from tensorflow import keras
    rec = (ModelRegistry.query.filter_by(ticker=ticker, is_winner=True)
           .order_by(ModelRegistry.registered_at.desc()).first())
    if not rec: raise ValueError("no winner")
    model  = keras.models.load_model(rec.path_model)
    scaler = joblib.load(rec.path_scaler)
    return model, scaler, rec

def scale_fit_transform_for_train(scaler: MinMaxScaler,
                                  train: pd.Series,
                                  val: pd.Series,
                                  lookback: int,
                                  horizon: int):
    """
    Ajusta scaler no TREINO (somente), transforma treino/val, e cria janelas.
    """
    s_tr = train.values.reshape(-1, 1)
    s_va = val.values.reshape(-1, 1) if len(val) else np.empty((0, 1))
    scaler.fit(s_tr)
    tr_s = scaler.transform(s_tr)
    va_s = scaler.transform(s_va) if len(val) else np.empty((0, 1))

    Xtr, ytr = _make_supervised(tr_s, lookback, horizon)
    Xva, yva = _make_supervised(
        np.concatenate([tr_s[-lookback:], va_s], axis=0) if len(val) else tr_s,
        lookback, horizon
    )
    # Se usamos concat acima, Xva inclui janelas que cruzam a borda; mas os alvos yva
    # já fazem sentido porque estão alinhados ao segmento de validação.
    # Caso não tenha validação suficiente, Xva pode ficar vazio (ok).
    return scaler, Xtr, ytr, Xva, yva
# --- FIM PATCH ---

def build_lstm(input_shape):
    m = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)])
    m.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return m


def fit_model(close_series: pd.Series, lookback=DEFAULT_LOOKBACK, horizon=DEFAULT_HORIZON, epochs=15, batch_size=32):
    s = close_series.values.reshape(-1, 1)
    df = pd.DataFrame(s, columns=['close'])
    train, val = train_val_split(df)
    scaler, train_s, val_s = scale_fit_transform(train, val)
    win = SeriesWindow(lookback, horizon)
    Xtr, ytr = win.make(train_s)
    Xva, yva = win.make(val_s)
    model = build_lstm((lookback, 1))
    history = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch_size, verbose=0)
    va_pred = model.predict(Xva, verbose=0)
    mae = np.mean(np.abs(va_pred.flatten() - yva.flatten()))
    rmse = np.sqrt(np.mean((va_pred.flatten() - yva.flatten())**2))
    mape = float(np.mean(np.abs((yva.flatten() - va_pred.flatten()) / (yva.flatten() + 1e-9)))) * 100
    # >>> R² (coeficiente de determinação) <<<
    y_true = yva.flatten()
    y_hat  = va_pred.flatten()
    ss_res = np.sum((y_true - y_hat) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-9  # evita div/0 quando série é quase constante
    r2 = float(1.0 - ss_res / ss_tot)
    y_prev = val_s.flatten()[lookback-1: -1]
    y_true_dir = y_true
    dir_true = np.sign(y_true_dir - y_prev[:len(y_true_dir)])
    dir_pred = np.sign(y_hat - y_prev[:len(y_true_dir)])
    hits = int(np.sum(dir_true == dir_pred))
    accuracy = float(hits) / len(dir_true)
    return model, scaler, {'mae': float(mae), 'rmse': float(rmse), 'mape': float(mape), 'r2':   float(r2), 'hits': hits, 'accuracy': accuracy}


def predict_horizon(model, scaler, close_series: pd.Series, lookback=DEFAULT_LOOKBACK, horizon=DEFAULT_HORIZON):
    s = close_series.values.reshape(-1, 1)
    s_scaled = scaler.transform(s)
    last = s_scaled[-lookback:]
    X = last.reshape(1, lookback, 1)
    pred_scaled = model.predict(X, verbose=0)[0][0]
    return float(scaler.inverse_transform([[pred_scaled]])[0][0])
