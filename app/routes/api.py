
from flask import Blueprint, request, jsonify, current_app
from flasgger import swag_from
from datetime import datetime, date
from .. import db
from ..models import PrecoDiario, ResultadoMetricas, RetrainHistory
from ..ml.trainer import fit_model, predict_horizon
from ..ml.trainer import DEFAULT_HORIZON, DEFAULT_LOOKBACK
from ..models import db, PrecoDiario, ResultadoMetricas, ModelRegistry
from ..ml.trainer import DEFAULT_LOOKBACK
from ..ml.eval import rolling_backtest_1step, metrics_from_series
from ..ml.trainer import train_all_models_fast, load_best_model
from ..ml.constants import DEFAULT_LOOKBACK, DEFAULT_HORIZON
from ..monitoring import INFERENCE_LATENCY  # e/ou RETRAIN_COUNT, RETRAIN_DURATION se usar
from ..ml.data import load_close_series
from tensorflow import keras
import joblib, glob, os
import pandas as pd
import yfinance as yf
import os
from pandas.tseries.offsets import BDay
# ==== HELPERS ROBUSTOS ====
import time
import pandas as pd

def _normalize_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normaliza para colunas simples: Open, High, Low, Close, Adj_Close, Volume."""
    if df is None or df.empty:
        return pd.DataFrame()

    # Se vier MultiIndex do Yahoo (('Open','NVDA'), ...)
    if isinstance(df.columns, pd.MultiIndex):
        upper = ticker.upper()
        selected = None
        for level in range(df.columns.nlevels - 1, -1, -1):
            vals = [str(v).upper() for v in df.columns.get_level_values(level)]
            if upper in vals:
                selected = level
                break
        if selected is not None:
            try:
                df = df.xs(upper, axis=1, level=selected, drop_level=True)
            except Exception:
                df = df.swaplevel(selected, 0, axis=1)
                df = df[upper]
        else:
            df.columns = ['_'.join([str(x) for x in c if x is not None]) for c in df.columns]

    # Padroniza nomes
    rename = {}
    for c in df.columns:
        lc = str(c).lower().strip()
        if lc in ("open","high","low","close","volume","adj close","adj_close","adjclose"):
            if "adj" in lc:
                rename[c] = "Adj_Close"
            elif lc == "volume":
                rename[c] = "Volume"
            else:
                rename[c] = lc.capitalize()
    df = df.rename(columns=rename)

    if "Adj_Close" not in df.columns and "Close" in df.columns:
        df["Adj_Close"] = df["Close"]

    keep = [c for c in ["Open","High","Low","Close","Adj_Close","Volume"] if c in df.columns]
    df = df[keep].copy()
    if "Close" in df.columns:
        df = df.dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index)
    return df

def _fetch_yahoo_block(ticker: str, d0, d1):
    import yfinance as yf
    end_exc = pd.Timestamp(d1) + pd.Timedelta(days=1)  # end exclusivo
    # método 1
    df = yf.download(ticker, start=str(d0), end=str(end_exc.date()),
                     auto_adjust=False, progress=False, threads=False)
    if df is not None and not df.empty:
        return df
    # método 2
    df2 = yf.Ticker(ticker).history(start=str(d0), end=str(end_exc.date()),
                                    interval="1d", auto_adjust=False)
    return df2

def _fetch_stooq_block(ticker: str, d0, d1):
    """Tenta Stooq com NVDA.US e NVDA."""
    try:
        from pandas_datareader import data as pdr
    except Exception:
        return pd.DataFrame()

    for code in (f"{ticker}.US", ticker):
        try:
            df = pdr.DataReader(code, "stooq", start=d0, end=d1)
            if df is not None and not df.empty:
                df = df.sort_index()
                return df
        except Exception:
            continue
    return pd.DataFrame()

def _fetch_resilient_yearly(ticker: str, start: str):
    """Baixa por blocos anuais com retries/backoff + fallback Stooq."""
    start_date = pd.to_datetime(start).date()
    end_date = pd.Timestamp.today().date()
    frames = []
    y = start_date.year
    while y <= end_date.year:
        d0 = pd.Timestamp(f"{y}-01-01").date()
        if y == start_date.year and d0 < start_date:
            d0 = start_date
        d1 = pd.Timestamp(f"{y}-12-31").date()
        if d1 > end_date:
            d1 = end_date

        # Yahoo com retries/backoff
        got = False
        last_exc = None
        for i in range(6):
            try:
                df = _fetch_yahoo_block(ticker, d0, d1)
                if df is not None and not df.empty:
                    frames.append(_normalize_ohlcv(df, ticker))
                    got = True
                    break
                last_exc = RuntimeError("Yahoo vazio")
            except Exception as e:
                last_exc = e
            time.sleep(1.5 * (2 ** i))  # backoff

        if not got:
            # Fallback Stooq (tenta NVDA.US e NVDA)
            df = _fetch_stooq_block(ticker, d0, d1)
            if df is not None and not df.empty:
                frames.append(_normalize_ohlcv(df, ticker))
            # se também falhar, apenas pula o bloco

        time.sleep(1.0)  # pausa leve entre blocos
        y += 1

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=0)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out
# ==============================================

api_bp = Blueprint('api', __name__)

API_KEY = os.getenv('API_KEY', 'change-me')
MODELS_DIR = os.getenv('MODELS_DIR', 'models')
DISABLE_API_KEY = os.getenv('DISABLE_API_KEY') == '1'

from pathlib import Path
Path(MODELS_DIR).mkdir(exist_ok=True, parents=True)

from tensorflow import keras
import joblib
import glob

def _auth_ok(req):
    # Em DEV, ou se DISABLE_API_KEY=1, não exige header
    if DISABLE_API_KEY or req.remote_addr in ('127.0.0.1', '::1'):
        return True
    return req.headers.get('X-API-KEY') == API_KEY

def require_basic_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = os.getenv('ADMIN_USER', 'admin')
        pw = os.getenv('ADMIN_PASS')  # se vazio, não exige senha (DEV)
        if not pw:
            return f(*args, **kwargs)
        auth = request.authorization
        if not auth or auth.username != user or auth.password != pw:
            return Response('Auth required', 401, {'WWW-Authenticate': 'Basic realm="Admin"'})
        return f(*args, **kwargs)
    return wrapper

@api_bp.get('/health')
def health():
    return {'ok': True}

@api_bp.post('/update_data')
@swag_from({
    'tags': ['Dados'],
    'parameters': [
        {'name': 'ticker', 'in': 'query', 'schema': {'type': 'string'}, 'required': True},
        {'name': 'start',  'in': 'query', 'schema': {'type': 'string'}, 'required': False},
    ],
    'responses': {200: {'description': 'Atualizado'}}
})
def update_data():
    if not _auth_ok(request):
        return jsonify({'error': 'unauthorized'}), 401

    ticker = request.args.get('ticker', 'NVDA').upper()
    start  = request.args.get('start', '2010-01-01')

    df = _fetch_resilient_yearly(ticker, start)
    if df is None or df.empty:
        return {'ticker': ticker, 'rows_added': 0, 'note': 'no data (Yahoo/Stooq indisponíveis ou rede bloqueada)'}

    added = 0
    for idx, row in df.iterrows():
        d = idx.date()
        rec = PrecoDiario.query.filter_by(ticker=ticker, date=d).first()
        if rec:
            rec.open  = float(row["Open"])  if "Open"  in df.columns else rec.open
            rec.high  = float(row["High"])  if "High"  in df.columns else rec.high
            rec.low   = float(row["Low"])   if "Low"   in df.columns else rec.low
            rec.close = float(row["Close"]) if "Close" in df.columns else rec.close
            rec.adj_close = float(row["Adj_Close"]) if "Adj_Close" in df.columns else rec.adj_close
            rec.volume = int(row["Volume"]) if "Volume" in df.columns else rec.volume
        else:
            db.session.add(PrecoDiario(
                ticker=ticker, date=d,
                open=float(row["Open"]) if "Open" in df.columns else None,
                high=float(row["High"]) if "High" in df.columns else None,
                low=float(row["Low"])   if "Low"  in df.columns else None,
                close=float(row["Close"]) if "Close" in df.columns else None,
                adj_close=float(row["Adj_Close"]) if "Adj_Close" in df.columns else (float(row["Close"]) if "Close" in df.columns else None),
                volume=int(row["Volume"]) if "Volume" in df.columns else 0
            ))
            added += 1
    db.session.commit()

    first = df.index.min().strftime("%Y-%m-%d")
    last  = df.index.max().strftime("%Y-%m-%d")
    return {'ticker': ticker, 'rows_added': added, 'range': [first, last]}

from flask import request, jsonify, current_app
from ..ml.trainer import train_all_models_fast  # use o "fast" como você já está
from ..ml.constants import DEFAULT_LOOKBACK, DEFAULT_HORIZON
from ..utils.timing import Stopwatch  # <-- util do cronômetro (mostrei antes)

# (opcional) só se você criou isso em monitoring.py:
try:
    from ..monitoring import RETRAIN_COUNT, RETRAIN_DURATION
except Exception:
    RETRAIN_COUNT = None
    RETRAIN_DURATION = None

@api_bp.post('/train')
def train():
    payload  = request.get_json(silent=True) or {}
    ticker   = (payload.get('ticker') or request.args.get('ticker') or 'NVDA').upper()
    lookback = int(payload.get('lookback', DEFAULT_LOOKBACK))
    horizon  = int(payload.get('horizon',  DEFAULT_HORIZON))
    epochs   = int(payload.get('epochs',   1))
    batch    = int(payload.get('batch_size', 32))
    force    = bool(payload.get('force', False))

    out = train_all_models_fast(
        ticker, lookback, horizon,
        epochs=epochs, batch_size=batch, reuse_if_exists=not force
    )

    w = out.get("winner", {}) ; m = w.get("metrics", {})
    return {
        "ticker": ticker,
        "version": w.get("version"),
        "winner": w,
        "all_models": out.get("results", []),
        "reused": out.get("reused", False),
        "duration_sec": round(out.get("walltime_sec", 0.0), 2),
        "mae": m.get("mae"), "rmse": m.get("rmse"),
        "mape": m.get("mape"), "r2": m.get("r2"),
        "accuracy": m.get("accuracy"),
    }



# DETALHES DO VENCEDOR (para UI)
@api_bp.get('/models/best')
def best_model():
    ticker = request.args.get('ticker', 'NVDA').upper()
    try:
        _, _, rec = load_best_model(ticker)
    except Exception:
        return jsonify({"error":"no winner"}), 404
    return {
        "ticker": ticker,
        "model_id": rec.model_id,
        "model_name": rec.model_name,
        "version": rec.version,
        "registered_at": rec.registered_at.isoformat(),
        "metrics": {
            "mae": rec.mae, "rmse": rec.rmse, "mape": rec.mape,
            "r2": rec.r2, "accuracy": rec.accuracy
        }
    }

# TABELA RESUMO (para listar os 5 com métricas)
@api_bp.get('/models/summary')
def models_summary():
    ticker = request.args.get('ticker', 'NVDA').upper()
    try:
        rows = (ModelRegistry.query
                .filter_by(ticker=ticker)
                .order_by(ModelRegistry.registered_at.desc())
                .all())
    except Exception as e:
        # tabela ausente ou outro erro → responda vazio com motivo
        return {"ticker": ticker, "models": [], "note": f"{type(e).__name__}: {e}"}, 200
    out = []
    for r in rows:
        out.append({
            "model_id": r.model_id, "model_name": r.model_name, "version": r.version,
            "mae": r.mae, "rmse": r.rmse, "mape": r.mape, "r2": r.r2, "accuracy": r.accuracy,
            "is_winner": r.is_winner, "registered_at": r.registered_at.isoformat() if r.registered_at else None
        })
    return {"ticker": ticker, "models": out}


@api_bp.get('/predict')
@swag_from({'tags': ['Modelo'], 'parameters': [
    {'name': 'ticker', 'in': 'query', 'schema': {'type': 'string'}, 'required': True},
    {'name': 'date', 'in': 'query', 'schema': {'type': 'string'}, 'required': False},
    {'name': 'horizon', 'in': 'query', 'schema': {'type': 'integer'}, 'required': False},
], 'responses': {200: {'description': 'Previsão'}}})
def predict():
    from ..ml.trainer import load_best_model
    from ..ml.constants import DEFAULT_LOOKBACK, DEFAULT_HORIZON
    from ..models import PrecoDiario

    ticker   = request.args.get('ticker','NVDA').upper()
    lookback = int(request.args.get('lookback', DEFAULT_LOOKBACK))
    horizon  = int(request.args.get('horizon',  DEFAULT_HORIZON))

    # carrega melhor modelo; se não houver vencedor, responde claro
    try:
        model, scaler, rec = load_best_model(ticker)
    except ValueError:
        return jsonify(error="no winner model for this ticker; treine primeiro via POST /api/train"), 404

    # lê a série (prioriza adj_close)
    rows = (PrecoDiario.query
            .filter_by(ticker=ticker)
            .order_by(PrecoDiario.date.asc())
            .all())
    if not rows:
        return jsonify(error="no data"), 400

    idx   = pd.to_datetime([r.date for r in rows])
    vals  = [ (r.adj_close if r.adj_close is not None else r.close) for r in rows ]
    close = pd.Series(vals, index=idx, name="close").dropna()

    if len(close) < lookback:
        return jsonify(error=f"insufficient series length ({len(close)}) for lookback={lookback}"), 400

    # última janela
    s = close.values.reshape(-1, 1)
    s_scaled = scaler.transform(s)
    last = s_scaled[-lookback:].reshape(1, lookback, 1)

    yhat_scaled = model.predict(last, verbose=0)[0][0]
    yhat = float(scaler.inverse_transform([[yhat_scaled]])[0][0])

    # último dia disponível e PRÓXIMO DIA ÚTIL
    last_dt = pd.to_datetime(close.index[-1])
    next_dt = (last_dt + BDay(1)).date()    # <- dia útil, não calendário
    t0 = time.perf_counter()
    yhat_scaled = model.predict(last, verbose=0)[0][0]
    dur = time.perf_counter() - t0
    INFERENCE_LATENCY.labels(ticker=ticker, version=rec.version).observe(dur)

    return jsonify({
        "ticker": ticker,
        "version": rec.version,
        "pred": yhat,
        "date_cutoff": last_dt.date().isoformat(),  # último dia presente na base
        "date_next":   next_dt.isoformat()          # próximo dia ÚTIL (use no ponto vermelho)
    })

@api_bp.get('/simulate')
def simulate():
    """
    Prevê 1 dia por vez até a data alvo (ou por 'steps' = dias úteis).
    Query: ticker= NVDA, date=YYYY-MM-DD  (ou steps=250)
    """
    from flask import request, jsonify
    import json, numpy as np, pandas as pd
    from pandas.tseries.offsets import BDay
    from tensorflow.keras.models import load_model
    import joblib
    from ..models import ModelRegistry
    from ..ml.data import load_close_series  # ajuste se seu loader tiver outro nome

    ticker   = (request.args.get('ticker') or 'NVDA').upper()
    date_str = request.args.get('date')
    steps_in = request.args.get('steps', type=int)

    # vencedor mais recente
    rec = (ModelRegistry.query
           .filter_by(ticker=ticker, is_winner=True)
           .order_by(ModelRegistry.registered_at.desc())
           .first())
    if not rec:
        return jsonify(error="no winner model for this ticker"), 400

    close = load_close_series(ticker)  # pandas.Series index de datas
    if close is None or close.empty:
        return jsonify(error="no price series"), 400

    params   = json.loads(rec.params or '{}')
    lookback = int(params.get('lookback', 60))
    last_dt  = pd.to_datetime(close.index.max()).date()

    # resolve passos
    if steps_in and steps_in > 0:
        steps = int(min(steps_in, 252))  # limite de segurança
        target_dt = (pd.Timestamp(last_dt) + BDay(steps)).date()
    else:
        if not date_str:
            return jsonify(error="missing date or steps"), 400
        target_dt = pd.to_datetime(date_str).date()
        if target_dt <= last_dt:
            return jsonify(error="target must be after last available date"), 400
        steps = len(pd.bdate_range(last_dt, target_dt)) - 1
        steps = int(min(steps, 252))

    # carrega modelo e scaler
    model  = load_model(rec.path_model, compile=False)
    scaler = joblib.load(rec.path_scaler)

    s = close.values.astype(float).reshape(-1, 1)
    s_sc = scaler.transform(s)
    if len(s_sc) < lookback:
        return jsonify(error="series shorter than lookback"), 400
    window = s_sc[-lookback:].reshape(1, lookback, 1)

    preds, cur = [], last_dt
    for _ in range(steps):
        yhat_sc = float(model.predict(window, verbose=0).reshape(-1)[0])
        yhat    = float(scaler.inverse_transform([[yhat_sc]])[0, 0])
        cur     = (pd.bdate_range(cur, periods=2)[-1]).date()  # próximo dia útil
        preds.append({"date": cur.isoformat(), "pred": yhat})
        window = np.concatenate([window[:, 1:, :], [[[yhat_sc]]]], axis=1)

    return jsonify({
        "ticker": ticker,
        "from": last_dt.isoformat(),
        "to": target_dt.isoformat(),
        "steps": steps,
        "version": rec.version,
        "series": preds
    })


@api_bp.get('/metrics')
@swag_from({'tags': ['Métricas']})
def metrics():
    ticker = request.args.get('ticker', 'NVDA').upper()
    rows = ResultadoMetricas.query.filter_by(ticker=ticker).order_by(ResultadoMetricas.trained_at.desc()).limit(50).all()
    return jsonify([
        {
            'id': r.id, 'ticker': r.ticker, 'version': r.model_version, 'horizon': r.horizon,
            'mae': r.mae, 'rmse': r.rmse, 'mape': r.mape, 'accuracy': r.accuracy,
            'trained_at': r.trained_at.isoformat()
        } for r in rows
    ])

@api_bp.get('/retrain/history')
@swag_from({'tags': ['Métricas']})
def retrain_history():
    ticker = request.args.get('ticker', 'NVDA').upper()
    rows = RetrainHistory.query.filter_by(ticker=ticker).order_by(RetrainHistory.created_at.desc()).limit(100).all()
    return jsonify([
        {
            'id': r.id, 'ticker': r.ticker, 'version': r.model_version,
            'mae': r.mae, 'rmse': r.rmse, 'mape': r.mape, 'trigger': r.trigger,
            'created_at': r.created_at.isoformat()
        } for r in rows
    ])

@api_bp.post('/tasks/daily_update')
@swag_from({'tags': ['Tarefas'], 'description': 'Endpoint para ser chamado pelo Render Cron Job'})
def daily_update():
    if not _auth_ok(request):
        return jsonify({'error': 'unauthorized'}), 401
    ticker = (request.json or {}).get('ticker', 'NVDA').upper()

    with current_app.test_request_context(f'/api/update_data?ticker={ticker}'):
        resp = update_data()

    with current_app.test_request_context('/api/train', json={'ticker': ticker}):
        tr = train()

    return {'ok': True, 'updated': resp, 'train': tr}

@api_bp.get('/series')
def series():
    """Série OHLCV para o ticker (últimos N dias)."""
    ticker = request.args.get('ticker','NVDA').upper()
    limit = int(request.args.get('limit', 800))
    rows = (PrecoDiario.query
            .filter_by(ticker=ticker)
            .order_by(PrecoDiario.date.desc())
            .limit(limit).all())
    data = [{
        "date": r.date.isoformat(),
        "open": r.open, "high": r.high, "low": r.low,
        "close": r.close, "volume": r.volume
    } for r in reversed(rows)]
    return {"ticker": ticker, "data": data}

@api_bp.get('/metrics/history')
def metrics_history():
    from ..models import ModelRegistry
    ticker = (request.args.get('ticker') or 'NVDA').upper()
    rows = (ModelRegistry.query
            .filter_by(ticker=ticker)
            .order_by(ModelRegistry.registered_at.asc())
            .all())
    hist = []
    for r in rows:
        if r.mae is None:  # só entra se tiver métricas
            continue
        ts = r.registered_at.isoformat() if r.registered_at else None
        hist.append({
            "when": ts,
            "mae": r.mae,
            "rmse": r.rmse,
            "mape": r.mape,
            "r2": r.r2,
        })
    return {"ticker": ticker, "history": hist}


@api_bp.get('/backtest')
def backtest():
    from ..ml.trainer import load_best_model
    from ..ml.constants import DEFAULT_LOOKBACK
    from ..models import PrecoDiario

    ticker = request.args.get('ticker','NVDA').upper()
    window = int(request.args.get('window', 180))
    lookback = int(request.args.get('lookback', DEFAULT_LOOKBACK))

    # 1) carrega o VENCEDOR e seu SCALER salvo no treino
    model, scaler, rec = load_best_model(ticker)

    # 2) série (mesma fonte do treino)
    rows = (PrecoDiario.query.filter_by(ticker=ticker)
            .order_by(PrecoDiario.date.asc()).all())
    if not rows:
        return jsonify({"error":"no data"}), 400

    import pandas as pd
    close = pd.Series([r.close for r in rows],
                      index=pd.to_datetime([r.date for r in rows]))

    # 3) backtest 1-step usando o MESMO scaler do modelo vencedor
    from ..ml.eval import rolling_backtest_1step, metrics_from_series
    df_pred = rolling_backtest_1step(model, scaler, close,
                                     lookback=lookback, window=window)
    # padroniza nomes (opcional, se não aplicar o Patch 1)
    # rename = {}
    # if "real" in df_pred.columns:     rename["real"] = "y_true"
    # if "previsto" in df_pred.columns: rename["previsto"] = "y_pred"
    # if "pred" in df_pred.columns:     rename["pred"] = "y_pred"
    # df_pred = df_pred.rename(columns=rename)
    mets = metrics_from_series(df_pred)

    # resposta para os gráficos
    out = {
        "ticker": ticker,
        "version": rec.version,
        "registered_at": rec.registered_at.isoformat() if rec.registered_at else None,
        "metrics": mets,
        "series": {
            "dates": [d.strftime("%Y-%m-%d") for d in df_pred.index],
            "y_true": df_pred["y_true"].tolist(),
            "y_pred": df_pred["y_pred"].tolist(),
        }
    }
    return out

# --- LISTAR TICKERS DO BANCO ---
@api_bp.get('/tickers')
def list_tickers():
    # DISTINCT + resumo (primeira/última data e nº de linhas)
    rows = (db.session.query(
                PrecoDiario.ticker,
                db.func.min(PrecoDiario.date),
                db.func.max(PrecoDiario.date),
                db.func.count(PrecoDiario.id)
            )
            .group_by(PrecoDiario.ticker)
            .order_by(PrecoDiario.ticker.asc())
            .all())

    out = []
    for t, dmin, dmax, n in rows:
        out.append({"ticker": t, "start": dmin.isoformat() if dmin else None, "end":   dmax.isoformat() if dmax else None, "rows":  int(n)})
    return {"tickers": out}
