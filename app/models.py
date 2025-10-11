
from . import db
from datetime import datetime

class PrecoDiario(db.Model):
    __tablename__ = 'preco_diario'
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(16), index=True, nullable=False)
    date = db.Column(db.Date, index=True, nullable=False)
    open = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    close = db.Column(db.Float)
    adj_close = db.Column(db.Float)
    volume = db.Column(db.BigInteger)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint('ticker', 'date', name='uq_ticker_date'),)

class ResultadoMetricas(db.Model):
    __tablename__ = 'resultado_metricas'
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(16), index=True, nullable=False)
    model_version = db.Column(db.String(64), index=True, nullable=False)
    horizon = db.Column(db.Integer, default=1)
    split_start = db.Column(db.Date)
    split_end = db.Column(db.Date)
    mae = db.Column(db.Float)
    rmse = db.Column(db.Float)
    mape = db.Column(db.Float)
    r2 = db.Column(db.Float)
    hits = db.Column(db.Integer)
    accuracy = db.Column(db.Float)
    drift_mae = db.Column(db.Float)
    trained_at = db.Column(db.DateTime, default=datetime.utcnow)

class RetrainHistory(db.Model):
    __tablename__ = 'retrain_history'
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(16), index=True, nullable=False)
    model_version = db.Column(db.String(64), index=True, nullable=False)
    train_start = db.Column(db.Date)
    train_end = db.Column(db.Date)
    eval_start = db.Column(db.Date)
    eval_end = db.Column(db.Date)
    mae = db.Column(db.Float)
    rmse = db.Column(db.Float)
    mape = db.Column(db.Float)
    r2 = db.Column(db.Float)
    trigger = db.Column(db.String(16))  # schedule|drift|manual
    drift_stat = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ModelRegistry(db.Model):
    __tablename__ = "model_registry"
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String, index=True, nullable=False)
    model_id = db.Column(db.Integer, nullable=False)   # 1..5
    model_name = db.Column(db.String, nullable=False)
    version = db.Column(db.String, nullable=False)     # timestamp/hash
    path_model = db.Column(db.String, nullable=False)
    path_scaler = db.Column(db.String, nullable=False)
    mae = db.Column(db.Float)
    rmse = db.Column(db.Float)
    mape = db.Column(db.Float)
    r2 = db.Column(db.Float)
    accuracy = db.Column(db.Float)
    params = db.Column(db.Text)                        # JSON de hiperparâmetros
    is_winner = db.Column(db.Boolean, default=False, index=True)
    registered_at = db.Column(db.DateTime, server_default=db.func.now(), index=True)