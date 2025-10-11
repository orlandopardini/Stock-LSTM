# app/ml/data.py
from __future__ import annotations
import pandas as pd

from .. import db
from ..models import PrecoDiario  # usa o seu model mostrado

def load_close_series(ticker: str, prefer_adj: bool = True) -> pd.Series:
    """
    Lê do banco a série de fechamento do ticker e devolve um pandas.Series
    indexado por data (ordenado), em float. Se prefer_adj=True e houver
    adj_close, prioriza-o; caso contrário usa close.
    """
    # consulta ordenada por data
    rows = (db.session.query(
                PrecoDiario.date,
                PrecoDiario.adj_close,
                PrecoDiario.close
            )
            .filter(PrecoDiario.ticker == ticker)
            .order_by(PrecoDiario.date.asc())
            .all())

    if not rows:
        return pd.Series(dtype="float64", name="close")

    # monta index (datas) e valores
    idx = pd.to_datetime([r[0] for r in rows])  # Date -> Timestamp
    if prefer_adj and any(r[1] is not None for r in rows):
        vals = [float(r[1]) if r[1] is not None else float(r[2]) for r in rows]
        name = "adj_close"
    else:
        vals = [float(r[2]) for r in rows]
        name = "close"

    s = pd.Series(vals, index=idx, name=name)
    # remove NaN e garante ordem crescente
    s = s.dropna().sort_index()
    return s