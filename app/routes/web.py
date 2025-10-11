
from flask import Blueprint, render_template, request
from ..models import ResultadoMetricas, PrecoDiario
from .. import db
from datetime import datetime

web_bp = Blueprint('web', __name__)

@web_bp.get('/')
def home():
    tick = request.args.get('ticker', 'NVDA').upper()
    latest = ResultadoMetricas.query.filter_by(ticker=tick).order_by(ResultadoMetricas.trained_at.desc()).first()
    prices = PrecoDiario.query.filter_by(ticker=tick).order_by(PrecoDiario.date.desc()).limit(30).all()
    return render_template('index.html', ticker=tick, latest=latest, prices=list(reversed(prices)))

@web_bp.get('/simulate')
def simulate():
    return render_template('simulate.html')


