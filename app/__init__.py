from sqlalchemy import event
from sqlalchemy.engine import Engine
from flask import Flask
from flasgger import Swagger
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from pathlib import Path
import os

db = SQLAlchemy()
migrate = Migrate()

def create_app():
    app = Flask( __name__, instance_relative_config=True, static_folder="static", static_url_path="/static")
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret')

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', f"sqlite:///{os.path.join(app.instance_path, 'app.db')}")
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"connect_args": {"timeout": 30}}
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SWAGGER'] = {'title': 'Stock LSTM API', 'uiversion': 3}
    Swagger(app)

    db.init_app(app)
    migrate.init_app(app, db)
    @event.listens_for(Engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        # Só aplica se for sqlite
        try:
            # Alguns DBAPIs não são sqlite3; ignore nesses casos
            if dbapi_connection.__class__.__module__.startswith("sqlite3"):
                cur = dbapi_connection.cursor()
                cur.execute("PRAGMA journal_mode=WAL;")     # melhor concorrência
                cur.execute("PRAGMA synchronous=NORMAL;")   # performance ok
                cur.execute("PRAGMA busy_timeout=5000;")    # 5s de espera no lock
                cur.close()
        except Exception:
            # não explode a app se o PRAGMA falhar
            pass
    
    from . import models  # noqa
    from .routes.api import api_bp
    from .routes.web import web_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(web_bp)

    from .monitoring_simple import metrics_endpoint
    @app.get("/metrics")
    def _metrics():
        return metrics_endpoint()

    with app.app_context():
        db.create_all()
    return app
