# app/monitoring.py
import time
from flask import Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY

def _get_or_create(metric_cls, name, documentation, labelnames=(), **kwargs):
    # Reusa se já existir no registry (evita 'Duplicated timeseries')
    existing = REGISTRY._names_to_collectors.get(name)
    if existing is not None:
        return existing
    return metric_cls(name, documentation, labelnames, **kwargs)

# Definições (idempotentes)
HTTP_REQUESTS = _get_or_create(
    Counter, "http_requests", "Total de requisições HTTP",
    ["method", "endpoint", "http_status"]
)
HTTP_LATENCY = _get_or_create(
    Histogram, "http_request_duration_seconds", "Duração das requisições HTTP (s)",
    ["method", "endpoint"],
    buckets=(0.01,0.025,0.05,0.1,0.25,0.5,1,2,5,10)
)
INFERENCE_LATENCY = _get_or_create(
    Histogram, "inference_seconds", "Tempo de inferência do modelo (s)",
    ["ticker", "version"],
    buckets=(0.005,0.01,0.025,0.05,0.1,0.25,0.5,1,2,5)
)
INPROGRESS = _get_or_create(
    Gauge, "http_requests_in_progress", "Requisições em andamento",
    ["endpoint"]
)

RETRAIN_COUNT = _get_or_create(
    Counter, "retrain_total", "Qtd de retreinagens", ["ticker", "mode"]
)
RETRAIN_DURATION = _get_or_create(
    Histogram, "retrain_duration_seconds", "Tempo de treino/retreino (s)",
    ["ticker", "mode"],
    buckets=(5,10,20,30,45,60,90,120,180,300,600,1200)
)

def metrics_endpoint(app):
    @app.get("/metrics")
    def _metrics():
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

def timing_middleware(app):
    @app.before_request
    def _start():
        try:
            from flask import request
            request._start_time = time.perf_counter()
            INPROGRESS.labels(endpoint=request.path).inc()
        except Exception:
            pass

    @app.after_request
    def _end(response):
        try:
            from flask import request
            dt = time.perf_counter() - getattr(request, "_start_time", time.perf_counter())
            HTTP_LATENCY.labels(request.method, request.path).observe(dt)
            HTTP_REQUESTS.labels(request.method, request.path, str(response.status_code)).inc()
            INPROGRESS.labels(endpoint=request.path).dec()
        except Exception:
            pass
        return response
