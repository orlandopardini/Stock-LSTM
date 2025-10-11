# app/monitoring_simple.py
import os
import psutil
from flask import Response
from prometheus_client import Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

REGISTRY = CollectorRegistry()

CPU_PERCENT = Gauge("app_cpu_percent", "CPU total (%) do sistema", registry=REGISTRY)
PROC_CPU_PERCENT = Gauge("app_process_cpu_percent", "CPU (%) do processo Flask", registry=REGISTRY)
MEM_RSS_BYTES = Gauge("app_memory_rss_bytes", "RAM (RSS) do processo em bytes", registry=REGISTRY)
MEM_VMS_BYTES = Gauge("app_memory_vms_bytes", "Memória virtual (VMS) do processo em bytes", registry=REGISTRY)

_ps = psutil.Process(os.getpid())
_ps.cpu_percent(None)  # baseline

def _collect_now():
    CPU_PERCENT.set(psutil.cpu_percent(interval=0.0))
    PROC_CPU_PERCENT.set(_ps.cpu_percent(interval=None))
    mi = _ps.memory_info()
    MEM_RSS_BYTES.set(mi.rss)
    MEM_VMS_BYTES.set(mi.vms)

def metrics_endpoint():
    _collect_now()
    data = generate_latest(REGISTRY)
    return Response(data, mimetype=CONTENT_TYPE_LATEST)
