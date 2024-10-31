from prometheus_client import multiprocess, Gauge

# Worker Options
workers = 8
worker_class = "uvicorn.workers.UvicornWorker"

# Address and Port for the workers to bind to
bind = "0.0.0.0:8000"

# Worker timeout
timeout = 120

# Log level
loglevel = "info"

# Access log
accesslog = "./logs/access.log"
errorlog = "./logs/error.log"

# Use Prometheus Gauge type to track number of active workers with 'livesum' mode
worker_gauge = Gauge("clip_active_workers", "Number of active workers for CLIP encoder service", multiprocess_mode="livesum")

def post_worker_init(worker):
    """Increment the number of workers when a new worker is initialized."""
    worker_gauge.inc()

def worker_exit(server, worker):
    """Decrement the number of workers when a worker exits."""
    worker_gauge.dec()

# Acess here for more details: https://prometheus.github.io/client_python/multiprocess/
def child_exit(server, worker):
    """Mark a process as dead when a worker exits."""
    multiprocess.mark_process_dead(worker.pid)
