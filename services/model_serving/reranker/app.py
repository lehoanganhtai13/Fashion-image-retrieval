import os
import io
import time

from typing import List

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from loguru import logger
from contextlib import asynccontextmanager

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, CollectorRegistry, make_asgi_app
from prometheus_client.multiprocess import MultiProcessCollector

from models import ReRanker


# Remove all handlers associated with the root logger object
logger.remove()
# Configure the logger
logger.add(
    "./logs/process.log",
    rotation="5 GiB",
    format="{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level} | {name}:{function}:{line} - {message}"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global reranker
    global counter
    global error_counter
    global histogram
    
    model_id = os.getenv('RERANKER_MODEL_ID')
    device = os.getenv('RERANKER_DEVICE')

    # Initialize the re-ranker
    print('Initializing the re-ranker...')
    reranker = ReRanker(model_id=model_id, device=device)
    logger.info(f"Re-ranker initialized with model ID {model_id} and device {device}")

    # Initialize the Prometheus metrics
    logger.info("Initializing Prometheus metrics...")
    counter = Counter("reranking_requests", "Number of re-ranking requests", ("method", "endpoint"))
    error_counter = Counter("reranking_errors_requests", "Number of errors", ("method", "endpoint"))
    histogram = Histogram("reranking_request_response_time_seconds", "Reranking request response time in seconds", ("method", "endpoint"), unit="seconds")

    logger.info("Model serving service started")

    # Clean up the reranker
    yield
    del reranker
    del counter
    del error_counter
    del histogram

# Add Prometheus ASGI middleware to expose /metrics for multi-process mode
# Acess for more details: https://prometheus.github.io/client_python/exporting/http/fastapi-gunicorn/
def make_metrics_app():
    registry = CollectorRegistry()
    MultiProcessCollector(registry)
    return make_asgi_app(registry=registry)

app = FastAPI(lifespan=lifespan)
app.mount("/metrics", make_metrics_app())

# Start instrumenting the FastAPI app
instrumentator = Instrumentator().instrument(app)
instrumentator.expose(app)

@app.get("/get_device")
async def get_device():
    """Get the device used by the re-ranker."""
    return {"device": reranker.device}

@app.post("/re-ranking")
async def re_ranking(request: Request, pairs: List[List[str]]):
    """Re-rank the given pairs of text and documents."""
    # Record the start time
    start_time = time.time()

    if not pairs:
        raise HTTPException(status_code=400, detail="No data provided")

    logger.info(f"Received pairs to re-rank: {pairs}")

    try:
        scores = reranker.rerank(pairs)
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # Record the end time
        end_time = time.time()
        elapsed_time = end_time - start_time
        histogram.labels(method="post", endpoint="/re-ranking").observe(elapsed_time)
        counter.labels(method="post", endpoint="/re-ranking").inc()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"sorted_indices": sorted_indices}


if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run("app:app", host="0.0.0.0", port=8004)
