import os
import io
import time

from typing import List, Dict

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from contextlib import asynccontextmanager

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, CollectorRegistry, make_asgi_app
from prometheus_client.multiprocess import MultiProcessCollector

from models import LLM


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
    global llm
    global counter
    global error_counter
    global histogram
    
    model_id = os.getenv('LLM_MODEL_ID')
    device = os.getenv('LLM_DEVICE')
    HF_token = os.getenv('HF_TOKEN')

    # Initialize the LLM
    print('Initializing the LLM...')
    llm = LLM(model_id=model_id, device=device, HF_token=HF_token)
    logger.info(f"LLM initialized with model ID {model_id} and device {device}")

    # Initialize the Prometheus metrics
    logger.info("Initializing Prometheus metrics...")
    counter = Counter("generating_requests", "Number of generating requests", ("method", "endpoint"))
    error_counter = Counter("generating_errors_requests", "Number of errors", ("method", "endpoint"))
    histogram = Histogram("generating_request_response_time_seconds", "Generating request response time in seconds", ("method", "endpoint"), unit="seconds")

    logger.info("Model serving service started")

    # Clean up the LLM
    yield
    del llm
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
    """Get the device used by the LLM."""
    return {"device": llm.device}

@app.post("/generate")
async def generate(request: Request, message: List[Dict[str, str]], max_new_tokens: int = 256):
    """Generate a response based on the given message."""
    # Record the start time
    start_time = time.time()

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    logger.info(f"Received message to generate: {message}")

    try:
        # Generate a response
        response = llm.generate(message, max_new_tokens)

        # Record the end time
        end_time = time.time()
        elapsed_time = end_time - start_time
        histogram.labels(method="post", endpoint="/generate").observe(elapsed_time)
        counter.labels(method="post", endpoint="/generate").inc()
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        error_counter.labels(method="post", endpoint="/generate").inc()
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
    return {"response": response}

@app.post("/stream")
async def stream(request: Request, message: List[Dict[str, str]], max_new_tokens: int = 256):
    """Stream the response based on the given message."""
    # Record the start time
    start_time = time.time()

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    logger.info(f"Received message to stream: {message}")

    try:
        # Generate a streaming response
        streamer = llm.streaming(message, max_new_tokens)

        # Record the end time
        end_time = time.time()
        elapsed_time = end_time - start_time
        histogram.labels(method="post", endpoint="/stream").observe(elapsed_time)
        counter.labels(method="post", endpoint="/stream").inc()
    except Exception as e:
        logger.error(f"Error streaming response: {e}")
        error_counter.labels(method="post", endpoint="/stream").inc()
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
    return StreamingResponse(streamer)


if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run("app:app", host="0.0.0.0", port=8002)
