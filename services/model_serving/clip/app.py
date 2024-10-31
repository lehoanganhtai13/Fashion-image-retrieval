import os
import io
import time

from PIL import Image
from typing import List

import uvicorn
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from loguru import logger
from contextlib import asynccontextmanager

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, CollectorRegistry, make_asgi_app
from prometheus_client.multiprocess import MultiProcessCollector

from models import CLIPEncoder


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
    global encoder
    global counter
    global error_counter
    global histogram
    
    model_id = os.getenv('CLIP_MODEL_ID')
    device = os.getenv('CLIP_DEVICE')

    # Initialize the CLIP encoder
    print('Initializing the CLIP encoder...')
    encoder = CLIPEncoder(model_id=model_id, device=device)
    logger.info(f"CLIP encoder initialized with model ID {model_id} and device {device}")

    # Initialize the Prometheus metrics
    logger.info("Initializing Prometheus metrics...")
    counter = Counter("encoding_requests", "Number of encoding requests", ("method", "endpoint"))
    error_counter = Counter("encoding_errors_requests", "Number of errors", ("method", "endpoint"))
    histogram = Histogram("encoding_request_response_time_seconds", "Encoding request response time in seconds", ("method", "endpoint"), unit="seconds")

    logger.info("Model serving service started")

    # Clean up the encoder
    yield
    del encoder
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
    """Get the device used for encoding."""
    return {"device": encoder.device}

@app.post("/encode-text")
async def encode_text(request: Request, texts: List[str]):
    """Encode the given batch of texts."""
    # Record the start time
    start_time = time.time()

    logger.info(f"Received texts to encode: {texts}")
    
    try:
        embeddings = encoder.get_text_embeddings(texts)
        # logger.info(f"Embeddings: {embeddings}")

        # Record the end time
        end_time = time.time()
        elapsed_time = end_time - start_time
        histogram.labels(method="post", endpoint="/encode-text").observe(elapsed_time)
        counter.labels(method="post", endpoint="/encode-text").inc()

        return {"embeddings": embeddings}
    except Exception as e:
        logger.error(f"Error encoding texts: {e}")
        error_counter.labels(method="post", endpoint="/encode-text").inc()
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/encode-image")
async def encode_image(request: Request, files: List[UploadFile] = File(...)):
    """Encode the given batch of images."""
    # Record the start time
    start_time = time.time()

    logger.info("Received image files to encode")

    try:
        images = [Image.open(io.BytesIO(await file.read())).convert("RGB") for file in files]
        embeddings = encoder.get_image_embeddings(images)
        # logger.info(f"Embeddings: {embeddings}")

        # Record the end time
        end_time = time.time()
        elapsed_time = end_time - start_time
        histogram.labels(method="post", endpoint="/encode-image").observe(elapsed_time)
        counter.labels(method="post", endpoint="/encode-image").inc()

        return {"embeddings": embeddings}
    except Exception as e:
        logger.error(f"Error encoding images: {e}")
        error_counter.labels(method="post", endpoint="/encode-image").inc()
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
