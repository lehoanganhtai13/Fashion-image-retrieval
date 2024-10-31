import os
import io
import time
from datetime import timedelta

from typing import List
from dotenv import load_dotenv
from PIL import Image

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from loguru import logger
from contextlib import asynccontextmanager

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, CollectorRegistry, make_asgi_app
from prometheus_client.multiprocess import MultiProcessCollector

from etl.utils import EncoderClient, VectorDatabase, Minio, BM25Client, ReRankerClient, LLMEncoderClient


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
    global bm25_client
    global re_ranker
    global re_ranker_device
    global LLMEncoder
    global milvus_client
    global minio_client
    global milvus_collection_name
    global milvus_index_type
    global milvus_metric_type
    global minio_bucket_name
    global counter
    global error_counter
    global histogram
    
    # Load the environment variables
    folder_path = os.path.dirname(__file__)
    load_dotenv(os.path.join(folder_path, ".env"))

    # Set which host to skip proxy
    os.environ['NO_PROXY'] = os.getenv('NO_PROXY_HOST', 'None')

    # Initialize the encoder client
    print('Initializing the encoder client...')
    logger.info(f"Connecting to the model serving server at {os.getenv('MODEL_SERVING_HOST')}:{os.getenv('MODEL_SERVING_PORT')}...")
    encoder = EncoderClient(
        host=os.getenv('MODEL_SERVING_HOST'),
        port=os.getenv('MODEL_SERVING_PORT'),
    )
    
    # Initialize the LLM encoder
    print("Initializing the LLM encoder...")
    logger.info(f"Connecting to the model serving server at {os.getenv('MODEL_SERVING_HOST')}:{os.getenv('EMBEDDER_MODEL_SERVING_PORT')}...")
    LLMEncoder = LLMEncoderClient(
        host=os.getenv("MODEL_SERVING_HOST"),
        port=os.getenv("EMBEDDER_MODEL_SERVING_PORT"),
    )

    # Initialize the ReRanker client
    print("Initializing the ReRanker client...")
    logger.info(f"Connecting to the model serving server at {os.getenv('MODEL_SERVING_HOST')}:{os.getenv('RE_RANKER_MODEL_SERVING_PORT')}...")
    re_ranker = ReRankerClient(
        host=os.getenv("MODEL_SERVING_HOST"),
        port=os.getenv("RE_RANKER_MODEL_SERVING_PORT"),
    )
    re_ranker_device = re_ranker.get_device()

    # Initialize the vector database client
    print('Initializing the Milvus client...')
    logger.info(f"Connecting to the Milvus server at {os.getenv('MILVUS_HOST')}:{os.getenv('MILVUS_PORT')}...")
    milvus_client = VectorDatabase(
        host=os.getenv('MILVUS_HOST'),
        port=os.getenv('MILVUS_PORT')
    )
    milvus_collection_name = [os.getenv("MILVUS_COLLECTION"), os.getenv("MILVUS_NAME_COLLECTION"), os.getenv("MILVUS_CATEGORY_COLLECTION")]
    for collection_name in milvus_collection_name:
        load_state = milvus_client.load_collection(collection_name)
        if not load_state:
            logger.error("Failed to load the Milvus collection into memory")
            raise Exception(f"Failed to load the Milvus collection {collection_name} into memory")
        milvus_index_type = os.getenv('MILVUS_INDEX_TYPE')
        milvus_metric_type = os.getenv('MILVUS_METRIC_TYPE')
    logger.info(f"Milvus collections loaded successfully")

    # Initialize the Minio client
    print('Initializing the Minio client...')
    logger.info(f"Connecting to the Minio server at {os.getenv('MINIO_ENDPOINT')}...")
    minio_client = Minio(
        endpoint=os.getenv('MINIO_ENDPOINT'),
        access_key=os.getenv('MINIO_ACCESS_KEY_ID'),
        secret_key=os.getenv('MINIO_SECRET_ACCESS_KEY'),
        secure=False
    )
    minio_bucket_name = os.getenv('MINIO_BUCKET_NAME')
    if not minio_client.bucket_exists(minio_bucket_name):
        logger.error(f"The Minio bucket {minio_bucket_name} does not exist")
        raise Exception("The Minio bucket does not exist")

    # Initialize the BM25 client
    print("Initializing the BM25 client...")
    bm25_client = BM25Client(
        storage=minio_client,
        bucket_name=minio_bucket_name,
    )

    # Set up the metrics
    logger.info("Setting up the metrics...")
    counter = Counter("search_image_requests_counter", "The number of search image requests", labelnames=("method", "endpoint"))
    error_counter = Counter("search_image_errors_counter", "The number of search image errors", labelnames=("method", "endpoint"))
    histogram = Histogram(
        "search_image_time_seconds_response",
        "The response time of the search image requests",
        unit="seconds",
        buckets=(0.5, 1.0, 1.5, 2.5, 5.0, float("inf")),
        labelnames=("method", "endpoint")
    )

    logger.info("Finished setting up all the clients and metrics")
    
    # Clean up all the clients when the server is shut down
    yield
    logger.info("Cleaning up the clients...")
    print("Cleaning up the clients...")
    del encoder
    del bm25_client
    del re_ranker
    del LLMEncoder
    del milvus_client
    del minio_client
    del milvus_collection_name
    del milvus_index_type
    del milvus_metric_type
    del minio_bucket_name
    del counter
    del error_counter
    del histogram

# Add Prometheus ASGI middleware to expose /metrics for multi-process mode
# Access for more details: https://prometheus.github.io/client_python/exporting/http/fastapi-gunicorn/
def make_metrics_app():
    registry = CollectorRegistry()
    MultiProcessCollector(registry)
    return make_asgi_app(registry=registry)

app = FastAPI(lifespan=lifespan)
app.mount("/metrics", make_metrics_app())

# Start instrumenting the FastAPI app
instrumentator = Instrumentator().instrument(app)
instrumentator.expose(app)

def category_detection(query: str):
    # Encode the query using LLM-based embedder
    query_embedding = LLMEncoder.encode_text([query])

    # Search for the top 5 categories
    results = milvus_client.search_vectors(
        collection_type="category",
        collection_name=milvus_collection_name[2],
        vectors=query_embedding,
        top_k=5,
        index_type=milvus_index_type,
        metric_type=milvus_metric_type
    )

    top_5_categories_list = [
        [res["entity"]["category_name"] for res in result]
        for result in results
    ]

    return top_5_categories_list

def image_retrieval(query: str, top_categories: List[str], top_p: int):
    # Add prefix to the query
    if "photo" not in query.lower():
        prompted_query = f"a photo of {query}."
    else:
        prompted_query = query

    # Encode the query using CLIP model
    dense_query_embedding = encoder.encode_text([prompted_query])

    # Search for the top p documents
    filtering_expression = " or ".join([f'product_category == "{category}"' for category in top_categories])
    results = milvus_client.search_vectors(
        collection_type="image",
        collection_name=milvus_collection_name[0],
        vectors=dense_query_embedding,
        top_k=top_p,
        index_type=milvus_index_type,
        metric_type=milvus_metric_type,
        filtering_expr=filtering_expression
    )
    image_id_list = [
        [res["entity"]["id"] for res in result]
        for result in results
    ]

    return image_id_list

def image_reranking(query: str, image_id_list: List[str], top_k: int):
    # Get the images from the Minio storage
    retrieved_vectors_limit = 16300
    product_names_list = []
    for i in range(0, len(image_id_list), retrieved_vectors_limit):
        batch_ids = image_id_list[i:i + retrieved_vectors_limit]
        entities = milvus_client.get_vectors(milvus_collection_name[0], batch_ids)
        names = [entity["product_name"] for entity in entities]
        product_names_list.extend(names)

    # Rerank the images based on the query and the product names
    sorted_indices = re_ranker.rerank(
        query=query,
        documents=product_names_list,
        top_k=top_k
    )

    top_k_image_ids = [image_id_list[i] for i in sorted_indices]

    return top_k_image_ids

@app.post("/search-image")
async def search(query: str = None, file: UploadFile = File(None), top_k: int = 1):
    """Search for the most top-k similar images to the given query."""
    # Mark the start time of the request
    start_time = time.time()

    # Perform search for either text or image not both
    if query and file:
        # Record the number of errors
        error_counter.labels(method="post", endpoint="/search-image").inc()

        logger.error("Please provide only one of the options --query or --file")
        raise HTTPException(status_code=400, detail="Please provide only one of the options --query or --file")

    if top_k > 100:
        # Record the number of errors
        error_counter.labels(method="post", endpoint="/search-image").inc()

        logger.error("The top-k value must be less than or equal to 100")
        raise HTTPException(status_code=400, detail="The top-k value must be less than or equal to 100")

    if query:
        logger.info(f"Received query: {query}")

        # Detect the top 5 categories
        top_5_categories = category_detection(query)[0]

        # Retrieve the images based on the query and the top 5 categories
        top_p = 100 if re_ranker_device != "cpu" else 10
        top_p_image_ids = image_retrieval(query, top_5_categories, top_p)

        # Rerank the images based on the query and the product names
        reranked_image_ids = image_reranking(query, top_p_image_ids[0], top_k)

        image_urls = []
        for _, image_id in enumerate(reranked_image_ids):
            image_path = f"images/{image_id}.jpg"
            image_url = minio_client.presigned_get_object(
                bucket_name=minio_bucket_name,
                object_name=image_path,
                expires=timedelta(hours=1)
            )
            image_urls.append(image_url)
        
        # Record the end time of the request
        end_time = time.time()
        # Record the duration of the request
        duration = end_time - start_time
        # Record the number of requests
        counter.labels(method="post", endpoint="/search-image").inc()
        # Record the duration of the request
        histogram.labels(method="post", endpoint="/search-image").observe(duration)

        return {"urls": image_urls}
    elif file:
        logger.info("Received image file to search")
        # Search by image
        image = Image.open(io.BytesIO(await file.read()))
        image = image.resize((224,224)).convert("RGB")
        embeddings = encoder.encode_image([image])
        result = milvus_client.search_vectors(
            collection_type="image",
            collection_name=milvus_collection_name[0],
            vectors=embeddings,
            top_k=top_k,
            index_type=milvus_index_type,
            metric_type=milvus_metric_type
        )[0]
        image_urls = []
        for r in result:
            image_url = minio_client.presigned_get_object(
                bucket_name=minio_bucket_name,
                object_name=f'images/{r["id"]}.jpg',
                expires=timedelta(hours=1)
            )
            image_urls.append(image_url)

        # Record the end time of the request
        end_time = time.time()
        # Record the duration of the request
        duration = end_time - start_time
        # Record the number of requests
        counter.labels(method="post", endpoint="/search-image").inc()
        # Record the duration of the request
        histogram.labels(method="post", endpoint="/search-image").observe(duration)

        return {"urls": image_urls}
    else:
        # Record the number of errors
        error_counter.labels(method="post", endpoint="/search-image").inc()

        logger.error("Please provide either --query or --file")
        raise HTTPException(status_code=400, detail="Please provide either --query or --file")

if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run("app:app", host="0.0.0.0", port=8008, reload=False)