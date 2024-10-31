import sys
import os

from dotenv import load_dotenv
from loguru import logger
from typing import List
import pandas as pd
import pickle
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# Add the services/app directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../services/etl')))

from utils import EncoderClient, Minio, VectorDatabase, ETL, LLMEncoderClient, ReRankerClient, BM25Client

# ==========Split dataframes into batches==========

def split_dataframe(df: pd.DataFrame, batch_size: int):
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        yield df.iloc[start:end].reset_index(drop=True)

# ==========Category detection==========

def search_category(queries):
    if not queries:
        return []

    # Check what product category the queries belong to
    query_embeddings = LLMEncoder.encode_text(queries)
    results = milvus_client.search_vectors(
        collection_type="category",
        collection_name=milvus_collections[2],
        vectors=query_embeddings,
        top_k=5,
        index_type=milvus_index_type,
        metric_type=milvus_metric_type
    )

    # Get the top 5 categories
    top_5_categories_list = [
        [res["entity"]["category_name"] for res in result]
        for result in results
    ]

    # Get the number of images in the top 5 categories
    images_count_list = []
    for result in results:
        category_count = 0
        for res in result:
            category_count += res["entity"]["category_count"]
        images_count_list.append(category_count)

    return top_5_categories_list, images_count_list

def category_detection_batch(batch, idx):
    score = 0
    results = []
    failed = []
    successful_indices = []
    successful_images_count = []

    # Search for the top 5 categories of the queries belonging to
    queries = batch["query"].tolist()
    top_5_categories_list, images_count_list = search_category(queries)

    # Check if the actual category is in the top 5 categories
    batch_rows = list(batch.iterrows())
    for index, row in batch_rows:
        query = row["query"]
        category = row["processed_category_name"]
        top_5_categories = top_5_categories_list[index]

        if category in top_5_categories:
            results.append((query, category, top_5_categories))
            successful_indices.append(row.name + idx*len(batch_rows))
            successful_images_count.append(images_count_list[index])
            score += 1
        else:
            failed.append((query, category, top_5_categories))

    return score, results, failed, successful_indices, successful_images_count

# ==========Image retrieval==========

def search_images(query_embeddings: List = None, categories: List[str] = None, top_p: int = 100):
    if query_embeddings is None or categories is None:
        return None

    # Search for top-p images in all of the images in the top 5 categories
    filtering_expression = " or ".join([f'product_category == "{category}"' for category in categories])
    results = milvus_client.search_vectors(
        collection_type="image",
        collection_name=milvus_collections[0],
        vectors=query_embeddings,
        top_k=top_p,
        index_type=milvus_index_type,
        metric_type=milvus_metric_type,
        filtering_expr=filtering_expression
    )
    return results

def image_retrieval_batch(batch: pd.DataFrame, idx: int):
    score = 0
    results = []
    failed = []
    successful_indices = []
    image_indices = [] # List of retrieved image indices for each query 

    batch_rows = list(batch.iterrows())

    # Encode the queries to dense and sparse embeddings
    query_list = [row["query"] for _, row in batch_rows]
    dense_query_embeddings = encoder.encode_text(query_list)
    sparse_query_embeddings = bm25_client.encode_queries(query_list)

    for index, row in batch_rows:
        query = row["query"]
        dense_embedding = [dense_query_embeddings[index]]
        sparse_embedding = [sparse_query_embeddings[index]]

        # Search for the top-p images in the top 5 categories
        results = search_images([dense_embedding, sparse_embedding], row["top_5_categories"][2], row["top_p"])[0]

        # Check if the actual product name is in the top-p images
        product_name_list = [res.product_name for res in results]
        image_id_list = [res.id for res in results]
        if row["name_en"] in product_name_list:
            results.append((query, row["name_en"], product_name_list))
            successful_indices.append(row.name + idx*len(batch_rows))
            image_indices.append(image_id_list)
            score += 1
        else:
            failed.append((query, row["name_en"], product_name_list))
        
    return score, results, failed, successful_indices, image_indices

# ==========Image re-ranking==========

def reranking(query_embedding: np.ndarray, image_embeddings: np.ndarray, top_p: int = 100):
    # Rerank the top-p images based on cosine similarity using vectorized operations
    similarities = cosine_similarity(query_embedding.reshape(1, -1), image_embeddings)[0]
    
    # Get the top_p indices sorted by similarity in descending order
    sorted_indices = np.argsort(similarities)[-top_p:][::-1]
    
    return sorted_indices

def process_single_query_with_embedder(index: int, row, query_embedding: np.ndarray, limit: int, top_k: int):
    ids = row['image_ids']
    product_name = row['name_en']

    name_embeddings, product_names = [], []

    # Fetch embeddings and product names in batches to avoid overloading memory
    for i in range(0, len(ids), limit):
        batch_ids = ids[i:i + limit]

        entities = milvus_client.get_vectors(milvus_collections[0], batch_ids)
        names = [entity["product_name"] for entity in entities]
        product_names.extend(names)
        
        entities = milvus_client.get_vectors(milvus_collections[1], batch_ids)
        embeddings = [entity["name_embedding"] for entity in entities]
        name_embeddings.extend(embeddings)

    # Convert image embeddings to a NumPy array for efficient computation
    name_embeddings = np.array(name_embeddings)
    
    # Rerank images and get top_k indices based on the query and name embeddings
    sorted_indices = reranking(query_embedding, name_embeddings, top_k)
    
    top_p_product_names = [product_names[i] for i in sorted_indices]
    
    # Check if the product_name is in the top_k
    success = product_name in top_p_product_names
    return index, success, row['query'], product_name, top_p_product_names, [ids[i] for i in sorted_indices]

def reranking_batch_with_embedder(batch: pd.DataFrame, idx: int, top_k: int):
    limit = 16300
    score = 0
    results = []
    failed = []
    successful_indices = []
    image_indices = []

    batch_rows = list(batch.iterrows())
    query_list = [row['query'] for _, row in batch_rows]
    
    # Encode all queries in one go to minimize overhead
    query_embeddings = LLMEncoder.encode_text(query_list)

    # Use multiprocessing to process each query in parallel
    with ProcessPoolExecutor() as executor:
        futures = []
        for i, row in enumerate(batch_rows):
            query_embedding = np.array(query_embeddings[i])
            futures.append(executor.submit(process_single_query_with_embedder, i, row[1], query_embedding, limit, top_k))
        
        # Collect the results from the futures as they complete
        for future in futures:
            index, success, query, product_name, top_p_product_names, sorted_image_ids = future.result()
            if success:
                results.append((query, product_name, top_p_product_names))
                successful_indices.append(index + idx * len(batch_rows))
                image_indices.append(sorted_image_ids)
                score += 1
            else:
                failed.append((query, product_name, top_p_product_names))
    
    return score, results, failed, successful_indices, image_indices

def process_single_query_with_reranker(index: int, row, query: str, limit: int, top_k_list: List[int]):
    ids = row["image_ids"]
    product_name = row["name_en"]

    success = [False for _ in top_k_list]
    top_p_product_names = [[] for _ in top_k_list]
    sorted_image_indices = [[] for _ in top_k_list]

    # Fetch product names of all previous images
    product_names_list = []
    for i in range(0, len(ids), limit):
        batch_ids = ids[i:i + limit]
        entities = milvus_client.get_vectors(milvus_collections[0], batch_ids)
        names = [entity["product_name"] for entity in entities]
        product_names_list.extend(names)
    
    # Rerank images based on the query and the product names
    sorted_indices = re_ranker.rerank(
        query=query,
        documents=product_names_list,
        top_k=top_k
    )
    for idx, top_k in enumerate(top_k_list):
        sorted_indices_top_k = sorted_indices[:top_k]
        
        top_p_product_names[idx] = [product_names_list[i_] for i_ in sorted_indices_top_k]
        sorted_image_indices[idx] = [ids[i_] for i_ in sorted_indices_top_k]

        # Check if the product_name is in the top_k
        success[idx] = product_name in top_p_product_names[idx]

    return index, success, row['query'], product_name, top_p_product_names, sorted_image_indices

def reranking_batch_with_reranker(batch: pd.DataFrame, idx: int, top_k_list: List[int]):
    limit = 16300
    scores = [0 for _ in top_k_list]
    results = [[] for _ in top_k_list]
    failed = [[] for _ in top_k_list]
    successful_indices = [[] for _ in top_k_list]
    image_indices = [[] for _ in top_k_list]

    batch_rows = list(batch.iterrows())
    query_list = [row["query"] for _, row in batch_rows]

    # Use multiprocessing to process each query in parallel
    with ProcessPoolExecutor() as executor:
        futures = []
        for i, row in enumerate(batch_rows):
            futures.append(executor.submit(process_single_query_with_reranker, i, row[1], query_list[i], limit, top_k_list))
        
        # Collect results from the futures as they complete
        for future in futures:
            index, success, query, product_name, top_p_product_names, sorted_image_ids = future.result()
            for idx, _ in enumerate(top_k_list):
                if success[idx]:
                    results[idx].append((query, product_name, top_p_product_names[idx]))
                    successful_indices[idx].append(index + idx * len(batch_rows))
                    image_indices[idx].append(sorted_image_ids[idx])
                    scores[idx] += 1
                else:
                    failed[idx].append((query, product_name, top_p_product_names[idx]))
    
    return score, results, failed, successful_indices, image_indices

# ==========Model functionalities==========

def initialize_model(model_id: str, device: str):
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    return model, processor, tokenizer

def encode_text(texts, model, tokenizer):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    embeddings = outputs.cpu().detach().numpy().tolist()
    del inputs, outputs
    torch.cuda.empty_cache()
    return embeddings

def encode_image(images, model, processor):
    inputs = processor(None, images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    embeddings = outputs.cpu().detach().numpy().tolist()
    del inputs, outputs
    torch.cuda.empty_cache()
    return embeddings

def run(model: CLIPModel, processor: CLIPProcessor, tokenizer: CLIPTokenizer, df: pd.DataFrame, batch_size: int):
    # Encode the images
    image_embeddings = []
    for index, batch in enumerate(batches):
        images = batch["image"].tolist()
        embeddings = encode_image(images, model, processor)
        image_embeddings.extend(embeddings)
        
    image_embeddings_tensor = torch.tensor(image_embeddings).to("cuda")

    score_1 = 0
    score_5 = 0
    score_10 = 0

    bar = tqdm(total=len(batches))
    for index, batch in enumerate(batches):
        query_list = batch["query"].tolist()
        query_embeddings = encode_text(query_list, model, tokenizer)

        for i, row in batch.iterrows():
            query_embedding = query_embeddings[i]
            query_embedding_tensor = torch.tensor(query_embedding).to("cuda")

            scores = F.cosine_similarity(query_embedding_tensor.unsqueeze(0), image_embeddings_tensor, dim=1).cpu().numpy()
            
            top_1_index = scores.argsort()[-1]
            top_5_indices = scores.argsort()[-5:][::-1]
            top_10_indices = scores.argsort()[-10:][::-1]

            score_1 += 1 if i + index*batch_size == top_1_index else 0
            score_5 += 1 if i + index*batch_size in top_5_indices else 0
            score_10 += 1 if i + index*batch_size in top_10_indices else 0
        
        bar.desc = f"Average cumulative score - Top 1: {score_1/len(df):.2%}, Top 5: {score_5/len(df):.2%}, Top 10: {score_10/len(df):.2%}"
        bar.update(1)

    bar.close()

    print(f"Total score - Top 1: {score_1/len(df):.2%}, Top 5: {score_5/len(df):.2%}, Top 10: {score_10/len(df):.2%}")
    logger.info(f"Total score - Top 1: {score_1/len(df):.2%}, Top 5: {score_5/len(df):.2%}, Top 10: {score_10/len(df):.2%}")

if __name__ == "__main__":

    # Remove all handlers associated with the root logger object
    logger.remove()
    # Configure the logger
    logger.add(
        "./logs/eval.log",
        rotation="5 GiB",
        format="{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level} | {name}:{function}:{line} - {message}"
    )

    load_dotenv("../services/etl/.env")
    os.environ["NO_PROXY"] = os.getenv("NO_PROXY_HOST", "None")

    # Initialize the CLIP encoder
    print("Initializing the CLIP encoder...")
    encoder = EncoderClient(
        host=os.getenv("MODEL_SERVING_HOST"),
        port=os.getenv("MODEL_SERVING_PORT"),
    )

    # Initialize the Milvus client and create a collection
    print("Initializing the Milvus client...")
    milvus_client = VectorDatabase(
        host=os.getenv("MILVUS_HOST"),
        port=os.getenv("MILVUS_PORT")
    )

    milvus_collections = [os.getenv("MILVUS_COLLECTION"), os.getenv("MILVUS_NAME_COLLECTION"), os.getenv("MILVUS_CATEGORY_COLLECTION")]
    for collection_name in milvus_collections:
        load_state = milvus_client.load_collection(collection_name)
        if not load_state:
            raise Exception(f"Failed to load the Milvus collection {collection_name} into memory")
    milvus_index_type = os.getenv("MILVUS_INDEX_TYPE")
    milvus_metric_type = os.getenv("MILVUS_METRIC_TYPE")

    # Initialize the Minio client and create a bucket
    print("Initializing the Minio client...")
    bucket_name = os.getenv("MINIO_BUCKET_NAME")
    minio_client = Minio(
        endpoint=os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY_ID"),
        secret_key=os.getenv("MINIO_SECRET_ACCESS_KEY"),
        secure=False
    )
    if not minio_client.bucket_exists(bucket_name):
        raise Exception(f"Bucket {bucket_name} does not exist!")

    # Initialize the LLM encoder
    print("Initializing the LLM encoder...")
    LLMEncoder = LLMEncoderClient(
        host=os.getenv("MODEL_SERVING_HOST"),
        port=os.getenv("EMBEDDER_MODEL_SERVING_PORT"),
    )

    # Initialize the ReRanker client
    print("Initializing the ReRanker client...")
    re_ranker = ReRankerClient(
        host=os.getenv("MODEL_SERVING_HOST"),
        port=os.getenv("RE_RANKER_MODEL_SERVING_PORT"),
    )

    # Initialize the BM25 client
    print("Initializing the BM25 client...")
    bm25_client = BM25Client(path="../services/etl/bm25/state_dict.json")

    df_processor = ETL(
        folder_name="",
        encoder=None,
        BM25_encoder=None,
        storage_client=None,
        db_client=None,
        collection_name="",
        bucket_name="",
        collection_type=""
    )

    # Define the parameters
    max_range = 10000
    batch_size = 1000

    # Load the test dataset
    df = pd.read_csv("./dataset/GLAMI-1M-test-en-queries.csv")
    df = df.iloc[:max_range].reset_index(drop=True)

    # Obtain the images
    df["image_file"] = os.path.join(os.getcwd(), "../services/etl/dataset/GLAMI-1M-test-dataset/images/") + df["image_id"].astype(str) + ".jpg"
    df["image"] = None
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch_image_files = df["image_file"][start:end]
        batch_images = df_processor.process_images_in_batch(batch_image_files)
        df.iloc[start:end, df.columns.get_loc("image")] = batch_images

    batches = list(split_dataframe(df, batch_size))

    # Define the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Starting the evaluation pipeline...")
    logger.info("Starting the evaluation pipeline...")

    # ===========OpenAI/CLIP-vit-32================

    model, processor, tokenizer = initialize_model("openai/clip-vit-base-patch32", device)

    print("Evaluating using the OpenAI/CLIP-vit-32 model...")
    run(model, processor, tokenizer, df, batch_size)

    del model, processor, tokenizer
    torch.cuda.empty_cache()

    print("========================================")

    # ===========patrickjohncyh/fashion-clip================

    model, processor, tokenizer = initialize_model("patrickjohncyh/fashion-clip", device)

    print("Evaluating using the Fashion-CLIP model...")
    run(model, processor, tokenizer, df, batch_size)

    del model, processor, tokenizer
    torch.cuda.empty_cache()

    print("========================================")

    # ===========Our solution================

    print("Evaluating using our solution...")

    print("Evaluating image category detection stage...")

    top_p = 50
    total_score = 0
    all_results = []
    all_failed = []
    all_successful_indices = []
    all_successful_images_count = []

    bar = tqdm(total=len(batches))
    for index, batch in enumerate(batches):
        score, results, failed, successful_indices, successful_images_count = category_detection_batch(batch, index)
        total_score += score
        all_results.extend(results)
        all_failed.extend(failed)
        all_successful_indices.extend(successful_indices)
        all_successful_images_count.extend(successful_images_count)
        bar.set_description(f"Processing batches - Average cumulative score: {total_score/len(df):.2%}")
        bar.update(1)
    bar.close()

    print(f"Total score: {total_score/len(df):.2%}")

    print("Evaluating image retrieval stage...")
    batch_size = 100
    total_score_2 = 0
    all_results_2 = []
    all_failed_2 = []
    all_successful_indices_2 = []
    all_image_indices_2 = []

    # Process in the dataset from the successful indices
    df2 = df.loc[all_successful_indices]

    # Add top-5 categories to the DataFrame
    df2["top_5_categories"] = all_results
    df2["top_p"] = [top_p for _ in all_successful_images_count]

    batches = list(split_dataframe(df2, batch_size))

    bar = tqdm(total=len(batches), desc="Processing batches")
    for index, batch in enumerate(batches):
        score, results, failed, successful_indices, image_indices = image_retrieval_batch(batch, index)
        total_score += score
        all_results_2.extend(results)
        all_failed_2.extend(failed)
        all_successful_indices_2.extend(successful_indices)
        all_image_indices_2.extend(image_indices)
        bar.set_description(f"Processing batches - Average cumulative score: {total_score/len(df2):.2%}")
        bar.update(1)
    bar.close()

    print(f"Total score in successful indices: {total_score/len(df2):.2%}")
    print(f"Total score in full dataset: {total_score/len(df):.2%}")

    print("Evaluating image re-ranking stage...")

    top_k = [1, 5, 10]
    batch_size = 100
    total_score = [0] * len(top_k)
    all_results_3 = [[] for _ in top_k]
    all_failed_3 = [[] for _ in top_k]
    all_successful_indices_3 = [[] for _ in top_k]
    all_image_indices_3 = [[] for _ in top_k]

    # Process in the dataset from the successful indices
    df3 = df2.reset_index(drop=True).loc[all_successful_indices_2].reset_index(drop=True)

    # Add retrieved image indices to the DataFrame
    df3['image_ids'] = all_image_indices_2

    batches = list(split_dataframe(df3, batch_size))

    bar = tqdm(total=len(batches), desc="Processing batches")
    for index, batch in enumerate(batches):
        scores, results, failed, successful_indices, image_indices = reranking_batch_with_reranker(batch, index, top_k)
        total_score = [total_score[i] + scores[i] for i in range(len(top_k))]
        for i in range(len(top_k)):
            all_results_3[i].extend(results[i])
            all_failed_3[i].extend(failed[i])
            all_successful_indices_3[i].extend(successful_indices[i])
            all_image_indices_3[i].extend(image_indices[i])
        bar.set_description(f"Processing batches - Average cumulative scores: Top-1: {total_score[0]/len(df3):.2%}, Top-5: {total_score[1]/len(df3):.2%}, Top-10: {total_score[2]/len(df3):.2%}")
        bar.update(1)
    bar.close()

print(f'Total score in all dataset: Top-1: {total_score[0]/len(df):.2%}, Top-5: {total_score[1]/len(df):.2%}, Top-10: {total_score[2]/len(df):.2%}')
print(f'Total score in successful indices: Top-1: {total_score[0]/len(df3):.2%}, Top-5: {total_score[1]/len(df3):.2%}, Top-10: {total_score[2]/len(df3):.2%}')
