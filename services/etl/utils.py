import io
import math
import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import List

import cudf
import pandas as pd
import requests
import time
from PIL import Image
from tqdm import tqdm

from deep_translator import GoogleTranslator
from minio import Minio
from pymilvus import MilvusClient, DataType, RRFRanker, Collection, connections, AnnSearchRequest
from unlimited_machine_translator.translator import machine_translator_df
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus.model.sparse import BM25EmbeddingFunction


class Dataset():
    """Dataset class to download and unzip the dataset."""
    def __init__(self, url: str, folder_name: str) -> None:
        self.url = url
        self.folder_name = folder_name

    def download(self):
        """Download the dataset from the given url."""

        root_folder_path = os.getcwd()
        parent_dataset_folder_path = os.path.join(root_folder_path, 'dataset')

        dataset_path = os.path.join(parent_dataset_folder_path, self.folder_name)
        if os.path.exists(dataset_path):
            print(f'{self.folder_name} dataset already exists')
            return -1

        if not os.path.exists(parent_dataset_folder_path):
            os.makedirs(parent_dataset_folder_path)

        response = requests.get(self.url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte

        with open(f'{self.folder_name}.zip', 'wb') as file, tqdm(
                desc=f'{self.folder_name}.zip',
                total=total_size_in_bytes,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))
        return 0

    def unzip(self):
        """Unzip the downloaded dataset."""

        # Create the target directory if it doesn't exist
        target_dir = os.path.join('dataset', self.folder_name)
        os.makedirs(target_dir, exist_ok=True)

        # Create a temporary directory for extraction
        temp_dir = os.path.join('dataset', 'temp_unzip')
        os.makedirs(temp_dir, exist_ok=True)

        with zipfile.ZipFile(f'{self.folder_name}.zip', 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find the first directory in the temporary directory
        extracted_folder = None
        for item in os.listdir(temp_dir):
            item_path = os.path.join(temp_dir, item)
            if os.path.isdir(item_path):
                extracted_folder = item_path
                break

        # Move contents from the extracted folder to the target directory
        if extracted_folder:
            for item in os.listdir(extracted_folder):
                s = os.path.join(extracted_folder, item)
                d = os.path.join(target_dir, item)
                shutil.move(s, d)
        else:
            # If no directory is found, move all files from temp_dir to target_dir
            print('No directory found in the zip file. Moving all files to the target directory...')
            for item in os.listdir(temp_dir):
                s = os.path.join(temp_dir, item)
                d = os.path.join(target_dir, item)
                shutil.move(s, d)

        # Remove the temporary directory
        shutil.rmtree(temp_dir)

        # Remove the zip file
        print(f'Removing {self.folder_name}.zip...')
        os.remove(f'{self.folder_name}.zip')

    def translate(self, file_name):
        """Translate the dataset to English using the Google Translate API."""

        root_folder_path = os.getcwd()
        df = pd.read_csv(os.path.join(root_folder_path, 'dataset', self.folder_name, file_name))

        current_dir = os.path.join(root_folder_path, 'dataset', 'temp_translating')
        os.makedirs(current_dir, exist_ok=True)

        df = df.drop_duplicates(subset=["image_id"])

        # Translate the dataset
        df = machine_translator_df(
            data_set=df,
            column_name="name",
            source_language="auto",
            target_language="en",
            Translator=GoogleTranslator,
            current_wd=current_dir
        )

        df = df[["image_id", "name_en", "category_name", "image_file"]].reset_index(drop=True)
        
        new_file_name = file_name.replace('.csv', '-en.csv')
        df.to_csv(os.path.join(root_folder_path, 'dataset', self.folder_name, new_file_name), index=False)
        shutil.rmtree(current_dir, ignore_errors=True)
        
        print(f'Translated dataset saved as {new_file_name}')

    def run(self):
        """Run the dataset download and unzip process."""

        print(f'Downloading {self.folder_name}.zip...')
        done = self.download()
        if done == -1:
            return -1
        print(f'Unzipping {self.folder_name}.zip...')
        self.unzip()
        print('Dataset is ready to use!')
        return 0


class EncoderClient():
    """EncoderClient class for text and image encoding operations."""
    def __init__(self, host: str = "localhost", port: str = "8000") -> None:
        self.host = host
        self.port = port

    def get_device(self):
        response = requests.get(f'http://{self.host}:{self.port}/get_device')
        return response.json()["device"]

    def encode_text(self, text: List[str]):
        response = requests.post(
            f'http://{self.host}:{self.port}/encode-text',
            json=text,
            timeout=300
        )
        try:
            result = response.json()["embeddings"]
            return result
        except Exception as e:
            print(f'Encoding {len(text)} texts...')
            print(text)
            print(response.status_code)
            print(response.text)
            print(response.json())
        # return response.json()["embeddings"]

    def encode_image(self, images: List[Image.Image]):
        # Convert the PIL images to bytes
        files = []
        for i, image in enumerate(images):
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            files.append(("files", (f"image_{i}.png", img_byte_arr, "image/png")))

        # Send the images to the server
        response = requests.post(
            f'http://{self.host}:{self.port}/encode-image',
            files=files,
            timeout=300
        )
        return response.json()["embeddings"]


class LLMEncoderClient():
    """LLMEncoderClient class for text encoding operations."""
    def __init__(self, host: str = "localhost", port: str = "8001") -> None:
        self.host = host
        self.port = port

    def get_device(self):
        response = requests.get(f'http://{self.host}:{self.port}/get_device')
        return response.json()["device"]

    def encode_text(self, text: List[str]):
        response = requests.post(
            f'http://{self.host}:{self.port}/embed-docs',
            headers={
                'accept': 'application/json',
                'Content-Type': 'application/json'
            },
            json=text
        )
        return response.json()["doc_embeddings"]
    
    def encode_query(self, query: List[str]):
        response = requests.post(
            f'http://{self.host}:{self.port}/embed-query',
            headers={
                'accept': 'application/json',
                'Content-Type': 'application/json'
            },
            json=query
        )
        return response.json()["query_embeddings"]


class BM25Client():
    def __init__(self, language: str = "en", path: str = "", storage: Minio = None, bucket_name: str = "", remove_after_load: bool = False) -> None:
        self.analyzer = build_default_analyzer(language=language)
        self.bm25 = BM25EmbeddingFunction(analyzer=self.analyzer)
        self.storage_client = storage
        self.bucket_name = bucket_name

        # Load the BM25 state dict if path is provided
        if path:
            self.bm25.load(path)
        elif self.bucket_name and self.storage_client:
            if not os.path.exists("./bm25_state_dict.json"):
                print("Downloading the BM25 state dict...")
                retry = 3
                while not os.path.exists("./bm25_state_dict.json"):
                    try:
                        self.storage_client.fget_object(bucket_name=self.bucket_name, object_name="bm25/state_dict.json", file_path="./bm25_state_dict.json")
                    except Exception as e:
                        print(e)
                        # Wailt for the file to be ready if exists
                        for i in range(5):
                            if os.path.exists("./bm25_state_dict.json"):
                                break
                            time.sleep(1)
                        if not os.path.exists("./bm25_state_dict.json"):
                            print("Failed to download the BM25 state dict. Retrying...")
                            retry -= 1
                            if retry == 0:
                                raise ValueError("Failed to download the BM25 state dict")

            print("Loading the BM25 state dict...")
            if not os.path.exists("./bm25_state_dict.json"):
                raise ValueError("BM25 state dict not found in the local directory")
            self.bm25.load("./bm25_state_dict.json")

            # Remove the state dict if exists
            if os.path.exists("./bm25_state_dict.json") and remove_after_load:
                print("Removing the BM25 state dict in the local directory...")
                os.remove("./bm25_state_dict.json")
        else:
            raise ValueError("Please provide the path to the BM25 state dict or the storage client and bucket name")

    def fit(self, data: List[str], path: str = "./bm25_state_dict.json"):
        self.bm25.fit(data)
        
        dir_ = path.rsplit("/", 1)[0]
        os.makedirs(dir_, exist_ok=True)
        self.bm25.save(path)

        # Load the BM25 state dict to MinIO starage
        if not (self.storage_client and self.bucket_name):
            raise ValueError("Please provide the storage client and bucket name to save the BM25 state dict")
        self.storage_client.fput_object(bucket_name=self.bucket_name, object_name="bm25/state_dict.json", file_path=path)

    def fit_transform(self, data: List[str], path: str = "./bm25_state_dict.json", storage: Minio = None, bucket_name: str = "") -> List:
        self.bm25.fit(data)
        
        dir_ = path.rsplit("/", 1)[0]
        os.makedirs(dir_, exist_ok=True)
        self.bm25.save(path)

        # Load the BM25 state dict to MinIO starage
        if not (self.storage_client and self.bucket_name):
            raise ValueError("Please provide the storage client and bucket name to save the BM25 state dict")
        self.storage_client.fput_object(bucket_name=self.bucket_name, object_name="bm25/state_dict.json", file_path=path)

        embeddings = list(self.bm25.encode_documents(data))
        return embeddings
    
    def encode_text(self, data: List[str]) -> List:
        embeddings = list(self.bm25.encode_documents(data))
        return embeddings

    def encode_queries(self, query: List[str]) -> List:
        embeddings = list(self.bm25.encode_queries(query))
        return embeddings
    

class ReRankerClient():
    """ReRankerClient class for re-ranking operations."""
    def __init__(self, host: str = "localhost", port: str = "8003") -> None:
        self.host = host
        self.port = port

    def get_device(self):
        response = requests.get(f'http://{self.host}:{self.port}/get_device')
        return response.json()["device"]

    def rerank(self, query: str, documents: List[str], top_k: int = 5):
        pairs = [ [query, doc] for doc in documents ]
        response = requests.post(
            f'http://{self.host}:{self.port}/re-ranking',
            headers={
                'accept': 'application/json',
                'Content-Type': 'application/json'
            },
            json=pairs,
            timeout=600
        )
        return response.json()["sorted_indices"][:top_k]


class LLMGeneratorClient():
    """LLMGeneratorClient class for text generation operations."""
    def __init__(self, host: str = "localhost", port: str = "8002") -> None:
        self.host = host
        self.port = port

    def get_device(self):
        response = requests.get(f'http://{self.host}:{self.port}/get_device')
        return response.json()["device"]

    def generate_text(self, messages: List[dict], max_new_tokens: int = 255):
        response = requests.post(
            f'http://{self.host}:{self.port}/generate?max_new_tokens={max_new_tokens}',
            headers={
                'accept': 'application/json',
                'Content-Type': 'application/json'
            },
            json=messages
        )
        return response.json()["response"]["content"]


class VectorDatabase():
    """MilvusClient class for vector database operations."""
    def __init__(self, host: str = 'localhost', port: str = '19530') -> None:
        connections.connect(host=host, port=port)
        self.client = MilvusClient(uri=f'http://{host}:{port}')
        self.reranker = RRFRanker()

    def create_collection(self, collection_type: str, collection_name: str, dimension: int = 512, index_type: str = "HNSW", metric_type: str = "COSINE", estimate_row_count: int = 60000):
        # Check if collection exists
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)

        # Check if the metric type and index type are supported
        supported_metric_types = ["L2", "COSINE"]
        assert metric_type in supported_metric_types, f"Metric type must be one of {supported_metric_types}"
        supported_index_types = ["IVF_FLAT", "HNSW"]
        assert index_type in supported_index_types, f"Index type must be one of {supported_index_types}"

        #======================Image Collection======================

        if collection_type == "image":

            # Create schema
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="image_embedding", datatype=DataType.FLOAT_VECTOR, dim=dimension)
            schema.add_field(field_name="name_embedding", datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(field_name="product_name", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="product_category", datatype=DataType.VARCHAR, max_length=65535)

            # Create index
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="id",
                index_type="STL_SORT"
            )
            index_params.add_index(
                field_name="product_category",
                index_type="INVERTED"
            )
            if index_type == "HNSW":
                index_params.add_index(
                    field_name="image_embedding",
                    index_type=index_type,
                    metric_type=metric_type,
                    params={"M": 16, "efConstruction": 500}
                )
            elif index_type == "IVF_FLAT":
                index_params.add_index(
                    field_name="image_embedding",
                    index_type=index_type,
                    metric_type=metric_type,
                    params={"nlist": 4*math.sqrt(estimate_row_count)}
                )
            index_params.add_index(
                field_name="name_embedding",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP"
            )

            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )

            print(f'Collection {collection_name} created successfully!')

        #======================Name Collection======================

        if collection_type == "name":

            # Create schema
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="name_embedding", datatype=DataType.FLOAT_VECTOR, dim=dimension)
            schema.add_field(field_name="category_name", datatype=DataType.VARCHAR, max_length=65535)

            # Create index
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="id",
                index_type="STL_SORT"
            )
            index_params.add_index(
                field_name="category_name",
                index_type="INVERTED"
            )

            if index_type == "HNSW":
                index_params.add_index(
                    field_name="name_embedding",
                    index_type=index_type,
                    metric_type=metric_type,
                    params={"M": 16, "efConstruction": 500}
                )
            elif index_type == "IVF_FLAT":
                index_params.add_index(
                    field_name="name_embedding",
                    index_type=index_type,
                    metric_type=metric_type,
                    params={"nlist": 4*math.sqrt(estimate_row_count)}
                )

            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )

            print(f'Collection {collection_name} created successfully!')

        #======================Category Collection======================

        if collection_type == "category":

            # Create schema
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="category_embedding", datatype=DataType.FLOAT_VECTOR, dim=dimension)
            schema.add_field(field_name="category_name", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="category_count", datatype=DataType.INT64)

            # Create index
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="id",
                index_type="STL_SORT"
            )
            if index_type == "HNSW":
                index_params.add_index(
                    field_name="category_embedding",
                    index_type=index_type,
                    metric_type=metric_type,
                    params={"M": 16, "efConstruction": 500}
                )
            elif index_type == "IVF_FLAT":
                index_params.add_index(
                    field_name="category_embedding",
                    index_type=index_type,
                    metric_type=metric_type,
                    params={"nlist": 4*math.sqrt(estimate_row_count)}
                )

            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )

            print(f'Collection {collection_name} created successfully!')

    def load_collection(self, collection_name: str) -> bool:
        if not self.client.has_collection(collection_name):
            print(f'Collection {collection_name} does not exist!')
            return
        
        # Load the collection
        self.client.load_collection(collection_name)

        # Check if the collection is loaded
        load_state = self.client.get_load_state(collection_name=collection_name)
        if load_state:
            print(f'Collection {collection_name} is loaded successfully!')
            return True
        else:
            print(f'Failed to load collection {collection_name}!')
            return False

    def insert_vectors(self, collection_type: str, collection_name: str, vector: list, sparse_vector: list, product_name: str, category: str, category_count: int, id: int):
        
        if collection_type == "image":
            self.client.insert(
                collection_name=collection_name,
                data={
                    "id": id,
                    "image_embedding": vector,
                    "name_embedding": sparse_vector,
                    "product_name": product_name,
                    "product_category": category
                }
            )
        elif collection_type == "name":
            self.client.insert(
                collection_name=collection_name,
                data={
                    "id": id,
                    "name_embedding": vector,
                    "category_name": category
                }
            )
        elif collection_type == "category":
            self.client.insert(
                collection_name=collection_name,
                data={
                    "id": id,
                    "category_embedding": vector,
                    "category_name": category,
                    "category_count": category_count,
                }
            )
        else:
            raise ValueError("Invalid collection type. Please provide one of 'image', 'name', 'category'")
        
    
    def get_vectors(self, collection_name: str, ids: List[int]):
        result = self.client.get(
            collection_name=collection_name,
            ids=ids
        )
        return result

    def hybrid_search(self, collection_name: str, query_embeddings: List, filtering_expr: str = "", top_k: int = 5, metric_type: str = "COSINE", index_type: str = "HNSW"):
        self.collection = Collection(collection_name)

        image_search_params = {
            "data": query_embeddings[0],
            "anns_field": "image_embedding",
            "param": {
                "metric_type": metric_type,
                "params": {"ef": top_k} if index_type == "HNSW" else {"nprobe": 8}
            },
            "limit": top_k,
            "expr": filtering_expr
        }
        image_search_request = AnnSearchRequest(**image_search_params)

        name_search_params = {
            "data": query_embeddings[1],
            "anns_field": "name_embedding",
            "param": {
                "metric_type": "IP",
                "params": {}
            },
            "limit": top_k,
            "expr": filtering_expr
        }
        name_search_request = AnnSearchRequest(**name_search_params)
        
        results = self.collection.hybrid_search(
            reqs=[image_search_request, name_search_request],
            rerank=self.reranker,
            limit=top_k,
            output_fields=["id", "product_name"]
        )
        return results

    def search_vectors(self, collection_type: str, collection_name: str, vectors: List[List], filtering_expr: str = "", top_k: int = 5, metric_type: str = "COSINE", index_type: str = "HNSW"):
        
        if collection_type == "image":
            # results = self.hybrid_search(
            #     collection_name=collection_name,
            #     query_embeddings=vectors,
            #     filtering_expr=filtering_expr,
            #     top_k=top_k,
            #     metric_type=metric_type,
            #     index_type=index_type
            # )
            results = self.client.search(
                collection_name=collection_name,
                data=vectors,
                anns_field="image_embedding",
                limit=top_k,
                output_fields=["id", "product_name"],
                search_params={
                    "metric_type": metric_type,
                    "params": {"ef": top_k} if index_type == "HNSW" else {"nprobe": 8},
                },
                filter=filtering_expr
            )
        elif collection_type == "name":
            results = self.client.search(
                collection_name=collection_name,
                data=vectors,
                anns_field="name_embedding",
                limit=top_k,
                output_fields=["id"],
                search_params={
                    "metric_type": metric_type,
                    "params": {"ef": top_k} if index_type == "HNSW" else {"nprobe": 8},
                },
                filter=filtering_expr
            )
        elif collection_type == "category":
            results = self.client.search(
                collection_name=collection_name,
                data=vectors,
                anns_field="category_embedding",
                limit=top_k,
                output_fields=["id", "category_name", "category_count"],
                search_params={
                    "metric_type": metric_type,
                    "params": {"ef": top_k} if index_type == "HNSW" else {"nprobe": 8},
                },
                filter=filtering_expr
            )
        else:
            raise ValueError("Invalid collection type. Please provide one of 'image', 'name', 'category'")
        
        return results
    

class ETL():
    """ETL class to extract, transform and load the dataset."""
    def __init__(self, folder_name: str, encoder, BM25_encoder: BM25Client, storage_client: Minio, db_client: VectorDatabase, collection_name: str, bucket_name: str, collection_type: str) -> None:
        self.dataset_folder_path = os.path.join(os.getcwd(), 'dataset', folder_name)
        self.encoder = encoder
        self.BM25_encoder = BM25_encoder
        self.storage_client = storage_client
        self.db_client = db_client
        self.collection_name = collection_name
        self.bucket_name = bucket_name
        self.collection_type = collection_type

    def get_image(self, image_path: str):
        # Resize the image to 224x224 if it's not already
        image = Image.open(image_path)
        if image.size != (224, 224):
            image = image.resize((224, 224))
        return image.convert('RGB')

    def process_images_in_batch(self, image_paths):
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.get_image, image_paths))
        return results

    def preprocess_category_name(self, category_name: str):
        category_name = category_name.replace("-", " ")
        category_name = category_name.replace("womens", "women's").replace("women s", "women's")
        category_name = category_name.replace("mens", "men's").replace("men s", "men's")
        category_name = category_name.replace("boys", "boy's").replace("boy s", "boy's")
        category_name = category_name.replace("girls", "girl's").replace("girl s", "girl's")
        category_name = category_name.replace(" and ", " or ")
        category_name = category_name.replace("t shirts", "t-shirts")
        return category_name

    def preprocess_category_names_in_batch(self, category_names):
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.preprocess_category_name, category_names))
        return results

    def get_dataset(self, num_sample: int = None, test: bool = False, csv_file_name: str = None, dataset_size: int = None):
        if not csv_file_name:
            if test:
                csv_file_name = 'GLAMI-1M-test.csv'
            else:
                csv_file_name = 'GLAMI-1M-train.csv'
        df = cudf.read_csv(self.dataset_folder_path + '/' + csv_file_name)

        # Modify image file path
        df['image_file'] = self.dataset_folder_path + '/images/' + df['image_id'].astype(str) + '.jpg'
        assert os.path.exists(df.loc[0, "image_file"])

        # Only take `category_name` and `image_file` features
        df = df[["category_name", "name_en", "image_file"]].reset_index(drop=True)

        if num_sample is not None:
            df = df[:num_sample]

        # Convert to pandas for applying the custom function
        df_pandas = df.to_pandas()
            
        batch_size = 256 # Optimal batch size for processing images, fixed after testing
        # Process images in batches
        df_pandas["image"] = None
        for start in range(0, len(df_pandas), batch_size):
            end = min(start + batch_size, len(df_pandas))
            batch_image_files = df_pandas["image_file"][start:end]
            batch_images = self.process_images_in_batch(batch_image_files)
            df_pandas.iloc[start:end, df_pandas.columns.get_loc("image")] = batch_images

        # Process category names in batches
        df_pandas["processed_category_name"] = None
        for start in range(0, len(df_pandas), batch_size):
            end = min(start + batch_size, len(df_pandas))
            batch_category_names = df_pandas["category_name"][start:end]
            batch_processed_names = self.preprocess_category_names_in_batch(batch_category_names)
            df_pandas.iloc[start:end, df_pandas.columns.get_loc("processed_category_name")] = batch_processed_names
        # Drop the original category name column
        df_pandas = df_pandas.drop(columns=["category_name"])

        if dataset_size is None:
            dataset_size = len(df_pandas)

        return df_pandas[:dataset_size]

    def run(self, num_sample: int = None, test_data: bool = False, batch_size: int = 256, csv_file_name: str = "GLAMI-1M-test-en.csv", dataset_size: int = None):
        """Run the ETL process."""
        if batch_size > 1000:
            print('Reaching the limit of the batch size. Please reduce the batch size to 1000 or lower.')
            return

        print('Extracting the dataset...')
        dataset = self.get_dataset(num_sample, test_data, csv_file_name, dataset_size)
        print(f'Number of samples in the dataset : {len(dataset)}')

        print('Loading the embeddings and images to the database and storage...')
        if test_data:
            root_bucket_name = 'images'
        else:
            root_bucket_name = 'train-images'

        if self.collection_type == "image":
            # Encode the name into sparse vectors using BM25
            print('Encoding the names into sparse vectors...')
            sparse_name_embeddings = self.BM25_encoder.fit_transform(dataset["name_en"], "./bm25/state_dict.json")

            for start in tqdm(range(0, len(dataset), batch_size), desc="Processing images"):
                end = min(start + batch_size, len(dataset))
                batch = dataset.iloc[start:end]

                # Store the batch in a list
                batch_rows = list(batch.iterrows())

                # Encode the images in the batch
                batch_images = [row["image"] for _, row in batch_rows]
                image_embeddings = self.encoder.encode_image(batch_images)
                
                # Insert the embeddings to the database
                for i, image_embedding in enumerate(image_embeddings):
                    _, row = batch_rows[i]
                    self.db_client.insert_vectors(
                        self.collection_type,
                        self.collection_name,
                        image_embedding,
                        sparse_name_embeddings[i],
                        row["name_en"],
                        row["processed_category_name"],
                        None,
                        start + i
                    )

                # batch = dataset.iloc[start:end]
                # Load images to the storage
                for i, (index, row) in enumerate(batch_rows):
                    image_file = row["image_file"]
                    object_name = f'{root_bucket_name}/{start + i}.jpg'
                    self.storage_client.fput_object(
                        bucket_name=self.bucket_name,
                        object_name=object_name,
                        file_path=image_file
                    )

        elif self.collection_type == "name":
            for start in tqdm(range(0, len(dataset), batch_size), desc="Processing names"):
                end = min(start + batch_size, len(dataset))
                batch = dataset.iloc[start:end]

                # Store the batch in a list
                batch_rows = list(batch.iterrows())

                # Encode the names in the batch
                batch_names = [row["name_en"] for _, row in batch_rows]
                name_embeddings = self.encoder.encode_text(batch_names)

                # Insert the embeddings to the database
                for i, name_embedding in enumerate(name_embeddings):
                    _, row = batch_rows[i]
                    self.db_client.insert_vectors(
                        self.collection_type,
                        self.collection_name,
                        name_embedding,
                        None,
                        None,
                        row["processed_category_name"],
                        None,
                        start + i
                    )

        elif self.collection_type == "category":
            # Create a list of categories
            print('Creating a list of categories...')
            dataset['id'] = dataset.index
            category_counts = dataset.groupby("processed_category_name").agg(
                count=("id", "size"),
                image_ids=("id", lambda x: list(x))
            ).reset_index()
            categories = [row["processed_category_name"] for _, row in category_counts.iterrows()]
            category_counts = [row["count"] for _, row in category_counts.iterrows()]

            # Encode the categories
            print('Encoding the categories...')
            category_embeddings = self.encoder.encode_text(categories)
            
            # Insert the embeddings to the database
            print('Inserting the embeddings to the database...')
            bar = tqdm(total=len(category_embeddings), desc="Inserting category embeddings")
            for i, category_embedding in enumerate(category_embeddings):
                self.db_client.insert_vectors(
                    self.collection_type,
                    self.collection_name,
                    category_embedding,
                    None,
                    None,
                    categories[i],
                    category_counts[i],
                    i
                )
                bar.update(1)
            bar.close()

        print('ETL process completed!')
        