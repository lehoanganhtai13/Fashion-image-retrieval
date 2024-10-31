import os
import argparse
from dotenv import load_dotenv
from utils import (
    Dataset,
    EncoderClient,
    BM25Client,
    LLMEncoderClient,
    Minio,
    VectorDatabase,
    ETL
)

if __name__ == '__main__':

    # Load environment variables
    folder_path = os.path.dirname(__file__)
    load_dotenv(os.path.join(folder_path, ".env"))

    # Define argument parser
    parser = argparse.ArgumentParser(description='Download, unzip and preprocess the dataset')
    parser.add_argument('--url', type=str, required=True, help='URL to download the dataset')
    parser.add_argument('--folder_name', type=str, required=True, help='Folder name to store the dataset')
    parser.add_argument('--test', action='store_true', help='Use the test dataset')
    # parser.add_argument('--skip_proxy', action='store_true', help='Skip proxy for the given host')
    # parser.add_argument('--no_proxy_host', type=str, default=None, help='Host to skip proxy')
    parser.add_argument('--model_id', type=str, default='openai/clip-vit-base-patch32', help='ID of the CLIP model')
    parser.add_argument('--device', type=str, default='cpu', help='Device used to run the model')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite the existing collection in Milvus')
    parser.add_argument('--fetch_only', action='store_true', help='Fetch the dataset only')
    parser.add_argument('--load_only', action='store_true', help='Load the embeddings to the database only')
    parser.add_argument('--num_sample', type=int, default=None, help='Number of samples to process')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size to use for encoding')
    parser.add_argument('--collection_type', type=str, default='all', help='Type of collection to process')
    args = parser.parse_args()

    if args.fetch_only and args.load_only:
        print('Please provide only one of the options --fetch_only or --load_only')
        exit()

    # Set which host to skip proxy
    os.environ['NO_PROXY'] = os.getenv('NO_PROXY_HOST', 'None')

    if not args.load_only:
        # Download and unzip the dataset at the 
        # beginning of the script if not in load_only mode
        dataset = Dataset(
            url=args.url,
            folder_name=args.folder_name
        )
        status = dataset.run()

        # Translate the dataset to English
        if args.test:
            file_name='GLAMI-1M-test.csv'
        else:
            file_name='GLAMI-1M-train.csv'
        if status == 0:
            dataset.translate(file_name=file_name)

    if args.fetch_only:
        print('Dataset fetched successfully!')
        exit()

    # ================== Process images ==================

    if args.collection_type == 'image' or args.collection_type == 'all':

        # Initialize the CLIP encoder
        print('Initializing the CLIP encoder...')
        encoder = EncoderClient(
            host=os.getenv('MODEL_SERVING_HOST'),
            port=os.getenv('MODEL_SERVING_PORT'),
        )

        # Initialize the Milvus client and create a collection
        print('Initializing the Milvus client...')
        milvus_client = VectorDatabase(
            host=os.getenv('MILVUS_HOST'),
            port=os.getenv('MILVUS_PORT')
        )
        if args.overwrite:
            milvus_client.create_collection(
                collection_type='image',
                collection_name=os.getenv('MILVUS_COLLECTION'),
                dimension=512,
                index_type=os.getenv('MILVUS_INDEX_TYPE'),
                metric_type=os.getenv('MILVUS_METRIC_TYPE')
            )

        # Initialize the Minio client and create a bucket
        print('Initializing the Minio client...')
        bucket_name = os.getenv('MINIO_BUCKET_NAME')
        minio_client = Minio(
            endpoint=os.getenv('MINIO_ENDPOINT'),
            access_key=os.getenv('MINIO_ACCESS_KEY_ID'),
            secret_key=os.getenv('MINIO_SECRET_ACCESS_KEY'),
            secure=False
        )
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        # Initialize the BM25 client
        print('Initializing the BM25 client...')
        bm25_client = BM25Client(
            storage=minio_client,
            bucket_name=os.getenv('MINIO_BUCKET_NAME'),
        )

        # The pipeline to extract the dataset, embed the images, save 
        # the images to storage and load the embeddings to the database
        etl = ETL(
            folder_name=args.folder_name,
            encoder=encoder,
            BM25_encoder=bm25_client,
            storage_client=minio_client,
            db_client=milvus_client,
            collection_name=os.getenv('MILVUS_COLLECTION'),
            bucket_name=bucket_name,
            collection_type='image',
        )
        etl.run(
            num_sample=args.num_sample,
            test_data=args.test,
            batch_size=args.batch_size,
            csv_file_name=file_name.replace('.csv', '-en.csv'),
            # dataset_size=10000
        )

    # ================== Process product names ==================

    if args.collection_type == 'name' or args.collection_type == 'all':

        # Initialize the LLM encoder
        print('Initializing the LLM encoder...')
        LLMEncoder = LLMEncoderClient(
            host=os.getenv('MODEL_SERVING_HOST'),
            port=os.getenv('EMBEDDER_MODEL_SERVING_PORT'),
        )

        # Initialize the Milvus client and create a collection
        print('Initializing the Milvus client...')
        milvus_client = VectorDatabase(
            host=os.getenv('MILVUS_HOST'),
            port=os.getenv('MILVUS_PORT')
        )
        if args.overwrite:
            milvus_client.create_collection(
                collection_type='name',
                collection_name=os.getenv('MILVUS_NAME_COLLECTION'),
                dimension=768,
                index_type=os.getenv('MILVUS_INDEX_TYPE'),
                metric_type=os.getenv('MILVUS_METRIC_TYPE')
            )

        # The pipeline to extract the dataset, preprocess the product names 
        etl = ETL(
            folder_name=args.folder_name,
            encoder=LLMEncoder,
            BM25_encoder=None,
            storage_client=None,
            db_client=milvus_client,
            collection_name=os.getenv('MILVUS_NAME_COLLECTION'),
            bucket_name='',
            collection_type='name',
        )
        etl.run(
            num_sample=args.num_sample,
            test_data=args.test,
            batch_size=args.batch_size,
            csv_file_name=file_name.replace('.csv', '-en.csv'),
            # dataset_size=10000
        )

    # ================== Process product categories ==================

    if args.collection_type == 'category' or args.collection_type == 'all':

        # Initialize the LLM encoder
        print('Initializing the LLM encoder...')
        LLMEncoder = LLMEncoderClient(
            host=os.getenv('MODEL_SERVING_HOST'),
            port=os.getenv('EMBEDDER_MODEL_SERVING_PORT'),
        )

        # Initialize the Milvus client and create a collection
        print('Initializing the Milvus client...')
        milvus_client = VectorDatabase(
            host=os.getenv('MILVUS_HOST'),
            port=os.getenv('MILVUS_PORT')
        )
        if args.overwrite:
            milvus_client.create_collection(
                collection_type='category',
                collection_name=os.getenv('MILVUS_CATEGORY_COLLECTION'),
                dimension=768,
                index_type=os.getenv('MILVUS_INDEX_TYPE'),
                metric_type=os.getenv('MILVUS_METRIC_TYPE')
            )

        # The pipeline to extract the dataset, preprocess the product categories
        etl = ETL(
            folder_name=args.folder_name,
            encoder=LLMEncoder,
            BM25_encoder=None,
            storage_client=None,
            db_client=milvus_client,
            collection_name=os.getenv('MILVUS_CATEGORY_COLLECTION'),
            bucket_name='',
            collection_type='category',
        )
        etl.run(
            num_sample=args.num_sample,
            test_data=args.test,
            batch_size=args.batch_size,
            csv_file_name=file_name.replace('.csv', '-en.csv'),
            # dataset_size=10000
        )
