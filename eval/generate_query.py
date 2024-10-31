import os
import sys
import time
import shutil
import requests

from dotenv import load_dotenv
import pandas as pd
from loguru import logger
from tqdm import tqdm

# Add the services/app directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../services/etl')))

from utils import ETL

system_prompt = (
    "You are a highly knowledgeable, efficient, and direct AI assistant. "
    "Employ multi-step reasoning to provide concise answers that focus on key information. "
    "Offer tactful suggestions to improve outcomes and actively engage in productive collaboration with the user. "
    "Ensure all your responses are in complete grammatically correct, straightforward, beautifully formatted and well-written within maximum 255 tokens. "
    "Do not write any special characters or emojis. "
    "Answer the user's question in a way that is easy to understand and provides the most helpful information. Avoid providing unnecessary information.\n"
)

example = (
    "With `product name = \"Canvas bag Kbas with Aztec pattern orange-black\"` and `product category = \"handbags\"` can be combined into `Orange-black Aztec pattern canvas handbag by Kbas` or `Kbas handbags with orange-black Aztec pattern`. "
    "Remember to keep the key clothing item in product category unchanged and the phrase must include the meaning of all the words in the original phrase."
)


if __name__ == '__main__':

    # Remove all handlers associated with the root logger object
    logger.remove()
    # Configure the logger
    logger.add(
        "./logs/generate.log",
        rotation="5 GiB",
        format="{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level} | {name}:{function}:{line} - {message}"
    )

    load_dotenv('../services/etl/.env')
    host = os.getenv('NO_PROXY_HOST', 'None')
    os.environ['NO_PROXY'] = host

    # Clone the dataset from the ETL directory
    os.makedirs('./dataset', exist_ok=True)
    source = './../services/etl/dataset/GLAMI-1M-test-dataset/GLAMI-1M-test-en.csv'
    destination = './dataset/GLAMI-1M-test-en.csv'
    shutil.copy(source, destination)

    # Load the dataset
    df = pd.read_csv('./dataset/GLAMI-1M-test-en.csv')

    df_processor = ETL(
        folder_name="",
        encoder=None,
        storage_client=None,
        db_client=None,
        collection_name="",
        bucket_name="",
    )

    batch_size = 1000
    df["processed_category_name"] = None
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch_category_names = df["category_name"][start:end]
        batch_processed_names = df_processor.preprocess_category_names_in_batch(batch_category_names)
        df.iloc[start:end, df.columns.get_loc("processed_category_name")] = batch_processed_names
    # Drop the original category name column
    df = df.drop(columns=["category_name"])
    print(f'The dataset contains {len(df)} items')

    print('Starting query generation process...')
    logger.info('Starting query generation process...')

    max_new_tokens = 255
    url = f"http://{host}:8002/generate?max_new_tokens={max_new_tokens}"

    bar = tqdm(total=len(df))
    for i, row in df.iterrows():
        product_name = row["name_en"]
        product_category = row["processed_category_name"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Combine the product name '{product_name}' and product category '{product_category}' "
                                        "into a new phrase that describes the item in a natural and coherent way, "
                                        "keeping the key information from both. For example:\n{example}\nOnly return the new phrase without any additional explanation."}
        ]

        try:
            response = requests.post(
                url,
                headers = {
                    'accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                json = messages
            ).json()["response"]["content"]

            df.loc[i, "query"] = response
            logger.info(f"Query generated for product {product_name} in category {product_category} - {response}")
            bar.update(1)

        except Exception as e:
            logger.error(f"Failed to generate a new phrase for product {product_name} in category {product_category}")
            bar.update(1)
            continue

    df.to_csv('./dataset/GLAMI-1M-test-en-queries.csv', index=False)
    bar.close()

    print('Query generation process completed!')
    logger.info('Query generation process completed!')

