import sys
import os
import time

from dotenv import load_dotenv
import pandas as pd
from loguru import logger
import argparse

# Add the services/app directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../services/etl')))

from utils import Dataset


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Download and unzip the training dataset')
    parser.add_argument('--url', type=str, required=True, help='URL to download the dataset')
    parser.add_argument('--folder_name', type=str, required=True, help='Folder name to store the dataset')
    args = parser.parse_args()

    # Set which host to skip proxy
    load_dotenv('../services/etl/.env')
    os.environ['NO_PROXY'] = os.getenv('NO_PROXY_HOST', 'None')

    dataset = Dataset(
        url=args.url,
        folder_name=args.folder_name
    )
    dataset.run()