from typing import List, Tuple
from pathlib import Path
from get_bucket_name_and_s3_key_from_s3_file_uri import (
     get_bucket_name_and_s3_key_from_s3_file_uri
)
import boto3
from concurrent.futures import ThreadPoolExecutor


def download_this_s3_file_uri_to_this_file_path(
    s3_client,  # please pass in the boto3 client
    s3_file_uri: str,
    file_path: Path
):
    
    bucket_name, s3_key = get_bucket_name_and_s3_key_from_s3_file_uri(
        s3_file_uri=s3_file_uri
    )

    try:
        s3_client.download_file(bucket_name, s3_key, file_path)
        print(f"Downloaded: {s3_file_uri} -> {file_path}")
    except Exception as e:
        print(f"Failed to download {s3_key}: {e}")


def download_list_of_s3_file_uris(
    s3_client: boto3.client,
    src_dst_pairs: List[Tuple[str, Path]]
):
    # Use ThreadPoolExecutor for parallel downloads
    max_threads = 10  # Adjust the number of threads based on your network and system capacity
    with ThreadPoolExecutor(max_threads) as executor:
        # Submit tasks to download files
        futures = [
            executor.submit(
                download_this_s3_file_uri_to_this_file_path,
                s3_client,
                s3_file_uri,
                file_path
            )
            for s3_file_uri, file_path in src_dst_pairs
        ]
        # Wait for all tasks to complete
        for future in futures:
            future.result()  # This will raise any exceptions that occurred during download

