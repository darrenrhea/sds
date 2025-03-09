import pprint
from could_be_an_s3_file_uri import (
     could_be_an_s3_file_uri
)
import boto3
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse
import concurrent.futures


def download_file(s3_uri, local_path):
    """
    Downloads a file from the given S3 URI to the specified local path.
    """
    # Parse the S3 URI. Expected format: s3://bucket/key
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')  # remove leading slash

    # Create an S3 client
    s3 = boto3.client('s3')

    # Download the file from S3 to the local file path
    s3.download_file(bucket, key, local_path)

    # Print status to indicate download is complete
    print(f"Downloaded {s3_uri} to {local_path}")


def download_s3_file_uris_to_file_paths(
    src_s3_file_uri_dst_file_path_pairs: List[Tuple[str, Path]],
    max_workers: int,
    verbose: bool = True
):
    """
    Downloads a bunch of s3 file uris to local file paths concurrently.
    
    Parameters:
        src_s3_file_uri_dst_file_path_pairs (list of tuples): Each tuple contains:
            - s3_uri (str): source S3 URI (format: 's3://bucket/key').
            - local_file (str): Absolute path to write a local file
        max_workers (int): Maximum number of parallel threads (default: 10).
    """

    for pair in src_s3_file_uri_dst_file_path_pairs:
        assert isinstance(pair, tuple), f"ERROR: {pair} is not a tuple"
        assert len(pair) == 2, f"ERROR: {pair} does not have 2 elements"
        s3_uri, local_file = pair
        assert isinstance(s3_uri, str), f"ERROR: {s3_uri} is not a string"
        assert could_be_an_s3_file_uri(s3_uri), f"ERROR: {s3_uri} is not an S3 URI"
        
        assert isinstance(local_file, Path), f"ERROR: {local_file} is not a Path"
        assert local_file.parent.is_dir(), f"ERROR: {local_file.parent} is not an extant directory"

    if verbose:
        print("Download these s3 locations to these local files:")
        pprint.pprint(src_s3_file_uri_dst_file_path_pairs)

    # Use ThreadPoolExecutor to download files concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks.
        future_to_file = {
            executor.submit(download_file, s3_uri, local_path): (s3_uri, local_path)
            for s3_uri, local_path in src_s3_file_uri_dst_file_path_pairs
        }

        # Process results as they complete.
        for future in concurrent.futures.as_completed(future_to_file):
            s3_uri, local_path = future_to_file[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Error downloading {s3_uri} to {local_path}: {exc}")

