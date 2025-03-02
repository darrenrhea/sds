import pprint
import time
from get_a_temp_file_path import (
     get_a_temp_file_path
)
from could_be_an_s3_file_uri import (
     could_be_an_s3_file_uri
)
import boto3
from pathlib import Path
from typing import List, Tuple
from boto3.s3.transfer import S3Transfer
from urllib.parse import urlparse
import concurrent.futures




def upload_file(transfer, local_file, s3_uri):
    """
    Helper function to upload a single file to its designated S3 URI.
    
    Parameters:
        transfer (S3Transfer): The transfer manager instance.
        local_file (str): Absolute path of the local file.
        s3_uri (str): The S3 URI in the format 's3://bucket/key'.
        
    Returns:
        str: Success message upon completion.
    """
    parsed_uri = urlparse(s3_uri)
    if parsed_uri.scheme != 's3':
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    
    bucket = parsed_uri.netloc
    key = parsed_uri.path.lstrip('/')
    
    transfer.upload_file(local_file, bucket, key)
    return f"Uploaded {local_file} to {s3_uri}"


def upload_file_paths_to_s3_file_uris(
    src_file_path_dst_s3_file_uri_pairs: List[Tuple[Path, str]],
    max_workers: int = 10,
    verbose: bool = True
):
    """
    Uploads a list of files to their respective S3 URIs concurrently.
    
    Parameters:
        file_pairs (list of tuples): Each tuple contains:
            - local_file (str): Absolute path to the local file.
            - s3_uri (str): Destination S3 URI (format: 's3://bucket/key').
        max_workers (int): Maximum number of parallel threads (default: 10).
    
    Example:
        upload_files([
            ("/absolute/path/to/file1.txt", "s3://my-bucket/path/to/file1.txt"),
            ("/absolute/path/to/file2.txt", "s3://my-bucket/path/to/file2.txt"),
        ])
    """
    for pair in src_file_path_dst_s3_file_uri_pairs:
        assert isinstance(pair, tuple), f"ERROR: {pair} is not a tuple"
        assert len(pair) == 2, f"ERROR: {pair} does not have 2 elements"
        local_file, s3_uri = pair
        assert isinstance(local_file, Path), f"ERROR: {local_file} is not a Path"
        assert local_file.is_file(), f"ERROR: {local_file} is not a file"
        assert isinstance(s3_uri, str), f"ERROR: {s3_uri} is not a string"
        assert could_be_an_s3_file_uri(s3_uri), f"ERROR: {s3_uri} is not an S3 URI"

    if verbose:
        print("Uploading these local files to these s3 locations:")
        pprint.pprint(src_file_path_dst_s3_file_uri_pairs)
  
    s3_client = boto3.client('s3')
    transfer = S3Transfer(s3_client)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit each file upload to the thread pool
        future_to_file = {
            executor.submit(upload_file, transfer, local_file, s3_uri): (local_file, s3_uri)
            for local_file, s3_uri in src_file_path_dst_s3_file_uri_pairs
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            local_file, s3_uri = future_to_file[future]
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"Error uploading {local_file} to {s3_uri}: {e}")


# Example usage:
if __name__ == "__main__":
    num_files = 1000
    
    src_file_path_dst_s3_file_uri_pairs = []

    for file_index in range(num_files):
        temp_file_path = get_a_temp_file_path(suffix=".txt")
        temp_file_path.write_text("1" * 2**20)

        p = (
            temp_file_path,
            f"s3://awecomai-temp/crap/file{file_index}.txt"
        )
         
        src_file_path_dst_s3_file_uri_pairs.append(p)

    
    start = time.time()
    upload_file_paths_to_s3_file_uris(
       src_file_path_dst_s3_file_uri_pairs=\
       src_file_path_dst_s3_file_uri_pairs,
       
       max_workers=100
    )
    stop = time.time()
    print(f"Elapsed time: {stop - start} seconds to upload {num_files} Megabytes")
