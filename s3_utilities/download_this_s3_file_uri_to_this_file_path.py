from get_bucket_name_and_s3_key_from_s3_file_uri import (
     get_bucket_name_and_s3_key_from_s3_file_uri
)


import sys
import boto3
from pathlib import Path
import pprint as pp
from colorama import Fore, Style

# prepare for people to do: from s3_utilities import *:
__all__ = [
    "get_bucket_name_and_s3_key_from_s3_file_uri",
]


def download_this_s3_file_uri_to_this_file_path(
    s3_file_uri: str,
    file_path: Path
):
    s3 = boto3.client("s3")

    bucket_name, s3_key = get_bucket_name_and_s3_key_from_s3_file_uri(
        s3_file_uri=s3_file_uri
    )

    try:
        resp = s3.get_object(
            Bucket=bucket_name,
            Key=s3_key 
        )
        pp.pprint(resp)
    except:
        print(f"{Fore.RED}WTF")
        print(f"bucket_name is: {bucket_name}")
        print(f"s3_key is: {s3_key}")
        print(f"aws s3 ls s3://{bucket_name}/{s3_key}")
        print(f"{Style.RESET_ALL}")
        sys.exit(1)

    chunks = [x for x in resp["Body"]]
    content= b''.join(chunks)

    absolute_file_path_destination = Path(file_path).resolve()

    with open(absolute_file_path_destination, "wb+") as fp:
        fp.write(content)

