from could_be_an_s3_file_uri import (
     could_be_an_s3_file_uri
)
from cs256_calculate_sha256_of_s3_object import (
     cs256_calculate_sha256_of_s3_object
)
import argparse


def s3sha256_cli_tool() -> None:
    arp = argparse.ArgumentParser(
        description="Calculate the SHA-256 hash of an S3 object."
    )
    arp.add_argument(
        "s3_object_url",
        type=str,
        help="The URL of the S3 object."
    )
    args = arp.parse_args()
    s3_object_url = args.s3_object_url

    assert (
        could_be_an_s3_file_uri(s3_file_uri=s3_object_url)
    ), f"ERROR: Expected {s3_object_url=} to be an S3 object URL but {s3_object_url} could not possibly be an S3 object URL."


    ans = cs256_calculate_sha256_of_s3_object(
        s3_object_url=s3_object_url
    )
    print(ans)