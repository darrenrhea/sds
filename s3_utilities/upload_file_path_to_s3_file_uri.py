import textwrap
from get_bucket_name_and_s3_key_from_s3_file_uri import (
     get_bucket_name_and_s3_key_from_s3_file_uri
)

from could_be_an_s3_file_uri import (
     could_be_an_s3_file_uri
)
import boto3
import botocore
from pathlib import Path
from colorama import Fore, Style
import pprint as pp
from hash_tools import base64_to_hexidecimal
from typing import Optional


def upload_file_path_to_s3_file_uri(
    file_path: Path,
    s3_file_uri: str,
    expected_hexidecimal_sha256: Optional[str],
    verbose: bool
):
    """
    Uploads the file at `file_path` to s3 at `s3_file_uri`.
    TODO: fix that it cannot upload files larger than 5GB,
    and make sure it puts the sha256 on there.

    time upload_to_s3 /media/drhea/muchspace/clips/chicago4k_inning1_slightly_longer.mp4

    An error occurred (EntityTooLarge) when calling the PutObject operation: Your proposed upload exceeds the maximum allowed size.
    """
    print(
        textwrap.dedent(
            f"""\
            {Fore.YELLOW}
            Essentially doing:
            aws s3 cp {file_path} {s3_file_uri}
            {Style.RESET_ALL}
            """
        )
    )
    assert isinstance(file_path, Path), f"{file_path} is not a Path."
    file_path = file_path.resolve()
    assert file_path.is_file(), f"{file_path} is not a file."
    assert could_be_an_s3_file_uri(s3_file_uri), f"There is no way that {s3_file_uri} is an s3 file uri."
    s3_client = boto3.client('s3')

    bucket, object_name = get_bucket_name_and_s3_key_from_s3_file_uri(
        s3_file_uri=s3_file_uri
    )

    try:
        # botocore.client.S3.upload_file
        with open(file_path, "rb") as fp:
            
            response = s3_client.put_object(
                Body=fp,
                Bucket=bucket,
                Key=object_name,
                ChecksumAlgorithm='sha256'
            )
    except botocore.exceptions.ClientError as error: 
        print(error)       
        return False

    base64_encoded_sha256 = response["ResponseMetadata"]["HTTPHeaders"]["x-amz-checksum-sha256"]
    hexidecimal_aws_calculated_sha256 = base64_to_hexidecimal(base64_encoded_sha256)
    if expected_hexidecimal_sha256 is not None:
        assert (
            hexidecimal_aws_calculated_sha256 == expected_hexidecimal_sha256
        ), f"{hexidecimal_aws_calculated_sha256} != {expected_hexidecimal_sha256}"
    if verbose:
        print("The response was:")
        pp.pprint(response)
        print(f"According to Amazon S3, the sha256 of the file in base64 is: {base64_encoded_sha256}")
        print(f"According to Amazon S3, the sha256 of the file in hexidecimal is: {hexidecimal_aws_calculated_sha256}")
    return True

