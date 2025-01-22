from hash_tools import (
     base64_to_hexidecimal
)

import json
import subprocess


def copy_s3_object_url_to_this_destination_with_sha256_checksum(
    src_s3_object_url: str,
    dst_s3_object_url: str
) -> str:
    """
    This only works for objects that are less than 5GB in size.
    create-multipart-upload
    https://docs.aws.amazon.com/AmazonS3/latest/API/API_CreateMultipartUpload.html
    """
    assert isinstance(src_s3_object_url, str), f"src_s3_object_url should be a string, but you gave something of type {type(src_s3_object_url)=} namely {src_s3_object_url=}"
    assert isinstance(dst_s3_object_url, str), f"dst_s3_object_url should be a string, but you gave something of type {type(dst_s3_object_url)=} namely {dst_s3_object_url=}"

    src_s3_object_url_without_prefix = src_s3_object_url[5:]
    src_bucket_name, src_key = src_s3_object_url_without_prefix.split('/', 1)

    dst_s3_object_url_without_prefix = dst_s3_object_url[5:]
    dst_bucket_name, dst_key = dst_s3_object_url_without_prefix.split('/', 1)

    args = [
        'aws',
        's3api',
        'copy-object',
        '--copy-source',
        f'{src_bucket_name}/{src_key}',
        '--checksum-algorithm',
        'SHA256',
        '--bucket',
        dst_bucket_name,
        '--key',
        dst_key,
        '--output',
        'json'
    ]
    print("doing")
    print("  \\\n".join(args))
    completed_process = subprocess.run(
        args=args,
        capture_output=True,
    )
    result = json.loads(
        completed_process.stdout.decode('utf-8')
    )

    sha256_in_base64 = result["CopyObjectResult"]["ChecksumSHA256"]
    
    print(f"{sha256_in_base64=}")

    sha256 = base64_to_hexidecimal(
        b64_str=sha256_in_base64
    )
    print(f"{sha256=}")
    return sha256
