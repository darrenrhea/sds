
from typing import Tuple


def get_bucket_name_and_s3_key_from_s3_file_uri(
    s3_file_uri: str
) -> Tuple[str, str]:
    bucket_name, s3_key = s3_file_uri[5:].split("/", 1)
    return bucket_name, s3_key

