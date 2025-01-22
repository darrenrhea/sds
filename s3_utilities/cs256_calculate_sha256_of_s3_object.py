from maybe_get_sha256_of_s3_object import (
     maybe_get_sha256_of_s3_object
)
from delete_s3_object import (
     delete_s3_object
)
import textwrap
from copy_s3_object_url_to_this_destination_with_sha256_checksum import (
     copy_s3_object_url_to_this_destination_with_sha256_checksum
)


def cs256_calculate_sha256_of_s3_object(s3_object_url: str) -> str:
    """
    Calculate the SHA256 hash of an S3 object.
    :param bucket_name: The name of the bucket containing the object.
    :param key: The key of the object.
    :return: The SHA256 hash of the object.
    """

    maybe_sha256_in_hexidecimal = maybe_get_sha256_of_s3_object(
        s3_object_url=s3_object_url
    )
    
    if maybe_sha256_in_hexidecimal is not None:
        return maybe_sha256_in_hexidecimal

    print("We need to calculate the sha256 of the object by the copy trick.")
    print("This will take a little bit of time.")
    # If we get here, then the object does not have a sha256 attached to it.
    # use the copy trick to get the sha256

    assert s3_object_url.startswith('s3://')
    s3_object_url_without_prefix = s3_object_url[5:]
    bucket_name, key = s3_object_url_without_prefix.split('/', 1)

    src_s3_object_url = s3_object_url
    dst_s3_object_url = f"s3://{bucket_name}/{key}_thisisacopy"
    
    sha256_in_hexidecimal = copy_s3_object_url_to_this_destination_with_sha256_checksum(
        src_s3_object_url=src_s3_object_url,
        dst_s3_object_url=dst_s3_object_url,
    )
    
    # delete the original object:
    delete_s3_object(
        s3_object_url=src_s3_object_url
    )
    
    # copy the copy back to the original:
    sha256_in_hexidecimal_second_time = copy_s3_object_url_to_this_destination_with_sha256_checksum(
        src_s3_object_url=dst_s3_object_url,
        dst_s3_object_url=src_s3_object_url,
    )
    assert sha256_in_hexidecimal == sha256_in_hexidecimal_second_time, "ERROR: The sha256 of the copy and the copy of the copy should be the same."
     # delete the copy:
    delete_s3_object(
        s3_object_url=dst_s3_object_url
    )
    
    return sha256_in_hexidecimal





