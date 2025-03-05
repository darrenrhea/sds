from print_red import (
     print_red
)
from print_yellow import (
     print_yellow
)
from is_sha256 import (
     is_sha256
)
from list_all_s3_keys_in_this_bucket_with_this_prefix import (
     list_all_s3_keys_in_this_bucket_with_this_prefix
)
import sys
import textwrap
from typing import Optional



def gaes3fuwts_get_an_extant_s3_file_uri_with_this_sha256(
    sha256: str,
) -> Optional[str]:
    """
    Sadly, we kept the extension on the file name when we stored files in s3 by their sha256.
    This makes it complicated to download the file from s3, because we need to know the extension.

    This finds it on s3, it will return None as a sign of failure.
    """
    the_sha256_hash = sha256
    assert (
        is_sha256(the_sha256_hash)
    ), f"ERROR: sha256 must be a string!, but you gave {sha256=} which has {type(sha256)=}"
    
    bucket_we_are_looking_in = "awecomai-shared"

    s3_subdirectory_we_are_looking_in = "sha256/" # should work if = ""
    
    prefix_we_are_looking_in = f"{s3_subdirectory_we_are_looking_in}{the_sha256_hash}"

    keys = list_all_s3_keys_in_this_bucket_with_this_prefix(
        bucket="awecomai-shared",
        prefix=prefix_we_are_looking_in
    )

    where_we_are_looking = f"s3://{bucket_we_are_looking_in}{s3_subdirectory_we_are_looking_in}/"

    if len(keys) == 0:
        print_red(
            textwrap.dedent(
                f"""\
                Note: The file with sha256 hash
                
                {the_sha256_hash}
                
                does not exist in s3.
                
                You can try for instance:

                aws s3 ls {where_we_are_looking}{the_sha256_hash}

                and you will see that it does not exist.
                """
            )
        )
        return None
    elif len(keys) > 1:
        print_yellow("WARNING: There are more than one file in {where_we_are_looking} with the prefix {the_sha256_hash}")
        for k in keys:
            print(k)
        sys.exit(1)
    
    key = keys[0]
    file_name = key.split("/")[-1]
    full_length_sha256 = file_name.split(".")[0]
    extension = file_name[len(full_length_sha256):]
    
    assert len(full_length_sha256) == 64
    assert full_length_sha256[:len(the_sha256_hash)] == the_sha256_hash
    return f"s3://{bucket_we_are_looking_in}/{key}"

   


