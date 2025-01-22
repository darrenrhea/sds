from list_all_s3_buckets import (
     list_all_s3_buckets
)
from pathlib import PurePath
from typing import List
from list_all_s3_keys_in_this_bucket_with_this_prefix import (
     list_all_s3_keys_in_this_bucket_with_this_prefix
)


def ls_with_glob_for_all_of_these_buckets(
    buckets: List[str],
    glob_pattern: str,
) -> List[str]:
    """
    This is the implementation of the s3ls CLI tool.
    This way it can be tested.
    """
    
   
    list_of_strings = []
    for bucket in buckets:
        print("Looking in bucket:", bucket)
        prefix = ""

        # note this is inherently recursive:
        L = list_all_s3_keys_in_this_bucket_with_this_prefix(
            bucket=bucket,
            prefix=prefix,
        )

        
        for key in L:
            assert key.startswith(prefix)
            after_prefix = key[len(prefix):]
            it_matches = PurePath(after_prefix).match(glob_pattern)
            
            if it_matches:
                list_of_strings.append(
                    f"s3://{bucket}/{key}"
                )
    
    return list_of_strings


if __name__ == "__main__":

    buckets = ["awecomai-original-videos", "awecomai-test-videos"]
    buckets = list_all_s3_buckets()
    # remove bad buckets:
    bad =[
        ""
    ]
    
    list_of_strings = ls_with_glob_for_all_of_these_buckets(
        buckets=buckets,
        glob_pattern="*DSCF0561*"
    )

    for s in list_of_strings:
        print(s)