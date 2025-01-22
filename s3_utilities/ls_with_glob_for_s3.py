from pathlib import PurePath
from typing import List
from list_all_s3_keys_in_this_bucket_with_this_prefix import (
     list_all_s3_keys_in_this_bucket_with_this_prefix
)


def ls_with_glob_for_s3(
    s3_pseudo_folder: str,
    glob_pattern: str,
    recursive: bool = False  
) -> List[str]:
    """
    This is the implementation of the s3ls CLI tool.
    This way it can be tested.
    """
    # allow people to not say the redundant s3:// part
    # but also allow it to be there since often people copy-pasta stuff

    if s3_pseudo_folder.startswith('s3://'):
        s3_pseudo_folder = s3_pseudo_folder[5:]

    if not s3_pseudo_folder.endswith('/'):
        s3_pseudo_folder += '/'
    
    bucket, prefix = s3_pseudo_folder.split('/', 1)
    # print(f"{bucket=}, {prefix=}", file=sys.stderr)

    # note this is inherently recursive:
    L = list_all_s3_keys_in_this_bucket_with_this_prefix(
        bucket=bucket,
        prefix=prefix,
    )

    list_of_strings = []
    for key in L:
        assert key.startswith(prefix)
        after_prefix = key[len(prefix):]
        if recursive:
            it_matches = PurePath(after_prefix).match(glob_pattern)
        if not recursive:
            it_matches = PurePath(after_prefix).match(glob_pattern)
            it_matches &= '/' not in after_prefix
            
        if it_matches:
            list_of_strings.append(
                f"s3://{bucket}/{key}"
            )
    return list_of_strings

