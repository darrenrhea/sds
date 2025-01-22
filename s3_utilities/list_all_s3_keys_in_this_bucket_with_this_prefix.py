"""
this is basically:
aws s3api list-objects-v2 --bucket awecomai-shared --prefix sha256/af6e2c6e133dc66
"""

import json
import subprocess

def list_all_s3_keys_in_this_bucket_with_this_prefix(
    bucket: str,
    prefix: str,
) -> dict:
    """
    List the objects in the bucket with the given prefix.
    It is already "recursive", i.e. slashes are no different than any other character.

    Parameters:
    - bucket: The name of the bucket.
    - prefix: The prefix to filter the objects by.

    Returns:
    - A dictionary with the objects in the bucket with the given prefix.
    """
    cmd = [
        "aws",
        "s3api",
        "list-objects-v2",
        "--bucket",
        bucket,
        "--prefix",
        prefix
    ]
    ans = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    obj = json.loads(ans.stdout)
    
    # if nothing is found
    if "Contents" not in obj:
        assert "RequestCharged" in obj, "The key 'RequestCharged' should be in the response."
        return []
    
    keys = [
        obj["Contents"][i]["Key"]
        for i in range(len(obj["Contents"]))
    ]
    return keys
    