from is_sha256 import (
     is_sha256
)
from pathlib import Path
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)


def glfptsfws_get_local_file_path_to_save_file_with_sha256(
    sha256: str,
    extension: str,
) -> Path:
    """
    Given a sha256 and a file extension such as
    .jpg, .png, .json, .json5
    this will return a file Path object where you might store
    the file with that sha256.
    """

    assert is_sha256(sha256)

    shared_dir = get_the_large_capacity_shared_directory()
    sha256_local_cache_dir = shared_dir / "sha256"
    
    d01 = sha256[0:2]
    d23 = sha256[2:4]
    d45 = sha256[4:6]
    d67 = sha256[6:8]

    sub_dir = sha256_local_cache_dir / d01 / d23 / d45 / d67
    sub_dir.mkdir(parents=True, exist_ok=True)
    file_path = sub_dir / f"{sha256}{extension}"
    return file_path
