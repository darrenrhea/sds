import textwrap
from print_red import (
     print_red
)
from is_sha256 import (
     is_sha256
)
from print_yellow import (
     print_yellow
)
import sys
from pathlib import Path
from typing import Optional
from hash_tools import sha256_of_file
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)


def gfposaiwad_get_file_path_of_sha256_assuming_it_was_already_downloaded(
    sha256: str,
    check: bool,
) -> Optional[Path]:
    """
    Please do not call this unless you are sure the file has already been downloaded.

    As long as the file has already been cached locally
    and you know the file's sha256 you can get a local path to it.

    This will only look for the file in the local computer's sha256 cache directory.
    If it finds it, wonderful,
    returns an absolute file Path to it on the local machine, possibly checking the sha256 hash.

    If it cannot find it locally, it will return None as a sign of failure.
    """
    the_sha256_hash = sha256
    assert is_sha256(the_sha256_hash), "ERROR: {sha256} not a valid sha256 hash!"
    
    shared_dir = get_the_large_capacity_shared_directory()
    sha256_local_cache_dir = shared_dir / "sha256"
    
    d01 = sha256[0:2]
    d23 = sha256[2:4]
    d45 = sha256[4:6]
    d67 = sha256[6:8]

    sub_dir = sha256_local_cache_dir / d01 / d23 / d45 / d67

    local_path = None
    if not sub_dir.is_dir():
        print_red(f"Note: The subdirectory:\n{sub_dir}\ndoes not even exist, so we definitely do not have the file locally!")
        return None

    candidates = []
    for p in sub_dir.iterdir():
        if not p.is_file():
            continue
        if p.stem.startswith(the_sha256_hash):
            candidates.append(p)
            local_path = p
    
    if len(candidates) == 0:
        print_red(f"Bizarrely, the subdirectory:\n{sub_dir}\ndoes exist, yet not file with sha256 hash {the_sha256_hash} is found in it!")
        return None
    elif len(candidates) == 1:
        local_path = candidates[0]
    elif len(candidates) > 1:
        print_yellow("WARNING: There are more than one file with the same sha256 hash, using the longest one:")
        for c in candidates:
            print_yellow(c)
        candidates.sort(
            key=lambda p: len(p.name),
            reverse=True
        )
        local_path = candidates[0]
    
    assert local_path is not None
    assert local_path.is_absolute()
    assert local_path.exists()
    assert local_path.is_file()
    if check:
        recalculated_sha256 = sha256_of_file(local_path)
        if recalculated_sha256[:len(the_sha256_hash)] != the_sha256_hash:
            for index, c in enumerate(the_sha256_hash):
                assert recalculated_sha256[index] == the_sha256_hash[index]
            print_red(
                textwrap.dedent(
                    f"""\
                    ERROR: The file:
                    
                    {local_path}
                    
                    f"exists locally, but it does not have correct sha256 that it claims:
                    {the_sha256_hash}
                    its shas256 actually comes out to be
                    {recalculated_sha256}
                    """
                )
            )
            sys.exit(1)
    
    return local_path



