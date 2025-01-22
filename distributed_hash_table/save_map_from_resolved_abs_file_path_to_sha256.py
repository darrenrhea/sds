from sha256_of_file import (
     sha256_of_file
)
from typing import List
import time
from pathlib import Path
import better_json as bj


def save_map_from_resolved_abs_file_path_to_sha256(
    file_paths: List[Path],
    out_json_path: Path
):
    """
    You can save a lot of time by thinking of two things:
    storing all the files in the cloud is a time-expensive operation,
    do it once and grossly.

    2: do this to be able to convert realpath to sha256 in a fast way.
    WARNING:
    It is almost not work indexing/memoizing though, the sha256 can be calculated
    from a file so quickly you might as well just calculate it on the fly.
    """
    start_time = time.time()
    file_path_to_sha256 = {}
    num_total = len(file_paths)
    for index, p in enumerate(file_paths):
        if index % 100 == 0:
            print(f"Stored the sha256 of {index}  / {num_total} files.")
        resolved = p.resolve()
        if not p.resolve().is_file():
            continue
        assert resolved.is_absolute()

        file_path_str = str(resolved)
        # sha256 = store_file_by_sha256(resolved)
        sha256 = sha256_of_file(resolved)
        file_path_to_sha256[file_path_str] = sha256
    stop_time = time.time()
    bj.dump(
        obj=file_path_to_sha256,
        fp=out_json_path
    )
    
    print(f"Elapsed time: {stop_time - start_time} seconds.")
