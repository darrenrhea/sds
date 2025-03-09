import time
from typing import List, Tuple
from download_s3_file_uris_to_file_paths import (
     download_s3_file_uris_to_file_paths
)
from pathlib import Path
from ls_with_glob_for_s3 import (
     ls_with_glob_for_s3
)


def datfitsfitf_download_all_the_files_in_this_s3_folder_into_this_folder(
    s3_folder: str,
    dst_dir_path: Path,
    glob_pattern: str,
    max_workers: int,
) -> List[Tuple[str, Path]]:
    """
    Download all the files at the top level of the s3_folder
    which match the given glob_pattern (use "*" for all files)
    into the given local folder dst_dir_path.
    max_workers = 100 is faster than max_workers = 10.
    Return a list of pairs of the form (s3_file_uri, dst_file_path)
    explaining what was downloaded to where.
    """
    assert s3_folder.startswith("s3://"), "s3_folder should start with s3://"
    assert s3_folder.endswith("/"), "s3_folder should end with a slash"
    
    s3_file_uris = ls_with_glob_for_s3(
        s3_pseudo_folder=s3_folder,
        glob_pattern=glob_pattern,
        recursive=False  
    )


    src_s3_file_uri_dst_file_path_pairs = []

    for s3_file_uri in s3_file_uris:
        name = s3_file_uri.split("/")[-1]
        p = (
            s3_file_uri,
            dst_dir_path / name
        )
            
        src_s3_file_uri_dst_file_path_pairs.append(p)

    start = time.time()
    download_s3_file_uris_to_file_paths(
        src_s3_file_uri_dst_file_path_pairs=\
        src_s3_file_uri_dst_file_path_pairs,
        max_workers=\
        max_workers,
    )
    stop = time.time()
    num_files = len(s3_file_uris)
    print(f"Elapsed time: {stop - start} seconds to download {num_files} megabytes using {max_workers=}")
    print(f"ls {dst_dir_path}")

    return src_s3_file_uri_dst_file_path_pairs
