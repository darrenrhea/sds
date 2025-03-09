from download_s3_file_uris_to_file_paths import (
     download_s3_file_uris_to_file_paths
)
import time
from get_a_temp_dir_path import (
     get_a_temp_dir_path
)


def test_download_s3_file_uris_to_file_paths_1():
    num_files = 1000
    max_workers = 100
    
    src_s3_file_uri_dst_file_path_pairs = []
    temp_dir_path = get_a_temp_dir_path()
    
    for file_index in range(num_files):
        p = (
            f"s3://awecomai-temp/crap/file{file_index}.txt",
            temp_dir_path / f"file{file_index}.txt",
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

    print(f"Elapsed time: {stop - start} seconds to download {num_files} megabytes using {max_workers=}")
    print(f"ls {temp_dir_path}") 


if __name__ == "__main__":
    test_download_s3_file_uris_to_file_paths_1()