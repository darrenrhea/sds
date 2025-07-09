from upload_file_paths_to_s3_file_uris import (
     upload_file_paths_to_s3_file_uris
)
import time
from pathlib import Path


def test_upload_file_paths_to_s3_file_uris_1():
# Example usage:

    first_frame_index = 0
    last_frame_index = 1000
    step = 1

    
    src_file_path_dst_s3_file_uri_pairs = []

    for frame_index in range(first_frame_index, last_frame_index + 1, step):
        basename = f"nfl-59778-skycam_{frame_index:06d}_original.jpg"

        s3_file_uri = f"s3://awecomai-video-frames/clips/nfl-59778-skycam/frames/{basename}"

        local_file_path = Path(
            f"/hd2/clips/nfl-59778-skycam/frames/{basename}"
        )

        p = (
            local_file_path,
            s3_file_uri
        )
         
        src_file_path_dst_s3_file_uri_pairs.append(p)

    num_files = len(src_file_path_dst_s3_file_uri_pairs)

    start = time.time()
    upload_file_paths_to_s3_file_uris(
       src_file_path_dst_s3_file_uri_pairs=\
       src_file_path_dst_s3_file_uri_pairs,
       
       max_workers=100
    )
    stop = time.time()
    print(f"Elapsed time: {stop - start} seconds to upload {num_files} files.")


if __name__ == "__main__":
    test_upload_file_paths_to_s3_file_uris_1()
    print("Test completed successfully.")