from upload_file_via_rsync import (
     upload_file_via_rsync
)
# from download_via_rsync import (
#      download_via_rsync
# )
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from pathlib import Path

import tempfile





def test_upload_file_via_rsync_1():
    clip_id = "munich2024-01-09-1080i-yadif"
    frame_index = 123456
    original_suffix = "_original.png"
    dst_machine = "appa"

    dst_shared_dir = get_the_large_capacity_shared_directory(
        computer_name=dst_machine
    )

    local_shared_dir = get_the_large_capacity_shared_directory()

    src_file_path = local_shared_dir / "clips" / clip_id / "frames" / f"{clip_id}_{frame_index:06d}{original_suffix}"

    dst_dir = Path("/home/anna/temp")

    dst_file_path = dst_dir / f"{clip_id}_{frame_index:06d}{original_suffix}"
    
    upload_file_via_rsync(
        dst_machine=dst_machine,
        src_file_path=src_file_path,
        dst_file_path=dst_file_path,
        verbose=True,
    ) 
    
    # download_via_rsync(
    #     src_machine=src_machine,
    #     src_file_path=src_file_path,
    #     dst_path=dst_file_path,
    #     verbose=True
    # )

    # print(dst_file_path)
    # assert dst_file_path.exists(), "File not downloaded?!"



if __name__ == "__main__":
    test_upload_file_via_rsync_1()
  


