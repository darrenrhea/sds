from download_via_rsync import (
     download_via_rsync
)
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from pathlib import Path

import tempfile


def test_download_via_rsync_1():
    clip_id = "munich2024-01-09-1080i-yadif"
    frame_index = 123456
    original_suffix = "_original.png"
    src_machine = "lam"
    src_shared_dir = get_the_large_capacity_shared_directory(
        computer_name=src_machine
    )

    src_path = src_shared_dir / "clips" / clip_id / "frames" / f"{clip_id}_{frame_index:06d}{original_suffix}"

    dst_dir = Path(tempfile.gettempdir())
    dst_file_path = dst_dir / f"{clip_id}_{frame_index:06d}{original_suffix}"
    
    download_via_rsync(
        src_machine=src_machine,
        src_path=src_path,
        dst_path=dst_file_path,
        verbose=True
    )

    if dst_file_path.exists():
        dst_file_path.unlink()
    
    download_via_rsync(
        src_machine=src_machine,
        src_path=src_path,
        dst_path=dst_file_path,
        verbose=True
    )

    print(dst_file_path)
    assert dst_file_path.exists(), "File not downloaded?!"



if __name__ == "__main__":
    test_download_via_rsync_1()


