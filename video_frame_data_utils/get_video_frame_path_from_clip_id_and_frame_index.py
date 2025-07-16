import sys
from what_computer_is_this import (
     what_computer_is_this
)
import textwrap
from download_file_via_rsync import (
     download_file_via_rsync
)
from pathlib import Path
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from prii import (
     prii
)
from colorama import Fore, Style


def get_video_frame_path_from_clip_id_and_frame_index(
    clip_id: str,
    frame_index: int,
    force_redownload: bool = False
) -> Path:
    """
    Suppose we have a database that says which computer(s) the video frames are saved on.
    Given the clip_id and frame_index as the video frame primary key,
    you may want to get a local copy of that video frame.

    TODO: might want to factor this through sha256s so people can fetch from s3.
    Later when there is postgres.
    """
    assert (
        isinstance(clip_id, str)
    ), textwrap.dedent(
        f"""\
        get_video_frame_path_from_clip_id_and_frame_index
        expected clip_id to be a string, but instead got
        {clip_id}
        which has type {type(clip_id)}
        """
    )
    assert (
        isinstance(frame_index, int)
    ), textwrap.dedent(
        f"""\
        get_video_frame_path_from_clip_id_and_frame_index
        expected frame_index to be an int, but instead got
        {frame_index}
        which has type {type(frame_index)}
        """
    )

    clip_id_to_machine_and_dir = {
        "hou-nyk-2024-11-04-sdi": ("lam", "/hd2"),
        "dal-min-2023-12-14-mxf": ("squanchy", "/shared"),
        "bos-mia-2024-04-21-mxf": ("lam", "/shared"),
        "bos-ind-2024-01-30-mxf": ("squanchy", "/shared"),
        "cle-mem-2024-02-02-mxf": ("squanchy", "/shared"),
        "dal-lac-2024-05-03-mxf": ("squanchy", "/shared"),
        "dal-bos-2024-06-06-srt": ("lam", "/shared"),
        "dal-bos-2024-06-12-mxf": ("lam", "/shared"),
        "dal-bos-2024-01-22-mxf": ("lam", "/shared"),
        "bos-dal-2024-06-06-srt": ("lam", "/shared"),
        "bos-dal-2024-06-09-mxf": ("lam", "/shared"),
        "dal-bos-2024-06-11-calibration": ("lam", "/shared"),
        "slgame1": ("lam", "/shared"),
        "slday2game1": ("lam", "/shared"),
        "slday3game1": ("lam", "/shared"),
        "slday4game1": ("lam", "/shared"),
        "slday5game1": ("lam", "/shared"),
        "slday6game1": ("lam", "/shared"),
        "slday7game1": ("lam", "/shared"),  # it was not staged?
        "slday8game1": ("lam", "/shared"),
        "slday9game1": ("lam", "/shared"),
        "slday10game1": ("lam", "/shared"),
        "chicago1080p": ("lam", "/shared"),
        "brewcub": ("lam", "/shared"),
        "chicago4k_inning1": ("lam", "/shared"),
        "hou-sas-2024-10-17-sdi": ("lam", "/shared"),
        "hou-was-2024-11-11-sdi": ("lam", "/shared"),
        "hou-gsw-2024-11-02-sdi": ("lam", "/shared"),
        "was-uta-2022-11-12-ingest": ("lam", "/hd2"),
        "chi-den-2022-11-13-ingest": ("lam", "/hd2"),
        "atl-bos-2022-11-16-ingest": ("lam", "/hd2"),
        "allstar-2025-02-16-sdi": ("lam", "/hd2"),
        "bal2024_senegal": ("lam", "/hd2"),
        "rabat": ("lam", "/hd2"),
        "nfl-59778-skycam": ("lam", "/hd2"),
        "okstfull": ("lam", "/home/drhea/awecom/data", ".jpg"),
        "sl-2025-07-10-sdi": ("lam", "/shared"),
    }

    if clip_id not in clip_id_to_machine_and_dir:
        print(
            f"{Fore.YELLOW}WARNING:{clip_id=} not in {clip_id_to_machine_and_dir=}{Style.RESET_ALL}"
        )
        print("edit sds/video_frame_data_utils/get_video_frame_path_from_clip_id_and_frame_index.py")
        sys.exit(1)

 

    my_name = what_computer_is_this()
    
    v = clip_id_to_machine_and_dir[clip_id]
    src_machine = v[0]
    dir_str = v[1]

    if len(v) >= 3:
        suffix = v[2]
    else:
        suffix = "_original.jpg"
    original_name = f"{clip_id}_{frame_index:06d}{suffix}"


    remote_shared_dir = Path(dir_str)
    remote_original_file_path = remote_shared_dir / "clips" / clip_id / "frames" / original_name

    if my_name == src_machine:  # we are already on the machine that has the file
        return remote_original_file_path

    shared_dir = get_the_large_capacity_shared_directory()
    local_frames_dir_path = shared_dir / "clips" / clip_id / "frames"
    local_frames_dir_path.mkdir(parents=True, exist_ok=True)
    local_original_file_path = local_frames_dir_path / original_name
    
    if force_redownload or not local_original_file_path.is_file():
        download_file_via_rsync(
            src_machine=src_machine,
            src_file_path=remote_original_file_path,
            dst_file_path=local_original_file_path,
            verbose=True
        )
        
    assert (
        isinstance(local_original_file_path, Path)
    ), f"{local_original_file_path=} is not a Path"
    
    assert (
        local_original_file_path.is_file()
    ), f"{local_original_file_path=} is not an extant file"

    return local_original_file_path


def demo():
    clip_ids = [
        "ind-okc-2025-06-11-hack",
    ]
    clip_id = clip_ids[0]
       
        
    frame_indices = [
        3779,
    ]

    for frame_index in frame_indices:
        original_file_path = get_video_frame_path_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index
        )
        
        prii(original_file_path)


if __name__ == "__main__":
    demo()