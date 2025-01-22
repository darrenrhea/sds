from remote_file_exists import (
     remote_file_exists
)
from get_flattened_file_path import (
     get_flattened_file_path
)
from download_file_via_rsync import (
     download_file_via_rsync
)
from prii import (
     prii
)
import textwrap
from pathlib import Path
from colorama import Fore, Style


def get_flattened_local_file_path(
    kind: str,  # "original" or "onscreen"
    clip_id: str,
    frame_index: int,
    board_id: str,
    rip_width: int,
    rip_height: int,
    force_redownload: bool = False
) -> Path:
    """
    This will download the asset file for you if it is not staged locally.

    You may want to get a locally stored copy
    of one of the flattened original frames or onscreen frames.

    TODO: might want to factor this through sha256s so people can fetch from s3.
    Later when there is postgres.
    """
    assert (
        kind in ["original", "onscreen"]
    ), textwrap.dedent(
        f"""\
        get_flattened_local_file_path
        expected kind to be 'original' or 'onscreen', but instead got
        {kind}
        which has type {type(kind)}
        """
    )

    assert (
        isinstance(clip_id, str)
    ), textwrap.dedent(
        f"""\
        get_flattened_local_file_path
        expected clip_id to be a string, but instead got
        {clip_id}
        which has type {type(clip_id)}
        """
    )

    assert (
        isinstance(frame_index, int)
    ), textwrap.dedent(
        f"""\
        get_flattened_local_file_path
        expected frame_index to be an int, but instead got
        {frame_index}
        which has type {type(frame_index)}
        """
    )

    clip_id_to_machine = {
        "dal-min-2023-12-14-mxf": "squanchy",
        "bos-mia-2024-04-21-mxf": "lam",
        "bos-ind-2024-01-30-mxf": "squanchy",
        "cle-mem-2024-02-02-mxf": "squanchy",
        "dal-lac-2024-05-03-mxf": "squanchy",
        "dal-bos-2024-06-06-srt": "lam",
        "dal-bos-2024-06-12-mxf": "lam",
        "dal-bos-2024-01-22-mxf": "lam",
        "bos-dal-2024-06-06-srt": "lam",
        "bos-dal-2024-06-09-mxf": "lam",
        "dal-bos-2024-06-11-calibration": "lam",
        "slgame1": "lam",
        "slday8game1": "lam",
        "chicago1080p": "lam",
        "brewcub": "lam",
    }
    
    local_file_path = get_flattened_file_path(
        kind=kind,
        clip_id=clip_id,
        frame_index=frame_index,
        board_id=board_id,
        rip_width=rip_width,
        rip_height=rip_height,
        computer_name=None
    )

    local_frames_dir_path = local_file_path.parent
    local_frames_dir_path.mkdir(parents=True, exist_ok=True)

    if clip_id not in clip_id_to_machine:
        assert (
            local_file_path.is_file()
        ), f"ERROR: {local_file_path=} is not a file, and we don't know where to get it from"
        print(
            f"{Fore.YELLOW}WARNING:{clip_id=} not in {clip_id_to_machine=}{Style.RESET_ALL}"
        )
    else:
        src_machine = clip_id_to_machine[clip_id]


        remote_file_path = get_flattened_file_path(
            kind=kind,
            clip_id=clip_id,
            frame_index=frame_index,
            board_id=board_id,
            rip_width=rip_width,
            rip_height=rip_height,
            computer_name=src_machine
        )

        assert (
            remote_file_exists(
                remote_abs_file_path_str=str(remote_file_path),
                sshable_abbrev=src_machine
            )
        ), f"ERROR: file {remote_file_path=} does not exist on {src_machine=}"

        # print(f"Downloading {src_machine}:{remote_file_path=}")
        if force_redownload or not local_file_path.is_file():
            download_file_via_rsync(
                src_machine=src_machine,
                src_file_path=remote_file_path,
                dst_file_path=local_file_path,
                verbose=True
            )
        
    assert (
        isinstance(local_file_path, Path)
    ), f"{local_file_path=} is not a Path"
    
    assert (
        local_file_path.is_file()
    ), f"{local_file_path=} is not an extant file"

    return local_file_path

       