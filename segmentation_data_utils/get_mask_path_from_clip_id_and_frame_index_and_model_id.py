from print_red import (
     print_red
)
from collections import OrderedDict
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


def get_mask_path_from_clip_id_and_frame_index_and_model_id(
    clip_id: str,
    frame_index: int,
    model_id: str,
    force_redownload: bool = False
) -> Path:
    """
    You may want to get a local copy
    of a video frame from the clip_id and frame_index.
    TODO: might want to factor this through sha256s so people can fetch from s3.
    Later when there is postgres.
    """
    assert (
        isinstance(clip_id, str)
    ), textwrap.dedent(
        f"""\
        get_mask_path_from_clip_id_and_frame_index_and_model_id
        expected clip_id to be a string, but instead got
        {clip_id}
        which has type {type(clip_id)}
        """
    )
    assert (
        isinstance(frame_index, int)
    ), textwrap.dedent(
        f"""\
        get_mask_path_from_clip_id_and_frame_index_and_model_id
        expected frame_index to be an int, but instead got
        {frame_index}
        which has type {type(frame_index)}
        """
    )

    assert (
        isinstance(model_id, str)
    ), textwrap.dedent(
        f"""\
        get_mask_path_from_clip_id_and_frame_index_and_model_id
        expected clip_id to be a string, but instead got
        {model_id}
        which has type {type(model_id)}
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
        "nfl-59778-skycam": "lam",
    }

    shared_dir = get_the_large_capacity_shared_directory()
    original_name = f"{clip_id}_{frame_index:06d}_{model_id}.png"
    local_frames_dir_path = shared_dir / "inferences"
    local_frames_dir_path.mkdir(parents=True, exist_ok=True)
    local_original_file_path = local_frames_dir_path / original_name
    
    if clip_id not in clip_id_to_machine:
        print_red(f"The clip_id {clip_id=} is not in the known clip_id_to_machine mapping.")
        assert (
            local_original_file_path.is_file()
        ), f"ERROR: {local_original_file_path=} is not a file, and we don't know where to get it from"
        print(
            f"{Fore.YELLOW}WARNING:{clip_id=} not in {clip_id_to_machine=}{Style.RESET_ALL}"
        )
    else:
        src_machine = clip_id_to_machine[clip_id]

        remote_shared_dir = get_the_large_capacity_shared_directory(src_machine)
        
        remote_original_file_path = remote_shared_dir / "inferences" / original_name

        

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


if __name__ == "__main__":
    clip_id = "dal-bos-2024-06-12-mxf"
    
    model_id = "dallasfixups28"
    
    ad_id_to_frames_in_which_it_appears = OrderedDict([
        ("DAL_NBA_Finals_Courtside_2560x96_v01", [357000, ]),
        ("WNBA_NYL_LVA_SAT_3_ABC_CS_DAL", [383000, 384000, 385000, 387000, 391000, 392000, ]),
        ("ABC_WNBA_NYL_LVA_CS_DAL", [723000, 741000, 749000, 750000, ]),
        ("Hotels_dot_com_CS_DAL", [395000, 403000, ]),
        ("YTTV_CS_DAL", [432000, 433000, 436000, 443000, 445000]),
        ("one_for_all_dallas_CS_DAL", [466000, 470000, 471000, ]),
        ("Summer_League_Awareness_CS_DAL", [485000, 502000, 507000, ]),
        ("BB4_Sony_CS_DAL", [511000, 512000, 515000, 516000, 523000, ]),
        ("Statefarm_CS_DAL", [543000, 544000, 549000]),
        ("ABC_Finals_Game4_Fri_CS_DAL", [618000, 621000]),
        ("ABC_NHL_Stanley_Cup_Game4_CS_DAL", [649000, 650000, 654000]),
        ("NBA_Store_Finals_CS_DAL", [664000, 668000, 689000]),
        ("NBA_Draft_Awareness_CS_DAL", [690000,692000, 698000, 699000]),
        ("Tissot_CS_DAL", [772000, 774000, 776000]),
        ("Finals_Friday_830_ABC_CS_DAL", [846000, 852000, 855000]),
    ])

    frame_indices = []
    for some_frame_indices in ad_id_to_frames_in_which_it_appears.values():
        frame_indices.extend(some_frame_indices)
    
    frame_indices = sorted(list(set(frame_indices)))
    print(f"{frame_indices=}")
    
    for frame_index in frame_indices:
        original_file_path = get_mask_path_from_clip_id_and_frame_index_and_model_id(
            clip_id=clip_id,
            frame_index=frame_index,
            model_id=model_id,
        )
        
        prii(original_file_path)