from closest_string_in_list import (
     closest_string_in_list
)
from print_yellow import (
     print_yellow
)
from pathlib import Path

def get_mother_dir_of_frames_dir_from_clip_id(
    clip_id: str,
):
    """
    Given a clip_id, frame_index, returns the path to the original video frame.
    """
    clip_id_to_folder = {  # alphabetical order to avoid duplicates
        "allstar-2025-02-16-sdi": "/hd2",
        "bal_game2_bigzoom": "/hd2",
        "bal_rabat_20250410_aug": "/hd2",
        "bal_rabat_20250412_2": "/hd2",
        "bal_rabat_20250412_3": "/hd2",
        "bal_rabat_20250412_4": "/hd2",
        "bal_rabat_20250412": "/hd2",
        "bal2024_egypt": "/hd2",
        "bal2024_rwanda": "/hd2",
        "bal2024_senegal": "/hd2",
        "bal2024_southafrica": "/hd2",
        "fus-aia-2025-04-05": "/hd2",
        "GSWvNOP_PGM_core_tnt_10-20-2024": "/shared",
        "hou-gsw-2024-11-02-sdi": "/hd2",
        "hou-ind-2024-11-20-sdi": "/hd2",
        "hou-lac-2024-11-15-sdi": "/hd2",
        "ind-bos-2024-10-30-hack": "/mnt/nas/volume1/videos",
        "ind-lal-2023-02-02-mxf": "/mnt/nas/volume1/videos",
        "ind-tor-2022-11-11-mxf": "/mnt/nas/volume1/videos",
        "nfl-59773-skycam-ddv3": "/hd2",
        "nfl-59778-skycam": "/hd2",
        "nfl-59778-skycam2": "/media/drhea/corsair4tb",
        "okc-ind-2025-06-05-youtube": "/mnt/nas/volume1/videos",
        "okc-ind-2025-06-08-youtube": "/mnt/nas/volume1/videos",
        "okc-phi-2022-12-31-mxf": "/mnt/nas/volume1/videos",
        "rabat": "/hd2",
        "rwanda-2025-05-17-sdi8": "/media/drhea/corsair4tb3",
        "short-bal": "/hd2",
        "stadepart2": "/hd2",
        "ind-okc-2025-06-11-hack": "/shared",
        "ind-okc-2025-06-11-hack_2": "/hd2",
        "ind-okc-2025-06-11-hack_3": "/hd2"
    }
    if clip_id in clip_id_to_folder:
        mother_dir_of_frames_dir  = Path(clip_id_to_folder[clip_id])
    else:
        valid_strings = list(clip_id_to_folder.keys())
        closest_string_in_list(
            s=clip_id,
            valid_strings=valid_strings,
            crash_on_inexact=True,
        )
        # no one blows out to /shared anymore
        # mother_dir_of_frames_dir = get_the_large_capacity_shared_directory()

    print_yellow(f"For {clip_id}, using {mother_dir_of_frames_dir} as the shared directory")
    
    folder = mother_dir_of_frames_dir / "clips" / clip_id / "frames"
    assert folder.is_dir(), f"{folder} should be a directory"

    return mother_dir_of_frames_dir
