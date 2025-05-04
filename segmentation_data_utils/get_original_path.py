from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from pathlib import Path

def get_original_path(
    clip_id: str,
    frame_index: int,
):
    """
    Given a clip_id, frame_index, returns the path to the original video frame.
    """
    clip_id_to_folder = {
        "allstar-2025-02-16-sdi": "/hd2",
        "bal2024_egypt": "/hd2",
        "hou-ind-2024-11-20-sdi": "/hd2",
        "hou-lac-2024-11-15-sdi": "/hd2",
        "nfl-59773-skycam-ddv3": "/hd2",
        "allstar-2025-02-16-sdi": "/hd2",
        "nfl-59778-skycam": "/hd2",
        "bal2024_egypt": "/hd2",
        "bal2024_senegal": "/hd2",
        "bal2024_southafrica": "/hd2",
        "bal2024_rwanda": "/hd2",
        "short-bal": "/hd2",
        "stadepart2": "/hd2",
        "fus-aia-2025-04-05": "/hd2",
        "rabat": "/hd2",
        "bal_rabat_20250410_aug": "/hd2",
        "bal_game2_bigzoom": "/hd2",
        "bal_rabat_20250412": "/hd2",
        "bal_rabat_20250412_2": "/hd2",
        "bal_rabat_20250412_3": "/hd2",
        "bal_rabat_20250412_4": "/hd2",
        "hou-gsw-2024-11-02-sdi": "/hd2",
        "GSWvNOP_PGM_core_tnt_10-20-2024": "/shared"
    }
    if clip_id in clip_id_to_folder:
        shared_dir  = Path(clip_id_to_folder[clip_id])
    else:
        shared_dir = Path("/shared")
        # no one blows out to /shared anymore
        # shared_dir = get_the_large_capacity_shared_directory()


    
    folder = shared_dir / "clips" / clip_id / "frames"
    original_path = folder / f"{clip_id}_{frame_index:06d}_original.png"
    if original_path.exists():
        return original_path
    else:
        original_path2 = folder / f"{clip_id}_{frame_index:06d}_original.jpg"
        if not original_path2.exists():
            raise FileNotFoundError(f"Neither {original_path} nor {original_path2} exists")
        return original_path2
    