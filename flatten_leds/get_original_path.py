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
    shared_dir = get_the_large_capacity_shared_directory()
    clip_id_to_folder = {
        "hou-ind-2024-11-20-sdi": "/hd2",
        "hou-lac-2024-11-15-sdi": "/hd2",
        "allstar-2025-02-16-sdi": "/hd2",
    }
    print(f"{clip_id=}")
    if clip_id in clip_id_to_folder:
        shared_dir  = Path(clip_id_to_folder[clip_id])
    else:
        shared_dir = get_the_large_capacity_shared_directory()

    
    folder = shared_dir / "clips" / clip_id / "frames"
    original_path = folder / f"{clip_id}_{frame_index:06d}_original.png"
    if original_path.exists():
        return original_path
    else:
        original_path2 = folder / f"{clip_id}_{frame_index:06d}_original.jpg"
        if not original_path2.exists():
            raise FileNotFoundError(f"Neither {original_path} nor {original_path2} exists")
        return original_path2
    