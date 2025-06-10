from print_yellow import (
     print_yellow
)
from get_mother_dir_of_frames_dir_from_clip_id import (
     get_mother_dir_of_frames_dir_from_clip_id
)
from pathlib import Path

def get_original_path(
    clip_id: str,
    frame_index: int,
):
    """
    Given a clip_id, frame_index, returns the path to the original video frame.
    """
    
    shared_dir = get_mother_dir_of_frames_dir_from_clip_id(
        clip_id
    )
    print_yellow(f"For {clip_id}, using {shared_dir} as the shared directory")
    
    folder = shared_dir / "clips" / clip_id / "frames"
    original_path = folder / f"{clip_id}_{frame_index:06d}_original.png"
    if original_path.exists():
        return original_path
    else:
        original_path2 = folder / f"{clip_id}_{frame_index:06d}_original.jpg"
        if not original_path2.exists():
            raise FileNotFoundError(f"Neither {original_path} nor {original_path2} exists")
        return original_path2
    