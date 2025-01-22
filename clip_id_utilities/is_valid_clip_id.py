from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)


def is_valid_clip_id(clip_id: str) -> bool:
    """
    Do something to validate the clip_id.
    """
    shared_dir = get_the_large_capacity_shared_directory()
    frames_dir = shared_dir / "clips" / clip_id
    if not frames_dir.is_dir():
        print(f"ERROR: {frames_dir=} does not exist")
        return False
    return True