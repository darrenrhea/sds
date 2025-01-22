from pathlib import Path
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
import better_json as bj


# this is memoization of loading the large jsonlines files:
clip_id_to_camera_poses_and_index_offset = dict()


def get_screen_corners_from_clip_id_and_frame_index(
    clip_id,
    frame_index
):
    """
    TODO: make this pull from a big jsonlines file that is known by sha256,
    so that we don't have to stage things.

    If you specify a clip_id and a frame_index, this function will return
    as dictionary from screen_name to a dictionary from corner name to a 2D (x, y) point
    in pixel coordinates.  The corner_names are tl, bl, br, tr (the U convention).
    and the screen_names are the names of the screens in the clip, usually
    "left" and "right".
    """
    
    shared_dir = get_the_large_capacity_shared_directory()

    screen_corners_file_path = Path(
        shared_dir / "clips" / clip_id / "screen_corners" / f"{clip_id}_{frame_index:06d}.json"
    )
    
    screen_name_to_corner_name_to_xy = bj.load(
        screen_corners_file_path
    )

    return screen_name_to_corner_name_to_xy