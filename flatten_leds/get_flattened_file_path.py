from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)


def get_flattened_file_path(
    kind: str,  # "original" or "onscreen"
    clip_id: str,
    frame_index: int,
    board_id: str,  # "left" or "right"
    rip_width: int,
    rip_height: int,
):
    """
    /home/darren/r/flatten_leds/get_flattened_file_path.py
    Given the kind (original or onscreen), clip_id, frame_index, board_id, rip_width, rip_height,
    returns the file path of the flattened LED board image.

    Ideally, This might be used to:
    1. stage the flattened images on the hard drive in the first place
    2. read it from the hard drive later.
    """
    assert kind in ["original", "onscreen"], "kind must be 'original' or 'onscreen'"
    assert isinstance(frame_index, int), "frame_index must be an int"
    assert board_id in ["left", "right"], "board_id must be 'left' or 'right'"
    assert isinstance(rip_width, int), "rip_width must be an int"
    assert isinstance(rip_height, int), "rip_height must be an int"

    shared_dir = get_the_large_capacity_shared_directory()
    out_dir = shared_dir / "clips" / clip_id / "flat" / board_id / f"{rip_width}x{rip_height}" / "frames"
    original_path = out_dir / f"{clip_id}_{board_id}_{frame_index:06d}_{kind}.png"
    
    return original_path
