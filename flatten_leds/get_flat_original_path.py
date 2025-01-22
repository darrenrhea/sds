from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)


def get_flat_original_path(
    clip_id: str,
    frame_index: int,
    board_id: str,
    rip_width: int,
    rip_height: int,
):
    """
    Given a clip_id, frame_index, board_id, rip_width, and rip_height,
    returns the file path of the flattened LED board image.
    """
    shared_dir = get_the_large_capacity_shared_directory()
    out_dir = shared_dir / "clips" / clip_id / "flat" / board_id / f"{rip_width}x{rip_height}" / "frames"
    original_path = out_dir / f"{clip_id}_{frame_index:06d}_original.png"
    
    return original_path
