from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)


def get_visibility_mask_path(
    clip_id: str,
    frame_index: int,
    board_id: str,
    rip_width: int,
    rip_height: int,
):
    """
    Given a clip_id, frame_index, rip_width, and rip_height, return the path to the visibility mask of the flattened LED board.
    """
    shared_dir = get_the_large_capacity_shared_directory()   
    
    out_dir = shared_dir / "clips" / clip_id / "flat" / board_id / f"{rip_width}x{rip_height}" / "frames"
    visibility_mask_path = out_dir / f"{clip_id}_{frame_index:06d}_onscreen.png"
    return visibility_mask_path
