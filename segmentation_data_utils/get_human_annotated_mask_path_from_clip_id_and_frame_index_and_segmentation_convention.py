from print_red import (
     print_red
)
from find_file_path_with_this_name import (
     find_file_path_with_this_name
)
from pathlib import Path
from typing import Optional


def get_human_annotated_mask_path_from_clip_id_and_frame_index_and_segmentation_convention(
    clip_id: str,
    frame_index: int,
    segmentation_convention: str,
) -> Optional[Path]:
    
    precalculated = [
        "slgame1",
        "slday2game1",
        "slday3game1",
        "slday4game1",
        "slday5game1",
        "slday6game1",
        "slday8game1",
        "slday9game1",
        "slday10game1",
    ]
    if clip_id in precalculated and segmentation_convention == "floor_not_floor":
        mother_dir = Path(f"/shared/preannotations/{segmentation_convention}")
        folder = mother_dir / clip_id
        mask_path = folder / f"{clip_id}_{frame_index:06d}_nonfloor.png"
    else:
        assert segmentation_convention in ["floor_not_floor", "led"]
        suffix = {
            "floor_not_floor": "floor",
            "led": "led",
        }[segmentation_convention]

        repo_dir = Path(
            f"~/r/{clip_id}_{suffix}"
        ).expanduser().resolve()
        
        file_name = f"{clip_id}_{frame_index:06d}_nonfloor.png"
            
        mask_path = find_file_path_with_this_name(
            dir_path_to_search=repo_dir,
            file_name=file_name,
        )
        
    if mask_path is None:
        print_red(f"ERROR: mask_path is None for {clip_id=} {frame_index=} {segmentation_convention}")
        return None

    return mask_path
  