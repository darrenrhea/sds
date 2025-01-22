from pathlib import Path
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
import numpy as np
import PIL.Image


def get_mask_rgba_from_clip_id_and_frame_index(
    clip_id,
    frame_index,
    anti_aliasing_factor=1,
) -> np.ndarray:
    """
    Currently returns rgb_hwc_np_u8.
    We have ambitions to return something of higher precision.
    This handles where we have staged originals on all computers.
    If you specify a clip_id and a frame_index,
    this function will return the rgb_hwc_u8 image at that frame.
    TODO: higher precision like float16.
    """
    
    shared_dir = get_the_large_capacity_shared_directory()
    
    original_image_path = Path(
        shared_dir / "clips" / f"{clip_id}/conventions/led/preann/{clip_id}_{frame_index:06d}_nonfloor.png"
    ).expanduser()

    assert original_image_path.exists()

    original_image_pil = PIL.Image.open(original_image_path).convert("RGBA")

    resized_pil = original_image_pil.resize(
        (
            original_image_pil.width * anti_aliasing_factor,
            original_image_pil.height * anti_aliasing_factor
        ),
        resample=PIL.Image.Resampling.BILINEAR
    )

    rgb_np_u8 = np.array(resized_pil)
    assert rgb_np_u8.dtype == np.uint8
    assert rgb_np_u8.shape[2] == 4
    return rgb_np_u8