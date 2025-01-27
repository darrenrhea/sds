from get_video_frame_path_from_clip_id_and_frame_index import (
     get_video_frame_path_from_clip_id_and_frame_index
)

import numpy as np
import PIL.Image


def get_original_frame_from_clip_id_and_frame_index(
    clip_id: str,
    frame_index: int,
) -> np.ndarray:
    """
    If you specify a string clip_id and an int frame_index,
    this function will return the rgb_hwc_np_u8 image at that frame,
    regardless of whether or not it is locally staged.

    Currently this returns rgb_hwc_np_u8.
    We have ambitions to return something of higher precision.
    
    This also handles, per each computer's idiosyncracies,
    where we choose to stage the "blown-out" original video frames on that computer.

   
    TODO: maybe return higher precision like float32 instead.
    """
    
    original_image_path = get_video_frame_path_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index
    )

    try:
        original_pil = PIL.Image.open(original_image_path).convert("RGB")
    except OSError as e:
        print(f"ERROR: could not open {original_image_path} as an image.")
        raise e
    

    rgb_hwc_np_u8 = np.array(original_pil)
    assert rgb_hwc_np_u8.dtype == np.uint8
    assert rgb_hwc_np_u8.shape[0] == 1080
    assert rgb_hwc_np_u8.shape[1] == 1920
    assert rgb_hwc_np_u8.shape[2] == 3
    return rgb_hwc_np_u8