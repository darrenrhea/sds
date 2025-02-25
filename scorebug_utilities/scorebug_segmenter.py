from write_rgba_hwc_np_u8_to_png import (
     write_rgba_hwc_np_u8_to_png
)
import shutil
from get_video_frame_path_from_clip_id_and_frame_index import (
     get_video_frame_path_from_clip_id_and_frame_index
)
from make_rgba_from_original_and_mask_paths import (
     make_rgba_from_original_and_mask_paths
)
import sys
from make_rgba_hwc_np_u8_from_rgb_and_alpha import (
     make_rgba_hwc_np_u8_from_rgb_and_alpha
)
from prii import (
     prii
)
from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

def compute_stable_pixel_mask(
    images: List[np.array],
    threshold=100
):
    """
    Computes a mask highlighting pixels where the color hardly changes across multiple images.
    
    :param image_paths: List of file paths for input images.
    :param threshold: Variance threshold below which pixels are considered stable.
    :return: The mask image.
    """
    if len(images) < 2:
        raise ValueError("At least two images are required.")

   
    # Ensure all images are of the same size
    height, width, channels = images[0].shape
    for img in images:
        if img.shape != (height, width, channels):
            raise ValueError("All images must have the same dimensions.")

    # Convert images to float32 for precision
    images_array = np.array(images, dtype=np.float32)

    # Compute per-pixel mean and variance
    mean_image = np.mean(images_array, axis=0)
    variance_image = np.var(images_array, axis=0)

    # Compute the grayscale variance by averaging across RGB channels
    variance_gray = np.max(variance_image, axis=2)


    # Create a mask where low variance means stable pixels (255), high variance means unstable pixels (0)
    mask = np.where(variance_gray < threshold, 255, 0).astype(np.uint8)

    return mask


def main():
    # Specify input folder and image format
    input_folder = "input_images"
    clip_id = "allstar-2025-02-16-sdi"
    frame_indices_in_which_scorebug_is_the_same = [
        260400,
        260500,
        260600,
        260700,
        260800,
        260900,
        261000,
        
    ]

    images = [
        get_original_frame_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index,
        )
        for frame_index in frame_indices_in_which_scorebug_is_the_same
    ]
    for image in images:
        prii(image)

 
    # Compute stable pixel mask
    mask = compute_stable_pixel_mask(images)
    mask[...] = 0
    # xmin = 1408
    # xmax = 1829
    # ymin = 858
    # ymax = 986
    mask[858:986, 1408:1829] = 255
    
    prii(mask)
    
    # for image in images:
    #     rgba = make_rgba_hwc_np_u8_from_rgb_and_alpha(
    #         rgb=image,
    #         alpha=mask
    #     )
    #     prii(rgba)

    



    for frame_index in range(262000, 600000+1, 1000):
        original_path = get_video_frame_path_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index
        )
        # we used the model that basically works except for the scorebug
        mask_path = Path(
            f"/shared/preannotations/floor_not_floor/{clip_id}/{clip_id}_{frame_index:06d}_nonfloor.png"
        )
        rgba = make_rgba_from_original_and_mask_paths(
            original_path=original_path,
            mask_path=mask_path,
            flip_mask=False,
            quantize=True
        )
        rgba[:, :, 3] = np.maximum(rgba[:, :, 3], mask)
        out_dir = Path(f"/shared/scorebug_fixed")
        out = out_dir / f"{clip_id}_{frame_index:06d}_nonfloor.png"
        shutil.copy(
            src=original_path,
            dst=out_dir / f"{clip_id}_{frame_index:06d}_original.jpg"
        )
        write_rgba_hwc_np_u8_to_png(
            rgba_hwc_np_u8=rgba,
            out_abs_file_path=out,
            verbose=True,
        )
        # prii(rgba, out=out)
  

if __name__ == "__main__":
    main()