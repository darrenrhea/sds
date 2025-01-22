from correct_each_channel_independently import (
     correct_each_channel_independently
)
from typing import List




import numpy as np

from skimage.color import rgb2hsv, hsv2rgb


def color_correct_in_hsv_space(
    rgb_from_to_mapping_array: np.ndarray,  # shape=(n_colors, 2, 3)
    uncorrected_rgbs: List[np.ndarray],  # images to color correct
):
    assert rgb_from_to_mapping_array.shape[1] == 2
    assert rgb_from_to_mapping_array.shape[2] == 3
    channel_to_config = {
        0: {
            "min": 0.0,
            "max":  1.0,
            "degree": 1,
        },
        1: {
            "min": 0.0,
            "max":  1.0,
            "degree": 1,
        },
        2: {
            "min": 0.0,
            "max":  1.0,
            "degree": 1,
        }
    }

    hsv_from_to_mapping_array = np.zeros_like(rgb_from_to_mapping_array)
    
    for i in range(rgb_from_to_mapping_array.shape[0]):
        hsv_from_to_mapping_array[i, 0, :] = rgb2hsv(
            rgb_from_to_mapping_array[i, 0, :].reshape(1, 1, 3) / 255.0
        )
        hsv_from_to_mapping_array[i, 1, :] = rgb2hsv(
            rgb_from_to_mapping_array[i, 1, :].reshape(1, 1, 3) / 255.0
        )

    print("hsv_from_to_mapping_array:")
    print(hsv_from_to_mapping_array)
    
   
    uncorrected_rgb_f64s = []
    for uncorrected_rgb in uncorrected_rgbs:
        uncorrected_rgb_f64 = uncorrected_rgb.astype(np.float64) / 255.0
        uncorrected_rgb_f64s.append(uncorrected_rgb_f64)


    uncorrected_hsv_f64s = []
    for uncorrected_rgb_f64 in uncorrected_rgb_f64s:
        uncorrected_hsv_f64 = rgb2hsv(uncorrected_rgb_f64)
        uncorrected_hsv_f64s.append(uncorrected_hsv_f64)

    corrected_hsv_f64s = []
    for uncorrected_hsv_f64 in uncorrected_hsv_f64s:
       
        corrected_hsv_f64 = correct_each_channel_independently(
            uncorrected=uncorrected_hsv_f64,
            from_to_mapping_array=hsv_from_to_mapping_array,
            channel_to_config=channel_to_config,
            graph_it=True,
        )
        corrected_hsv_f64s.append(corrected_hsv_f64)

    corrected_rgbs = []
    for corrected_hsv_f64 in corrected_hsv_f64s:
        corrected_rgb_f64 = hsv2rgb(corrected_hsv_f64)
        corrected_rgb_u8 = np.round(corrected_rgb_f64 * 255.0).clip(2, 255).astype(np.uint8)
        corrected_rgbs.append(corrected_rgb_u8)
    
    return corrected_rgbs
