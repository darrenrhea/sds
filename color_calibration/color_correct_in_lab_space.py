from correct_each_channel_independently import (
     correct_each_channel_independently
)
from typing import List




import numpy as np

from skimage.color import rgb2lab, lab2rgb


def color_correct_in_lab_space(
    rgb_from_to_mapping_array: np.ndarray,  # shape=(n_colors, 2, 3)
    uncorrected_rgbs: List[np.ndarray],  # images to color correct
):
    assert rgb_from_to_mapping_array.shape[1] == 2
    assert rgb_from_to_mapping_array.shape[2] == 3

    channel_to_config = {
        0: {
            "min": 0,
            "max": 100,
            "degree": 1,
        },
        1: {
            "min": -128,
            "max": 127,
            "degree": 3,
        },
        2: {
            "min": -128,
            "max": 127,
            "degree": 3,
        }
    }

    lab_from_to_mapping_array = np.zeros_like(rgb_from_to_mapping_array)
    for i in range(rgb_from_to_mapping_array.shape[0]):
        lab_from_to_mapping_array[i, 0, :] = rgb2lab(
            rgb_from_to_mapping_array[i, 0, :].reshape(1, 1, 3) / 255.0
        )[0, 0]
        lab_from_to_mapping_array[i, 1, :] = rgb2lab(
            rgb_from_to_mapping_array[i, 1, :].reshape(1, 1, 3) / 255.0
        )[0, 0]
    
   
    uncorrected_rgb_f64s = []
    for uncorrected_rgb in uncorrected_rgbs:
        uncorrected_rgb_f64 = uncorrected_rgb.astype(np.float64) / 255.0
        uncorrected_rgb_f64s.append(uncorrected_rgb_f64)


    uncorrected_lab_f64s = []
    for uncorrected_rgb_f64 in uncorrected_rgb_f64s:
        uncorrected_lab_f64 = rgb2lab(uncorrected_rgb_f64)
        uncorrected_lab_f64s.append(uncorrected_lab_f64)

    corrected_lab_f64s = []
    for uncorrected_lab_f64 in uncorrected_lab_f64s:
       
        corrected_lab_f64 = correct_each_channel_independently(
            uncorrected=uncorrected_lab_f64,
            from_to_mapping_array=lab_from_to_mapping_array,
            channel_to_config=channel_to_config
        )
        corrected_lab_f64s.append(corrected_lab_f64)

    corrected_rgbs = []
    for corrected_lab_f64 in corrected_lab_f64s:
        corrected_rgb_f64 = lab2rgb(corrected_lab_f64)
        corrected_rgb_u8 = np.round(corrected_rgb_f64 * 255.0).clip(0, 255).astype(np.uint8)
        corrected_rgbs.append(corrected_rgb_u8)
    
    return corrected_rgbs
