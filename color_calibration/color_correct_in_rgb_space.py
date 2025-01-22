from correct_each_channel_independently import (
     correct_each_channel_independently
)
from typing import List




import numpy as np

import pprint as pp
import sys


def color_correct_in_rgb_space(
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

    from_to_mapping_array = rgb_from_to_mapping_array.astype(np.float64) / 255.0

    pp.pprint(rgb_from_to_mapping_array)
 
    uncorrected_rgb_f64s = []
    for uncorrected_rgb in uncorrected_rgbs:
        uncorrected_rgb_f64 = uncorrected_rgb.astype(np.float64) / 255.0
        uncorrected_rgb_f64s.append(uncorrected_rgb_f64)


    corrected_rgb_f64s = []
    for uncorrected_rgb_f64 in uncorrected_rgb_f64s:
       
        corrected_rgb_f64 = correct_each_channel_independently(
            uncorrected=uncorrected_rgb_f64,
            from_to_mapping_array=from_to_mapping_array,
            channel_to_config=channel_to_config,
            graph_it=True,
        )
        corrected_rgb_f64s.append(corrected_rgb_f64)

    corrected_rgbs = []
    for corrected_rgb_f64 in corrected_rgb_f64s:
        corrected_rgb_u8 = np.round(corrected_rgb_f64 * 255.0).clip(2, 255).astype(np.uint8)
        corrected_rgbs.append(corrected_rgb_u8)
    
    return corrected_rgbs
