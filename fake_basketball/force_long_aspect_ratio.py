from prii import (
     prii
)
import numpy as np

import PIL.Image

def force_long_aspect_ratio(rgb_np_u8: np.array) -> np.array:
    """
    copies the hwc 3 times horizontally, then resize to 2560 x 96
    """
    if rgb_np_u8.shape[0] == 96 and rgb_np_u8.shape[1] == 2560:
        return rgb_np_u8
    
    if  rgb_np_u8.shape[0] == 182 and rgb_np_u8.shape[1] == 5034:
        image_pil = PIL.Image.fromarray(rgb_np_u8)
        resized_pil = image_pil.resize(
            size=(2560, 96),
            resample=PIL.Image.LANCZOS
        )
        resized = np.array(resized_pil)
        return resized
   
     
    tripled = np.concatenate([rgb_np_u8, ] * 3, axis=1)
    tripled_pil = PIL.Image.fromarray(tripled)
    resized_pil = tripled_pil.resize(
        size=(2560, 96),
        resample=PIL.Image.LANCZOS
    )
    resized = np.array(resized_pil)
    return resized