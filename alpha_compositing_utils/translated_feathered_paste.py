import numpy as np
from typing import Tuple

from feathered_paste_for_images_of_the_same_size import (
     feathered_paste_for_images_of_the_same_size
)

from translated_paste_onto_blank_canvas import (
     translated_paste_onto_blank_canvas
)



def translated_feathered_paste(
    bottom_layer_color_np_uint8: np.ndarray,  # image we insert the basketball, player-cutout, whatever ontop of
    top_layer_rgba_np_uint8: np.ndarray,
    top_xy: Tuple[int, int],  # a point on the cutout
    bottom_xy: Tuple[int, int]  # a point on the bottom layer
):
    """
    This will paste the top_layer_rgba_np_uint8 onto the bottom_layer_color_np_uint8
    such that 
    This is supposed to be the basic alpha-channel controlled overlay
    of an alpha-channeled cutout ontop of something else which is opaque color.
    """

    i0 = bottom_xy[1] - top_xy[1]
    j0 = bottom_xy[0] - top_xy[0]

    desired_height = bottom_layer_color_np_uint8.shape[0]
    desired_width = bottom_layer_color_np_uint8.shape[1]

    full_size_top_layer_rgba_np_uint8 = translated_paste_onto_blank_canvas(
        desired_height=desired_height,
        desired_width=desired_width,
        top_layer_rgba_np_uint8=top_layer_rgba_np_uint8,
        i0=i0,
        j0=j0
    )
  
    composition_np_uint8 = feathered_paste_for_images_of_the_same_size(
        bottom_layer_color_np_uint8=bottom_layer_color_np_uint8,
        top_layer_rgba_np_uint8=full_size_top_layer_rgba_np_uint8
    )

    translated_mask = full_size_top_layer_rgba_np_uint8[:, :, 3]

    
    return composition_np_uint8, translated_mask
