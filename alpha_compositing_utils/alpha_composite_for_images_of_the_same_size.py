import numpy as np


def alpha_composite_for_images_of_the_same_size(
    bottom_layer_color_np_uint8,
    top_layer_color_np_uint8,
    top_opacity_np_uint8,
):
    """
    This one assumes they are already the same size, and no translation is needed.
    See 
    """
    assert (
        bottom_layer_color_np_uint8.shape[2] == 3
    ), f"{bottom_layer_color_np_uint8.shape=} but must have 3 channels"
    
    assert (
        top_layer_color_np_uint8.shape[2] == 3
    ), f"{top_layer_color_np_uint8.shape=} but must have 3 channels"

    assert (
        bottom_layer_color_np_uint8.shape[0] == top_layer_color_np_uint8.shape[0]
    ), f"{bottom_layer_color_np_uint8.shape=}, yet {top_layer_color_np_uint8.shape=}"
    
    assert (
        bottom_layer_color_np_uint8.shape[1] == top_layer_color_np_uint8.shape[1]
    ), f"{bottom_layer_color_np_uint8.shape=}, yet {top_layer_color_np_uint8.shape=}"
    
 
    opacity_float = top_opacity_np_uint8[:, :, np.newaxis].astype(np.float32) / 255.0

    composition_np_uint8 = np.clip(
        np.round(
            (1.0 - opacity_float) * bottom_layer_color_np_uint8.astype(np.float32)
            +
            opacity_float * top_layer_color_np_uint8.astype(np.float32)
        ),
        0,
        255
    ).astype(np.uint8)
    return composition_np_uint8

