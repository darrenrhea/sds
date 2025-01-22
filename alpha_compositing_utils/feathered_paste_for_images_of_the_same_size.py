import numpy as np


def feathered_paste_for_images_of_the_same_size(
    bottom_layer_color_np_uint8,
    top_layer_rgba_np_uint8
):
    """
    This one assumes they are already the same size, and no translation is needed.
    See 
    """
    assert bottom_layer_color_np_uint8.shape[2] == 3
    assert (
        bottom_layer_color_np_uint8.shape[0] == top_layer_rgba_np_uint8.shape[0]
    ), f"{bottom_layer_color_np_uint8.shape=}, yet {top_layer_rgba_np_uint8.shape=}"
    assert bottom_layer_color_np_uint8.shape[1] == top_layer_rgba_np_uint8.shape[1]

    # just the color channel of the ad, playercutout, whatever
    top_opacity_np_uint8 = top_layer_rgba_np_uint8[:, :, 3]

    # just the alpha channel of the ad, playercutout, whatever
    top_layer_color_np_uint8 = top_layer_rgba_np_uint8[:, :, :3]

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

