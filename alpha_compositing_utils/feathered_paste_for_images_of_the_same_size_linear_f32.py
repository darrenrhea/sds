import numpy as np


def feathered_paste_for_images_of_the_same_size_linear_f32(
    bottom_layer_rgb_np_linear_f32,
    top_layer_rgba_np_linear_f32
):
    """
    It is easier and more appropriate to compose in a linear space.
    The rgb channels are in a linear space where addition makes sense to do.
    This one assumes they are already the same size, and no translation is needed.
    See 
    """
    assert bottom_layer_rgb_np_linear_f32.dtype == np.float32
    assert top_layer_rgba_np_linear_f32.dtype == np.float32
    assert bottom_layer_rgb_np_linear_f32.ndim == 3
    assert top_layer_rgba_np_linear_f32.ndim == 3
    assert bottom_layer_rgb_np_linear_f32.shape[2] == 3
    assert top_layer_rgba_np_linear_f32.shape[2] == 4

    assert np.min(bottom_layer_rgb_np_linear_f32) >= -0.01
    assert np.max(bottom_layer_rgb_np_linear_f32) <= 1.01
    
    assert np.min(top_layer_rgba_np_linear_f32) >= -0.01
    assert np.max(top_layer_rgba_np_linear_f32) <= 1.01

    assert bottom_layer_rgb_np_linear_f32.shape[2] == 3
    assert (
        bottom_layer_rgb_np_linear_f32.shape[0] == top_layer_rgba_np_linear_f32.shape[0]
        and
        bottom_layer_rgb_np_linear_f32.shape[1] == top_layer_rgba_np_linear_f32.shape[1]
    ), f"{bottom_layer_rgb_np_linear_f32.shape=}, yet {top_layer_rgba_np_linear_f32.shape=}"
    

    # just the alpha channel of the ad, playercutout, whatever
    top_opacity_np_linear_f32 = top_layer_rgba_np_linear_f32[:, :, 3]

    # just the rgb channels of the ad, playercutout, whatever
    top_layer_color_np_linear_f32 = top_layer_rgba_np_linear_f32[:, :, :3]

    opacity_float = top_opacity_np_linear_f32[:, :, np.newaxis]

    composition_np_linear_f32 = (
        (1.0 - opacity_float) * bottom_layer_rgb_np_linear_f32
        +
        opacity_float * top_layer_color_np_linear_f32
    )
    return composition_np_linear_f32

