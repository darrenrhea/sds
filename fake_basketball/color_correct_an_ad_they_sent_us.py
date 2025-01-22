from color_correct_rgb_via_polynomial_coefficients_linear_f32_in_and_out import (
     color_correct_rgb_via_polynomial_coefficients_linear_f32_in_and_out
)
import numpy as np




def color_correct_an_ad_they_sent_us_as_rgb_hwc_np_u8_to_rgba_hwc_np_linear_f32(
    degree: int,
    coefficients: np.ndarray,
    rgb_hwc_np_linear_f32: np.ndarray,  # the image to color correct
):
    """
    The color of ads they sent us is corrected.
    """
   
    color_corrected_rgb_np_linear_f32 = color_correct_rgb_via_polynomial_coefficients_linear_f32_in_and_out(
        degree=degree,
        coefficients=coefficients,
        rgb_hwc_np_linear_f32=rgb_hwc_np_linear_f32
    )
    
    color_corrected_rgba_np_linear_f32 = add_an_opaque_alpha_channel_to_create_rgba_hwc_np_f32(
        color_corrected_rgb_np_linear_f32
    )
    # we have to add an alpha channel to the texture even though it's not transparent:
    color_corrected_rgba_np_linear_f32 = np.zeros(
        shape=(
            rgb_hwc_np_linear_f32.shape[0],
            rgb_hwc_np_linear_f32.shape[1],
            4
        ),
        dtype=np.float32
    )

    color_corrected_rgba_np_linear_f32[:, :, :3] = color_corrected_rgb_np_linear_f32
    color_corrected_rgba_np_linear_f32[:, :, 3] = 1.0
    

    return color_corrected_rgba_np_linear_f32

    
