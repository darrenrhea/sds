from colorama import Fore, Style
from prii_linear_f32 import (
     prii_linear_f32
)
from convert_u8_to_linear_f32 import (
     convert_u8_to_linear_f32
)
from convert_linear_f32_to_u8 import (
     convert_linear_f32_to_u8
)
from color_correct_rgb_hwc_np_u8_image_via_polynomial_coefficients_u8_in_f32_out import (
     color_correct_rgb_hwc_np_u8_image_via_polynomial_coefficients_u8_in_f32_out
)
import copy
from insert_quads_into_camera_posed_image_behind_mask import (
     insert_quads_into_camera_posed_image_behind_mask
)
from math import comb

from pathlib import Path
import numpy as np

from prii import (
     prii
)
from CameraParameters import (
     CameraParameters
)



def show_color_correction_result(
    use_linear_light: bool,
    degree: int,
    coefficients: np.ndarray,
    original_rgb_np_u8,
    ad_placement_descriptor,
    mask_hw_np_u8,
    camera_pose,
    uncorrected_texture_rgb_np_u8,
    original_out_path: Path,
    color_corrected_out_path: Path,
):
    """
    Say you already fit a polynomial to color correct.
    this shows how well it works,
    so that you can flipflop it.
    """
    assert isinstance(degree, int)
    assert degree >= 1
    assert isinstance(coefficients, np.ndarray)
    assert coefficients.shape[1] == 3
    assert isinstance(camera_pose, CameraParameters)
    
    prii(mask_hw_np_u8, caption="mask_hw_np_u8 inside show_color_correction_result:")
    num_coefficients_should_be = comb(degree + 3, 3)

    assert (
        coefficients.shape[0] == num_coefficients_should_be
    ), f"ERROR: {coefficients.shape=}, {num_coefficients_should_be=}"

    prii(uncorrected_texture_rgb_np_u8, caption="uncorrected_texture_rgb_np_u8:")
    color_corrected_texture_rgb_f32 = color_correct_rgb_hwc_np_u8_image_via_polynomial_coefficients_u8_in_f32_out(
        use_linear_light=use_linear_light,
        degree=degree,
        coefficients=coefficients,
        rgb_hwc_np_u8=uncorrected_texture_rgb_np_u8,
    )

        
    if use_linear_light:
        color_corrected_texture_rgb_np_u8 = convert_linear_f32_to_u8(color_corrected_texture_rgb_f32)
    else:
        color_corrected_texture_rgb_np_u8 = np.round(color_corrected_texture_rgb_f32 * 255.0).clip(0, 255).astype(np.uint8)

    prii(color_corrected_texture_rgb_np_u8, caption="corrected_texture:")


    color_corrected_texture_rgba_f32 = np.zeros(
        shape=(
            color_corrected_texture_rgb_f32.shape[0],
            color_corrected_texture_rgb_f32.shape[1],
            4
        ),
        dtype=np.float32
    )

    color_corrected_texture_rgba_f32[:, :, :3] = color_corrected_texture_rgb_f32
    # kinda unclear why we are carrying around a 4th channel, maybe refactor.
    color_corrected_texture_rgba_f32[:, :, 3] = 1.0
    
    textured_ad_placement_descriptors = [
        copy.deepcopy(ad_placement_descriptor),
    ]

    textured_ad_placement_descriptors[0].texture_rgba_np_f32 = color_corrected_texture_rgba_f32
    
    mask_hw_np_u8[...] = 255 * (mask_hw_np_u8 > 128).astype(np.uint8)
    
   
    if use_linear_light:
        
        original_rgb_np_linear_f32 = convert_u8_to_linear_f32(original_rgb_np_u8)

        overwritten_with_its_own_ad = insert_quads_into_camera_posed_image_behind_mask(
            use_linear_light=use_linear_light,
            original_rgb_np_linear_f32=original_rgb_np_linear_f32,
            camera_pose=camera_pose,
            mask_hw_np_u8=255-mask_hw_np_u8,
            textured_ad_placement_descriptors=textured_ad_placement_descriptors,
            anti_aliasing_factor=2,
        )
        
        prii_linear_f32(
            original_rgb_np_linear_f32,
            caption=f"this is the actual original video frame, saved to {original_out_path}",
            out=original_out_path
        )
    
        prii_linear_f32(
            overwritten_with_its_own_ad,
            caption=f"this is augmented with its own ad with color correction, saved to {color_corrected_out_path}",
            out=color_corrected_out_path
        )
    else:
        print(f"{Fore.RED}WARNING: using non-linear light, this is not recommended for color correction{Style.RESET_ALL}")
        overwritten_with_its_own_ad = insert_quads_into_camera_posed_image_behind_mask(
            use_linear_light=use_linear_light,
            original_rgb_np_u8=original_rgb_np_u8,
            camera_pose=camera_pose,
            mask_hw_np_u8=255-mask_hw_np_u8,
            textured_ad_placement_descriptors=textured_ad_placement_descriptors,
        )
   
        prii(
            original_rgb_np_u8,
            caption="this is the original video frame, original.jpg:",
            out=original_out_path
        )
    
        prii(
            overwritten_with_its_own_ad,
            caption="this is augmented with its own ad with color correction:",
            out=color_corrected_out_path
        )

