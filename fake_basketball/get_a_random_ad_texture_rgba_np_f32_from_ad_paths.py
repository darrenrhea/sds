from force_long_aspect_ratio import (
     force_long_aspect_ratio
)
from color_correct_rgb_via_polynomial_coefficients_linear_f32_in_and_out import (
     color_correct_rgb_via_polynomial_coefficients_linear_f32_in_and_out
)
from color_correct_rgb_hwc_np_u8_image_via_polynomial_coefficients_u8_in_f32_out import (
     color_correct_rgb_hwc_np_u8_image_via_polynomial_coefficients_u8_in_f32_out
)
from convert_u8_to_linear_f32 import (
     convert_u8_to_linear_f32
)
from prii import (
     prii
)
from color_correct_rgb_hwc_np_u8_image_via_polynomial_coefficients import (
     color_correct_rgb_hwc_np_u8_image_via_polynomial_coefficients
)
from augment_texture import (
     augment_texture
)
import numpy as np
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)



def get_a_random_ad_texture_rgba_np_f32_from_ad_paths(
    use_linear_light: bool,
    degree: int,
    coefficients: np.ndarray,
    albu_transform,
    ad_name_to_paths_that_dont_need_color_correction,
    ad_name_to_paths_that_do_need_color_correction,
):
    """
    TODO: delete this.
    """
    all_ad_names = set()
    
    for key, value in ad_name_to_paths_that_dont_need_color_correction.items():
        if len(value) > 0:
            all_ad_names.add(key)
    
    for key, value in ad_name_to_paths_that_do_need_color_correction.items():
        if len(value) > 0:
            all_ad_names.add(key)
    
    ad_name = np.random.choice(list(all_ad_names))

    print(f"We randomly chose the ad {ad_name}")

    ad_paths_that_do_need_color_correction = ad_name_to_paths_that_do_need_color_correction.get(ad_name, [])
    ad_paths_that_dont_need_color_correction = ad_name_to_paths_that_dont_need_color_correction.get(ad_name, [])

    assert (
        len(ad_paths_that_dont_need_color_correction) > 0
        or
        len(ad_paths_that_do_need_color_correction) > 0
    ), f"ERROR: {ad_name=} has no PNGs for it."

    which = np.random.randint(0, 2)
    if len(ad_paths_that_dont_need_color_correction) == 0:
        which = 0
        # print("Since there no ripped ads available, using only ads that need color correction.")
    if len(ad_paths_that_do_need_color_correction) == 0:
        which = 1
        # print("Using one they gave us")
        # print("Since there are no ads that need color correction (at least that we know how to correct), only using rips.")

    
    if which == 0: # pick one that needs color correction
        index = np.random.randint(0, len(ad_paths_that_do_need_color_correction))
        ad_texture_png_path = ad_paths_that_do_need_color_correction[index]
        needs_color_correction = True
        # print(f"Chose {ad_texture_png_path} which needs color correction")

    else: # pick one that doesn't need color correction:        
        index = np.random.randint(0, len(ad_paths_that_dont_need_color_correction))
        ad_texture_png_path = ad_paths_that_dont_need_color_correction[index]

        needs_color_correction = False
        # print(f"Chose {ad_texture_png_path} which does not need color correction")

    # they never have transparent bits since they are for the LED board:
    unaugmented_texture_rgb_np_u8 = open_as_rgb_hwc_np_u8(ad_texture_png_path)

    if which == 0:
        texture_rgb_np_u8 = augment_texture(
            rgb_np_u8=unaugmented_texture_rgb_np_u8,
            transform=albu_transform
        )
    else:  # which == 1 might mean doens't need color correction, probably a rip.
        rand_bit = np.random.randint(0, 2)
        if rand_bit == 0:
            texture_rgb_np_u8 = unaugmented_texture_rgb_np_u8
        else:
            texture_rgb_np_u8 = augment_texture(
                rgb_np_u8=unaugmented_texture_rgb_np_u8,
                transform=albu_transform
            )
    # this was only for dallas_mavericks, which needed triplication sometimes:
    # if which == 1:
    #     texture_rgb_np_u8 = force_long_aspect_ratio(texture_rgb_np_u8)
    #     assert texture_rgb_np_u8.shape[0] == 96, f"ERROR: {texture_rgb_np_u8.shape=}"
    #     assert texture_rgb_np_u8.shape[1] == 2560, f"ERROR: {texture_rgb_np_u8.shape=}"

    
    texture_rgb_np_linear_f32 = convert_u8_to_linear_f32(texture_rgb_np_u8)

    if use_linear_light:
        if needs_color_correction:   
            color_corrected_texture_rgb_np_linear_f32 = color_correct_rgb_via_polynomial_coefficients_linear_f32_in_and_out(
                degree=degree,
                coefficients=coefficients,
                rgb_hwc_np_linear_f32=texture_rgb_np_linear_f32
            )
        else:  # nothing to do
            color_corrected_texture_rgb_np_linear_f32 = texture_rgb_np_linear_f32
        
        # we have to add an alpha channel to the texture even though it's not transparent:
        color_corrected_texture_rgba_np_linear_f32 = np.zeros(
            shape=(texture_rgb_np_u8.shape[0], texture_rgb_np_u8.shape[1], 4),
            dtype=np.float32
        )

        color_corrected_texture_rgba_np_linear_f32[:, :, :3] = color_corrected_texture_rgb_np_linear_f32
        color_corrected_texture_rgba_np_linear_f32[:, :, 3] = 1.0
        

        return color_corrected_texture_rgba_np_linear_f32

           
    else: 
        if needs_color_correction: 
            color_corrected_texture_rgb_np_u8 = color_correct_rgb_hwc_np_u8_image_via_polynomial_coefficients(
                degree=degree,
                coefficients=coefficients,
                rgb_hwc_np_u8=texture_rgb_np_u8,
            )          
        else:
            color_corrected_texture_rgb_np_u8 = texture_rgb_np_u8
    

        # we have to add an alpha channel to the texture even though it's not transparent:
        color_corrected_texture_rgba_np_u8 = np.zeros(
            shape=(texture_rgb_np_u8.shape[0], texture_rgb_np_u8.shape[1], 4),
            dtype=np.uint8
        )

        color_corrected_texture_rgba_np_u8[:, :, :3] = color_corrected_texture_rgb_np_u8
        color_corrected_texture_rgba_np_u8[:, :, 3] = 255
        
        ad_texture_rgba_np_f32 = color_corrected_texture_rgba_np_u8.astype(np.float32)

        return ad_texture_rgba_np_f32
