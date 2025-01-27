from add_an_opaque_alpha_channel_to_create_rgba_hwc_np_f32 import (
     add_an_opaque_alpha_channel_to_create_rgba_hwc_np_f32
)
from assert_is_floor_texture_jsonable_object import (
     assert_is_floor_texture_jsonable_object
)
import numpy as np
from prii_named_xy_points_on_image import (
     prii_named_xy_points_on_image
)
from convert_u8_to_linear_f32 import (
     convert_u8_to_linear_f32
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from open_as_rgba_hwc_np_u8 import (
     open_as_rgba_hwc_np_u8
)
from load_color_correction_from_sha256 import (
     load_color_correction_from_sha256
)
from prii_linear_f32 import (
     prii_linear_f32
)
from color_correct_rgb_via_polynomial_coefficients_linear_f32_in_and_out import (
     color_correct_rgb_via_polynomial_coefficients_linear_f32_in_and_out
)





def get_color_corrected_floor_texture_with_margin_added(
     floor_texture_jsonable_object: dict,
     verbose=False,
) -> dict:
    """
    """
    assert_is_floor_texture_jsonable_object(floor_texture_jsonable_object)
   
    texture_sha256 = floor_texture_jsonable_object["texture_sha256"]
    uncorrected_margin_color_rgb = floor_texture_jsonable_object["uncorrected_margin_color_rgb"]

    color_correction_sha256 = floor_texture_jsonable_object["color_correction_sha256"]
    texture_width_in_pixels = floor_texture_jsonable_object["texture_width_in_pixels"]
    texture_height_in_pixels = floor_texture_jsonable_object["texture_height_in_pixels"]
    points = floor_texture_jsonable_object["points"]
    texture_sha256 = floor_texture_jsonable_object["texture_sha256"]
    image_file_path = get_file_path_of_sha256(sha256=texture_sha256)

    # this ensures that the image have 4 channels regardless of whether the original image has 3 or 4 channels:
    rgba_hwc_np_u8 = open_as_rgba_hwc_np_u8(
        image_path=image_file_path
    )
    assert rgba_hwc_np_u8.shape[2] == 4
    assert rgba_hwc_np_u8.shape[0] == texture_height_in_pixels
    assert rgba_hwc_np_u8.shape[1] == texture_width_in_pixels

    if verbose:
        print("Here are the points prior to adding margins:")
        prii_named_xy_points_on_image(
            name_to_xy=points,
            image=rgba_hwc_np_u8,
            output_image_file_path=None,
            default_color=(0, 255, 0),  # green is the default
            dont_show=False,
        )


    # add margins:
    margin_color_rgba = [
        uncorrected_margin_color_rgb[0],
        uncorrected_margin_color_rgb[1],
        uncorrected_margin_color_rgb[2],
        255
    ]
    
    lm = 600
    rm = 600
    tm = 600
    bm = 600
    new_points = dict()
    for point_name in points:
        new_points[point_name] = (
            points[point_name][0] + lm,
            points[point_name][1] + tm
        )
    
    with_margin_added = np.zeros(
        shape=(
            texture_height_in_pixels + tm + bm,
            texture_width_in_pixels + lm + rm,
            4
        ),
        dtype=np.uint8
    )

    with_margin_added += np.array(margin_color_rgba, dtype=np.uint8)
    with_margin_added[tm:tm + texture_height_in_pixels, lm:lm + texture_width_in_pixels] = rgba_hwc_np_u8

    
    if verbose:
        print("Here are the points after adding margins:")
        prii_named_xy_points_on_image(
            name_to_xy=new_points,
            image=with_margin_added,
            output_image_file_path=None,
            default_color=(0, 255, 0),  # green is the default
            dont_show=False,
        )

    # go to linear light:
    uncorrected_texture_rgba_np_linear_f32 = convert_u8_to_linear_f32(
        x=with_margin_added
    )

    color_correction_sha256 = floor_texture_jsonable_object["color_correction_sha256"]

    degree, coefficients = load_color_correction_from_sha256(
       sha256=color_correction_sha256
    )


    if verbose:
        prii_linear_f32(
            x=uncorrected_texture_rgba_np_linear_f32,
            caption="this is the uncorrected floor texture",
        )

        
    color_corrected_texture_rgb_np_linear_f32 = color_correct_rgb_via_polynomial_coefficients_linear_f32_in_and_out(
        degree=degree,
        coefficients=coefficients,
        rgb_hwc_np_linear_f32=uncorrected_texture_rgba_np_linear_f32[:, :, :3],
    )

    if verbose:
        prii_linear_f32(
            x=color_corrected_texture_rgb_np_linear_f32,
            caption="this is the color corrected floor texture",
        )

    color_corrected_texture_rgba_np_linear_f32 = add_an_opaque_alpha_channel_to_create_rgba_hwc_np_f32(
            color_corrected_texture_rgb_np_linear_f32
    )

    answer = dict(
        color_corrected_texture_rgba_np_linear_f32=color_corrected_texture_rgba_np_linear_f32,
        points=new_points,
    )

    return answer

