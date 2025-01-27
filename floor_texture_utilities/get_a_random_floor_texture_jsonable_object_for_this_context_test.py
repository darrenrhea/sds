import numpy as np
from prii_named_xy_points_on_image import (
     prii_named_xy_points_on_image
)
from get_a_random_floor_texture_jsonable_object_for_this_context import (
     get_a_random_floor_texture_jsonable_object_for_this_context
)
from is_sha256 import (
     is_sha256
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
from add_shadows import (
     add_shadows
)
from AdPlacementDescriptor import (
     AdPlacementDescriptor
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





def test_get_a_random_floor_texture_jsonable_object_for_this_context_1():
    """
    """
    floor_id = "24-25_HOU_CORE"
    possibility = get_a_random_floor_texture_jsonable_object_for_this_context(
        floor_id=floor_id,
    )
    texture_sha256 = possibility["texture_sha256"]
    assert is_sha256(texture_sha256)
    uncorrected_margin_color_rgb = possibility["uncorrected_margin_color_rgb"]
    
    assert len(uncorrected_margin_color_rgb) == 3
    for x in uncorrected_margin_color_rgb:
        assert isinstance(x, int)
        assert 0 <= x <= 255
    color_correction_sha256 = possibility["color_correction_sha256"]
    assert is_sha256(color_correction_sha256)
    
    texture_width_in_pixels = possibility["texture_width_in_pixels"]
    assert isinstance(texture_width_in_pixels, int)
    assert texture_width_in_pixels > 0
    texture_height_in_pixels = possibility["texture_height_in_pixels"]
    assert isinstance(texture_height_in_pixels, int)
    assert texture_height_in_pixels > 0
    points = possibility["points"]
    assert isinstance(points, dict)
    assert "top_left_interior_court_corner_xy" in points
    assert "bottom_right_interior_court_corner_xy" in points
    for key in points:
        assert isinstance(points[key], tuple)
        assert len(points[key]) == 2
        assert isinstance(points[key][0], int) or isinstance(points[key][0], float)
        assert isinstance(points[key][1], int) or isinstance(points[key][1], float)





    texture_sha256 = possibility["texture_sha256"]
    image_file_path = get_file_path_of_sha256(sha256=texture_sha256)

    # this ensures that the image have 4 channels regardless of the original image:
    rgba_hwc_np_u8 = open_as_rgba_hwc_np_u8(
        image_path=image_file_path
    )
    assert rgba_hwc_np_u8.shape[2] == 4
    assert rgba_hwc_np_u8.shape[0] == texture_height_in_pixels
    assert rgba_hwc_np_u8.shape[1] == texture_width_in_pixels

    print("Here are the points")
    prii_named_xy_points_on_image(
        name_to_xy=points,
        image=rgba_hwc_np_u8,
        output_image_file_path=None,
        default_color=(0, 255, 0),  # green is the default
        dont_show=False,
    )


    # add margins:
    margin_color_rgba = [171, 39, 34, 255]
    lm = 400
    rm = 400
    tm = 400
    bm = 400
    for point in points:
        points[point] = (
            points[point][0] - lm,
            points[point][1] - tm
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

    uncorrected_texture_rgba_np_linear_f32 = convert_u8_to_linear_f32(
        x=with_margin_added
    )



    color_correction_sha256 = possibility["color_correction_sha256"]

    degree, coefficients = load_color_correction_from_sha256(
       sha256=color_correction_sha256
    )

    print_in_iterm2 = True

    if print_in_iterm2:
        prii_linear_f32(
            x=uncorrected_texture_rgba_np_linear_f32,
            caption="this is the uncorrected floor texture",
        )

        
    color_corrected_texture_rgb_np_linear_f32 = color_correct_rgb_via_polynomial_coefficients_linear_f32_in_and_out(
        degree=degree,
        coefficients=coefficients,
        rgb_hwc_np_linear_f32=uncorrected_texture_rgba_np_linear_f32[:, :, :3],
    )

    if print_in_iterm2:
        prii_linear_f32(
            x=color_corrected_texture_rgb_np_linear_f32,
            caption="this is the color corrected floor texture",
        )

 



if __name__ == "__main__":
    test_get_a_random_floor_texture_jsonable_object_for_this_context_1()
