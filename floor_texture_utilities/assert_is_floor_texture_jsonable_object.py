from is_sha256 import (
     is_sha256
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from open_as_rgba_hwc_np_u8 import (
     open_as_rgba_hwc_np_u8
)


def assert_is_floor_texture_jsonable_object(
     floor_texture_jsonable_object: dict   
) -> dict:
    """
    """
   
    texture_sha256 = floor_texture_jsonable_object["texture_sha256"]
    assert is_sha256(texture_sha256)
    
    uncorrected_margin_color_rgb = floor_texture_jsonable_object["uncorrected_margin_color_rgb"]

    assert len(uncorrected_margin_color_rgb) == 3
    for x in uncorrected_margin_color_rgb:
        assert isinstance(x, int)
        assert 0 <= x <= 255
    color_correction_sha256 = floor_texture_jsonable_object["color_correction_sha256"]
    assert is_sha256(color_correction_sha256)
    
    texture_width_in_pixels = floor_texture_jsonable_object["texture_width_in_pixels"]
    assert isinstance(texture_width_in_pixels, int)
    assert texture_width_in_pixels > 0
    texture_height_in_pixels = floor_texture_jsonable_object["texture_height_in_pixels"]
    assert isinstance(texture_height_in_pixels, int)
    assert texture_height_in_pixels > 0
    points = floor_texture_jsonable_object["points"]
    assert isinstance(points, dict)
    assert "top_left_interior_court_corner_xy" in points
    assert "bottom_right_interior_court_corner_xy" in points
    for key in points:
        assert isinstance(points[key], list)
        assert len(points[key]) == 2
        assert isinstance(points[key][0], int) or isinstance(points[key][0], float)
        assert isinstance(points[key][1], int) or isinstance(points[key][1], float)

    texture_sha256 = floor_texture_jsonable_object["texture_sha256"]
    image_file_path = get_file_path_of_sha256(sha256=texture_sha256)

    # this ensures that the image have 4 channels regardless of the original image:
    rgba_hwc_np_u8 = open_as_rgba_hwc_np_u8(
        image_path=image_file_path
    )
    assert rgba_hwc_np_u8.shape[2] == 4
    assert rgba_hwc_np_u8.shape[0] == texture_height_in_pixels
    assert rgba_hwc_np_u8.shape[1] == texture_width_in_pixels
