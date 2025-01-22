from prii_linear_f32 import (
     prii_linear_f32
)
from unpack_insertion_description_id import (
     unpack_insertion_description_id
)
import numpy as np

from render_then_get_from_to_mapping_array import (
     render_then_get_from_to_mapping_array
)


def get_from_to_mapping_array_f64_from_insertion_description_id(
    insertion_description_id: str,
    use_linear_light: bool,
) -> np.ndarray:
    """
    Someone who has bothered to make an insertion_description i.e. has already done the work of
    1. finding a video frame that shows the ad in question in the LED board 
    2. aligning the camera pose, the ads corners, the ad surface, for instance we are not sure the y coordinate.
    3. making and saving the mask-for-regression,
    i.e. a mask that errs on the side of caution, so that it definitely does not include people or objects,
    just a large amount of the LED board in an actual frame.
    From this we make regression data points.
    At least when you are talking about the same court and the same LED board,
    there is some hope that a single optoelectronic transfer function will work for all the ads,
    so you might run this several times and concatenate the results, then train a regression.
    """

    assert len(insertion_description_id) == 36
    assert all(c in "abcdef0123456789-" for c in insertion_description_id)


    z = unpack_insertion_description_id(
        insertion_description_id=insertion_description_id,
        use_linear_light=use_linear_light,
    )

    original_rgb_np_u8 = z["original_rgb_np_u8"]
    camera_pose = z["camera_pose"]
    mask_for_regression_hw_np_u8 = z["mask_for_regression_hw_np_u8"]
    texture_rgba_np_f32 = z["texture_rgba_np_f32"]
    
    prii_linear_f32(
        texture_rgba_np_f32,
        caption="prii_linear_f32 inside get_from_to_mapping_array_f64_from_insertion_description_id"
    )

    ad_placement_descriptor = z["ad_placement_descriptor"]

    from_to_mapping_array_f64 = render_then_get_from_to_mapping_array(
        use_linear_light=use_linear_light,
        original_rgb_np_u8=original_rgb_np_u8,
        camera_pose=camera_pose,
        mask_for_regression_hw_np_u8=mask_for_regression_hw_np_u8,
        ad_placement_descriptor=ad_placement_descriptor,
        texture_rgba_np_f32=texture_rgba_np_f32,
    )
    
    num_pixels = np.sum(mask_for_regression_hw_np_u8 > 128)
    assert from_to_mapping_array_f64.shape[0] == num_pixels
    assert from_to_mapping_array_f64.shape[1] == 2
    assert from_to_mapping_array_f64.shape[2] == 3
    assert from_to_mapping_array_f64.dtype == np.float64

    if use_linear_light:
        assert np.max(from_to_mapping_array_f64) <= 1.01, f"{np.max(from_to_mapping_array_f64)=} but the color cube is supposed to be [0, 1]^3"
        assert np.min(from_to_mapping_array_f64) >= -0.01, f"{np.min(from_to_mapping_array_f64)=} but the color cube is supposed to be [0, 1]^3"

    return from_to_mapping_array_f64
