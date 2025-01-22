from show_color_correction_result import (
     show_color_correction_result
)
from unpack_insertion_description_id import (
     unpack_insertion_description_id
)
from math import comb
from pathlib import Path
import numpy as np


def show_color_correction_result_on_insertion_description_id(
    # give the color correction polynomial coefficients:
    use_linear_light: bool,
    degree: int,
    coefficients: np.ndarray,
    # give the self_reproducing_insertion_description_id:
    insertion_description_id: str,
    # where to save it for flipflopping
    out_dir: Path,
):
    """
    Say you already fit a polynomial to color correct.
    Show how well it works on an self_reproducing_insertion_description
    so that you can flip flop it.
    """
    assert isinstance(degree, int)
    assert degree >= 1
    assert isinstance(coefficients, np.ndarray)
    assert coefficients.shape[1] == 3
    
    num_coefficients_should_be = comb(degree + 3, 3)

    assert (
        coefficients.shape[0] == num_coefficients_should_be
    ), f"ERROR: {coefficients.shape=}, {num_coefficients_should_be=}"

    ans = unpack_insertion_description_id(
        insertion_description_id=insertion_description_id,
        use_linear_light=use_linear_light,
    )
    # unpack ans:
    original_rgb_np_u8 = ans["original_rgb_np_u8"]
    camera_pose = ans["camera_pose"]
    mask_for_regression_hw_np_u8 = ans["mask_for_regression_hw_np_u8"]
    # texture_rgba_np_f32 = ans["texture_rgba_np_f32"]
    ad_placement_descriptor = ans["ad_placement_descriptor"]
    uncorrected_texture_rgb_np_u8 = ans["uncorrected_texture_rgb_np_u8"]
    del ans

    color_corrected_out_path = out_dir / "corrected.png"
    original_out_path = out_dir / "original.png"

    show_color_correction_result(
        use_linear_light=use_linear_light,
        degree=degree,
        coefficients=coefficients,
        original_rgb_np_u8=original_rgb_np_u8,
        camera_pose=camera_pose,
        ad_placement_descriptor=ad_placement_descriptor,
        mask_hw_np_u8=mask_for_regression_hw_np_u8,
        uncorrected_texture_rgb_np_u8=uncorrected_texture_rgb_np_u8,
        original_out_path=original_out_path,
        color_corrected_out_path=color_corrected_out_path,
    )
