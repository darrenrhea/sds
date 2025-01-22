import sys
import textwrap
from display_alignment_quality import (
     display_alignment_quality
)
from render_then_get_from_to_mapping_array import (
     render_then_get_from_to_mapping_array
)
from optimize_self_insertion import (
     optimize_self_insertion
)
from unpack_insertion_description_id import (
     unpack_insertion_description_id
)
import pprint as pp
import pyperclip
from save_color_correction_as_json import (
     save_color_correction_as_json
)
from show_color_correction_result_on_insertion_description_id import (
     show_color_correction_result_on_insertion_description_id
)
from get_color_correction_polynomial_coefficients_from_from_to_mapping_array_f64 import (
     get_color_correction_polynomial_coefficients_from_from_to_mapping_array_f64
)
from pathlib import Path



def osri_optimize_self_reproducing_insertion_cli_tool():
    """
    Example:
    
    osri_optimize_self_reproducing_insertion 2a04b7dd-8d83-4455-927e-002b16b11128

    Let's do
    ESPN_MIL_IND_FRI
    because it has a black background and red yellow blue and green and white.

    A "self_reproducing_insertion_description" is a bundle of:
    an original video frame
    its camera pose
    A mask that selects a large part of the LED board, but which
    definitely does not contain people nor other objects occlude the LED board.
    A 2-dimensional ad jpg that they sent us, that was been displayed on the the LED board at that time the video frame was taken.
    Maybe a description of a subrectangle of that image,
    because actually we insert only a subrectangle of the image.
    3D world coordinates of the 4 corners of the LED board that tell us where to insert.
    """
   
    insertion_description_id = sys.argv[1]
    assert len(insertion_description_id) == 36
    assert all(c in "abcdef0123456789-" for c in insertion_description_id)

    use_linear_light = False

    z = unpack_insertion_description_id(
        insertion_description_id=insertion_description_id,
        use_linear_light=use_linear_light,
    )

    original_rgb_np_u8 = z["original_rgb_np_u8"]
    camera_pose = z["camera_pose"]
    mask_for_regression_hw_np_u8 = z["mask_for_regression_hw_np_u8"]
    texture_rgba_np_f32 = z["texture_rgba_np_f32"]
    ad_placement_descriptor = z["ad_placement_descriptor"]

    display_alignment_quality(
        original_rgb_np_u8=original_rgb_np_u8,
        camera_pose=camera_pose,
        mask_for_regression_hw_np_u8=mask_for_regression_hw_np_u8,
        ad_placement_descriptor=ad_placement_descriptor,
        texture_rgba_np_f32=texture_rgba_np_f32,
        check_alignment_dir=Path.home() / "check_alignment",
    )

    print("Finished displaying alignment quality")
   
    best_ad_placement_descriptor, best_camera_pose = optimize_self_insertion(
        original_rgb_np_u8=original_rgb_np_u8,
        camera_pose=camera_pose,
        mask_for_regression_hw_np_u8=mask_for_regression_hw_np_u8,
        texture_rgba_np_f32=texture_rgba_np_f32,
        ad_placement_descriptor=ad_placement_descriptor,
        max_iters=10,
    )

    print("Going forward with the best_ad_placement_descriptor:")
    pp.pprint(best_ad_placement_descriptor)
    print("and the best_camera_pose:")
    print(best_camera_pose)

    display_alignment_quality(
        original_rgb_np_u8=original_rgb_np_u8,
        camera_pose=best_camera_pose,
        mask_for_regression_hw_np_u8=mask_for_regression_hw_np_u8,
        ad_placement_descriptor=best_ad_placement_descriptor,
        texture_rgba_np_f32=texture_rgba_np_f32,
        check_alignment_dir=Path.home() / "check_alignment2"
    )

    from_to_mapping_array_f64 = render_then_get_from_to_mapping_array(
        original_rgb_np_u8=original_rgb_np_u8,
        camera_pose=best_camera_pose,
        mask_for_regression_hw_np_u8=mask_for_regression_hw_np_u8,
        ad_placement_descriptor=best_ad_placement_descriptor,
        texture_rgba_np_f32=texture_rgba_np_f32,
        use_linear_light=False,
    )

    degree = 3

    coefficients = get_color_correction_polynomial_coefficients_from_from_to_mapping_array_f64(
        degree=degree,
        from_to_mapping_array_f64=from_to_mapping_array_f64,
    )

    color_correction_out_path = Path.home() / "color_correction.json"
    save_color_correction_as_json(
        degree=degree,
        coefficients=coefficients,
        out_path=color_correction_out_path
    )
    print(f"saved color correction to {color_correction_out_path}")

    out_dir = Path(
        "~/color_corrected"
    ).expanduser()

    out_dir.mkdir(exist_ok=True, parents=True)


    show_color_correction_result_on_insertion_description_id(
        insertion_description_id=insertion_description_id,
        degree=degree,
        coefficients=coefficients,
        out_dir=out_dir,
        use_linear_light=use_linear_light,
    )
   

    s = "flipflop ~/color_corrected"
    pyperclip.copy(s)
    print("We suggest you run the following command:")
    print(s)
    print("you can just paste it since it is already on the clipboard")

    print(
        textwrap.dedent(
            f"""\
            "camera_pose": {{
                "rod": [{best_camera_pose.rod[0]}, {best_camera_pose.rod[1]}, {best_camera_pose.rod[2]}],
                "loc": [{best_camera_pose.loc[0]}, {best_camera_pose.loc[1]}, {best_camera_pose.loc[2]}],
                "f": {best_camera_pose.f},
                "k1": {best_camera_pose.k1},
                "k2": {best_camera_pose.k2},
            }},
            """
        )
    )
    

    print(
        textwrap.dedent(
            f"""\
            "ad_placement_descriptor": {{
                "origin": [{best_ad_placement_descriptor.origin[0]}, {best_ad_placement_descriptor.origin[1]}, {best_ad_placement_descriptor.origin[2]}],
                "u": [{best_ad_placement_descriptor.u[0]}, {best_ad_placement_descriptor.u[1]}, {best_ad_placement_descriptor.u[2]}],
                "v": [{best_ad_placement_descriptor.v[0]}, {best_ad_placement_descriptor.v[1]}, {best_ad_placement_descriptor.v[2]}],
                "height": {best_ad_placement_descriptor.height},
                "width": {best_ad_placement_descriptor.width},
            }},
            """
        )
    )
 
 
