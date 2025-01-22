import sys
from display_alignment_quality import (
     display_alignment_quality
)
from unpack_insertion_description_id import (
     unpack_insertion_description_id
)
from pathlib import Path



def test_display_alignment_quality_1():
    """
    Example:
    
    osri_optimize_self_reproducing_insertion 

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
   
    insertion_description_id = "2a04b7dd-8d83-4455-927e-002b16b11128"

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

    print("Finished displaying_alignment_quality")


if __name__ == "__main__":
    test_display_alignment_quality_1()
    print("display_alignment_quality_test.py has finished running")
    sys.exit(0)