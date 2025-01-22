from prii_nonlinear_f32 import (
     prii_nonlinear_f32
)
from AdPlacementDescriptor import (
     AdPlacementDescriptor
)
from colorama import Fore, Style
from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from get_video_frame_path_from_clip_id_and_frame_index import (
     get_video_frame_path_from_clip_id_and_frame_index
)
import numpy as np
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from CameraParameters import (
     CameraParameters
)
import better_json as bj
from get_ad_placement_descriptor_from_jsonable import (
     get_ad_placement_descriptor_from_jsonable
)




def unpack_insertion_description_id(
    insertion_description_id: str,
    use_linear_light: bool,
) -> np.ndarray:
    """
    Someone who has bothered to make an self_reproducing_insertion_description i.e. has already done the work of
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
    
    insertion_desc = bj.load(
        f"~/r/color_correction_data/insertion_descriptions/{insertion_description_id}.json5"
    )

    ad_placement_descriptor_jsonable = insertion_desc["ad_placement_descriptor"]

    ad_placement_descriptor = get_ad_placement_descriptor_from_jsonable(
        ad_placement_descriptor_jsonable=ad_placement_descriptor_jsonable
    )
    assert isinstance(ad_placement_descriptor, AdPlacementDescriptor)

    clip_id = insertion_desc["codomain"]["clip_id"]
    mask_sha256 = insertion_desc["codomain"]["mask_for_regression"]["sha256"]

    # get info about the LED video frame to insert:
    subrectangle = insertion_desc["domain"]["subrectangle"]
    i_min = subrectangle["i_min"]
    i_max = subrectangle["i_max"]
    j_min = subrectangle["j_min"]
    j_max = subrectangle["j_max"]

    # the domain is the flat image they gave us that they claim they stuck into the LED board
    
    # the codomain image is from actual camera-recorded footage, so it has a camera pose:
    frame_index = insertion_desc["codomain"]["frame_index"]

    original_file_path = get_video_frame_path_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index
    )

    original_rgb_np_u8 = open_as_rgb_hwc_np_u8(
        image_path=original_file_path
    )

    if "camera_pose" in insertion_desc:
        loc = insertion_desc["camera_pose"]["loc"]
        rod = insertion_desc["camera_pose"]["rod"]
        f = insertion_desc["camera_pose"]["f"]
        k1 = insertion_desc["camera_pose"]["k1"]
        k2 = insertion_desc["camera_pose"]["k2"]
        camera_pose = CameraParameters(
            loc=loc,
            rod=rod,
            f=f,
            k1=k1,
            k2=k2
        )
    else:
        print(f"{Fore.RED}WARNING: no camera_pose in self-reproducing-insertion-description {insertion_description_id}{Style.RESET_ALL}")
        camera_pose = get_camera_pose_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index
        ) 

    camera_posed_original_video_frame = dict(
        original_file_path=original_file_path,
        frame_index=frame_index,
        clip_id=clip_id,
        camera_pose=camera_pose,
    )
    camera_pose = camera_posed_original_video_frame["camera_pose"]
    assert isinstance(camera_pose, CameraParameters)

    
    led_image_sha256 = insertion_desc["domain"]["sha256"]
    
    led_image_path = get_file_path_of_sha256(
        sha256=led_image_sha256
    )

    uncorrected_texture_rgb_np_u8 = open_as_rgb_hwc_np_u8(
        image_path=led_image_path
    )
    
    mask_path = get_file_path_of_sha256(
        sha256=mask_sha256
    )

    mask_for_regression_hw_np_u8 = open_alpha_channel_image_as_a_single_channel_grayscale_image(
        abs_file_path=mask_path
    )
    mask_for_regression_hw_np_u8 = 255 * (mask_for_regression_hw_np_u8 < 128).astype(np.uint8)

    uncorrected_texture_rgb_np_u8 = uncorrected_texture_rgb_np_u8[i_min:i_max, j_min:j_max, :]

    texture_rgba_np_f32 = np.zeros(
        shape=(
            uncorrected_texture_rgb_np_u8.shape[0],
            uncorrected_texture_rgb_np_u8.shape[1],
            4
        ),
        dtype=np.float32
    )

   
    if use_linear_light:
        texture_rgba_np_f32[:, :, :3] = (uncorrected_texture_rgb_np_u8.astype(np.float32) / 255.0) ** 2.2
        texture_rgba_np_f32[:, :, 3] = 1.0

    else:
        texture_rgba_np_f32[:, :, :3] = (uncorrected_texture_rgb_np_u8.astype(np.float32) / 255.0)
        texture_rgba_np_f32[:, :, 3] = 255
        prii_nonlinear_f32(
            texture_rgba_np_f32,
            caption="texture_rgba_np_f32"
        )
    
    ans = dict(
        original_rgb_np_u8=original_rgb_np_u8,
        camera_pose=camera_pose,
        mask_for_regression_hw_np_u8=mask_for_regression_hw_np_u8,
        texture_rgba_np_f32=texture_rgba_np_f32,
        ad_placement_descriptor=ad_placement_descriptor,
        uncorrected_texture_rgb_np_u8=uncorrected_texture_rgb_np_u8,
    )

    return ans 

