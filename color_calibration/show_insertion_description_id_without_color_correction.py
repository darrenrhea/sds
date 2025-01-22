import pyperclip
from get_ad_placement_descriptor_from_jsonable import (
     get_ad_placement_descriptor_from_jsonable
)
from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from get_video_frame_path_from_clip_id_and_frame_index import (
     get_video_frame_path_from_clip_id_and_frame_index
)
from insert_ad_into_camera_posed_original_video_frame import (
     insert_ad_into_camera_posed_original_video_frame
)
from pathlib import Path
import numpy as np
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from prii import (
     prii
)
from CameraParameters import (
     CameraParameters
)
import better_json as bj

import sys



def show_insertion_description_id_without_color_correction(
    insertion_description_id: str,
):
    """
    Shows an insertion_description
    probably so that you can check it is spacially well-aligned.
    You do not need color correction for this alignment process,
    and in fact the color correction is derived from this alignment,
    so aligning would usually come first.
    """
    
    insertion_desc_path = Path(
        f"~/r/color_correction_data/insertion_descriptions/{insertion_description_id}.json5"
    ).expanduser()

    insertion_desc = bj.load(
        insertion_desc_path
    )

    print(f"Loaded insertion_description from: {insertion_desc_path}")

    ad_placement_descriptor_jsonable = insertion_desc["ad_placement_descriptor"]

    ad_placement_descriptor = get_ad_placement_descriptor_from_jsonable(
        ad_placement_descriptor_jsonable=ad_placement_descriptor_jsonable
    )

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

    camera_pose = CameraParameters.from_dict(
        insertion_desc["camera_pose"]
    )

    camera_posed_original_video_frame = dict(
        original_file_path=original_file_path,
        frame_index=frame_index,
        clip_id=clip_id,
        camera_pose=camera_pose,
    )
    
    led_image_sha256 = insertion_desc["domain"]["sha256"]
    print(f"led_image_sha256: {led_image_sha256}")
    
    led_image_path = get_file_path_of_sha256(
        sha256=led_image_sha256
    )

    uncorrected_texture_rgb_np_u8 = open_as_rgb_hwc_np_u8(
        image_path=led_image_path
    )
    uncorrected_texture_rgb_np_u8 = uncorrected_texture_rgb_np_u8[i_min:i_max, j_min:j_max, :]

    prii(uncorrected_texture_rgb_np_u8, caption="uncorrected_texture_rgb_np_u8")

    mask_path = get_file_path_of_sha256(
        sha256=mask_sha256
    )

    prii(mask_path, caption="mask")

    mask_hw_np_u8 = open_alpha_channel_image_as_a_single_channel_grayscale_image(
        abs_file_path=mask_path
    )
    mask_hw_np_u8 = 255 * (mask_hw_np_u8 < 128).astype(np.uint8)

    

    texture_rgba_np_f32 = np.zeros(
        shape=(
            uncorrected_texture_rgb_np_u8.shape[0],
            uncorrected_texture_rgb_np_u8.shape[1],
            4
        ),
        dtype=np.float32
    )

    texture_rgba_np_f32[:, :, :3] = uncorrected_texture_rgb_np_u8
    # for figuring out the 3D alignment, this helps to make it a very different color
    # texture_rgba_np_f32[:, :, 0] = 0
    texture_rgba_np_f32[:, :, 3] = 255
    
    camera_pose = camera_posed_original_video_frame["camera_pose"]
    assert isinstance(camera_pose, CameraParameters)

    # We only insert one ad:
    ad_placement_descriptors = [
        ad_placement_descriptor,
    ]
    overwritten_with_its_own_ad = insert_ad_into_camera_posed_original_video_frame(
        original_rgb_np_u8=original_rgb_np_u8,
        texture_rgba_np_f32=texture_rgba_np_f32,
        ad_placement_descriptors=ad_placement_descriptors,
        camera_pose=camera_pose,
    )

    out_dir = Path(
        "~/uncorrected"
    ).expanduser()
    out_dir.mkdir(exist_ok=True, parents=True)
    original_out_path = out_dir / f"{clip_id}_{frame_index:06d}_original.jpg"
    color_corrected_out_path = out_dir / f"{clip_id}_{frame_index:06d}_insertion.jpg"
    prii(
        original_rgb_np_u8,
        caption=f"this is the original video frame, {clip_id}_{frame_index:06d}_original.jpg:",
        out=original_out_path
    )
   
    prii(
        overwritten_with_its_own_ad,
        caption="this is augmented with its own ad without any color correction:",
        out=color_corrected_out_path
    )


    s = f"flipflop {original_out_path} {color_corrected_out_path}"
    print(s)
    pyperclip.copy(s)
    
    
if __name__ == "__main__":
    insertion_description_id = sys.argv[1]
    show_insertion_description_id_without_color_correction(
        insertion_description_id=insertion_description_id
    )
