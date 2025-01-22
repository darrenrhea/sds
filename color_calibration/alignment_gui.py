from immediate_mode_gui_maker_simple_version import (
     immediate_mode_gui_maker_simple_version
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from pathlib import Path
import numpy as np
import pygame

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
from prii import (
     prii
)
from CameraParameters import (
     CameraParameters
)
import better_json as bj


def create_initial_state():
    """
    Shows an insertion_description
    probably so that you can check it is spacially well-aligned.
    You do not need color correction for this alignment process,
    and in fact the color correction is derived from this alignment,
    so aligning would usually come first.
    """
    insertion_description_id = "e4e347df-b49a-4f00-8990-d5d7489b0812"
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

    state = dict(
        original_rgb_np_u8=original_rgb_np_u8,
        texture_rgba_np_f32=texture_rgba_np_f32,
        ad_placement_descriptors=ad_placement_descriptors,
        camera_pose=camera_pose, 
        overwritten_with_its_own_ad=overwritten_with_its_own_ad,  
    )
    return state



def mutate_state_according_to_keys(keys, state):
    """
    Something has to mutate the state when people press keys.
    """
    camera_pose = state["camera_pose"]
    k1_delta = 0.01
    k2_delta = 0.1
    k1 = camera_pose.k1
    k2 = camera_pose.k2

    if keys[pygame.K_DOWN]:
        k1 -= k1_delta
    if keys[pygame.K_UP]:
        k1 += k2_delta

    if keys[pygame.K_LEFT]:
        k2 -= k2_delta
    if keys[pygame.K_RIGHT]:
        k2 += k2_delta
    
    camera_pose.k1 = k1
    camera_pose.k2 = k2
    state["camera_pose"] = camera_pose


def make_the_rgb_hwc_np_u8_you_want_to_display(state):
    # unpack_the_state:
    original_rgb_np_u8 = state["original_rgb_np_u8"]
    texture_rgba_np_f32 = state["texture_rgba_np_f32"]
    ad_placement_descriptors = state["ad_placement_descriptors"]
    camera_pose = state["camera_pose"]

    overwritten_with_its_own_ad = insert_ad_into_camera_posed_original_video_frame(
        original_rgb_np_u8=original_rgb_np_u8,
        texture_rgba_np_f32=texture_rgba_np_f32,
        ad_placement_descriptors=ad_placement_descriptors,
        camera_pose=camera_pose,
    )

    return overwritten_with_its_own_ad




def make_texts_to_display(state):
    camera_pose = state["camera_pose"]
    k1 = camera_pose.k1
    k2 = camera_pose.k2
    texts = []
    texts.append(f"{k1=}")
    texts.append(f"{k2=}")
    return texts



if __name__ == "__main__":
    immediate_mode_gui_maker_simple_version(
        create_initial_state=create_initial_state,
        make_the_rgb_hwc_np_u8_you_want_to_display=make_the_rgb_hwc_np_u8_you_want_to_display,
        make_texts_to_display=make_texts_to_display,
        mutate_state_according_to_keys=mutate_state_according_to_keys

    )