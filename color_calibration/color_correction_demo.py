from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from get_video_frame_path_from_clip_id_and_frame_index import (
     get_video_frame_path_from_clip_id_and_frame_index
)
from show_color_map_for_color_correction import (
     show_color_map_for_color_correction
)
from insert_ad_into_camera_posed_original_video_frame import (
     insert_ad_into_camera_posed_original_video_frame
)
from get_rgb_from_to_mapping_array import (
     get_rgb_from_to_mapping_array
)
from color_correct_in_lab_space import (
     color_correct_in_lab_space
)
from pathlib import Path
import numpy as np
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from get_world_coordinate_descriptors_of_ad_placements import (
     get_world_coordinate_descriptors_of_ad_placements
)
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from prii import (
     prii
)
from CameraParameters import (
     CameraParameters
)
import better_json as bj


def color_correction_demo():
    color_correction_context_id = "d49366a8-2459-4f33-ab47-843cb4cc0911"
    # color_correction_context_id = "5922a16d-0a20-467e-b72c-6a64ca78fabe"
    # color_correction_context_id = "4abb9876-9e07-4b0f-979c-6f51152341b1"
    color_correction_context = bj.load(
        f"~/r/color_correction_data/color_correction_contexts/{color_correction_context_id}.json5"
    )

    clip_id = color_correction_context["clip_id"]

    # the domain is the flat image they gave us that they claim they stuck into the LED board
    
    # the codomain image is from actual camera-recorded footage, so it has a camera pose:
    frame_index = color_correction_context["codomain"]["frame_index"]

    # what_color_corrector_to_use = color_correct_in_hsv_space
    what_color_corrector_to_use = color_correct_in_lab_space
    # what_color_corrector_to_use = color_correct_in_rgb_space

    # any camera_posed_human_annotation is also enough to be a camera_posed_original_video_frames:
    # camera_posed_original_video_frames = get_a_few_camera_posed_human_annotations()
    
   
    
    ad_placement_descriptors = get_world_coordinate_descriptors_of_ad_placements(
        clip_id=clip_id,
        with_floor_as_giant_ad=False,
        overcover_by=0.0,
    )

    ad_placement_descriptors = [
        x
        for x in ad_placement_descriptors
        if x.name != "LEDBRD0"
    ]
 
    original_file_path = get_video_frame_path_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index
    )

    original_rgb_np_u8 = open_as_rgb_hwc_np_u8(
        image_path=original_file_path
    )

    
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
    
    width_version_to_use = 1016
    # width_version_to_use = 1152

   

    led_image_sha256 = color_correction_context["domain"]["sha256"]
    led_image_path = get_file_path_of_sha256(
        sha256=led_image_sha256
    )
    texture_0_rgb_np_u8 = open_as_rgb_hwc_np_u8(
        image_path=led_image_path
    )

    if width_version_to_use == 1016:
        j_min = 58
        j_max = 967
        i_min = 14
        i_max = 126
    if width_version_to_use == 1152:
        j_min = 0
        j_max = 1152
        i_min = 0
        i_max = 144

    texture_0_rgb_np_u8 = texture_0_rgb_np_u8[i_min:i_max, j_min:j_max, :]


    texture_rgba_np_f32 = np.zeros(
        shape=(
            texture_0_rgb_np_u8.shape[0],
            texture_0_rgb_np_u8.shape[1],
            4
        ),
        dtype=np.float32
    )

    rgb_from_to_mapping_array = get_rgb_from_to_mapping_array(
        color_correction_context_id=color_correction_context_id
    )
    
    color_correct_it_yo = True
    if color_correct_it_yo:
        show_color_map_for_color_correction(
            color_map=rgb_from_to_mapping_array
        )
        color_corrected_rgb_u8s = what_color_corrector_to_use(
            rgb_from_to_mapping_array=rgb_from_to_mapping_array,
            uncorrected_rgbs=[
                texture_0_rgb_np_u8,
                rgb_from_to_mapping_array[:, 0:1, :]
            ]
        )
        color_corrected_texture_0_rgb_np_u8 = color_corrected_rgb_u8s[0]
        what_we_got = color_corrected_rgb_u8s[1]
        prii(what_we_got, caption="what we got:")
        from_desired_achieved = np.concatenate(
            [
                rgb_from_to_mapping_array[:, :, :].astype(np.uint8),
                what_we_got
            ],
            axis=1
        )
        bigger = from_desired_achieved.repeat(200, axis=0).repeat(200, axis=1)
        prii(bigger, caption="from_desired_achieved:")
        prii(texture_0_rgb_np_u8, caption="uncorrected_color:")
        prii(color_corrected_texture_0_rgb_np_u8, caption="corrected_color:")
    else:
        color_corrected_texture_0_rgb_np_u8 = texture_0_rgb_np_u8
    
    texture_rgba_np_f32[:, :, :3] = color_corrected_texture_0_rgb_np_u8
    # for figuring out the 3D alignment, this helps to make it a very different color
    # texture_rgba_np_f32[:, :, 0] = 0
    texture_rgba_np_f32[:, :, 3] = 255
    
    camera_pose = camera_posed_original_video_frame["camera_pose"]
    assert isinstance(camera_pose, CameraParameters)

    overwritten_with_its_own_ad = insert_ad_into_camera_posed_original_video_frame(
        original_rgb_np_u8=original_rgb_np_u8,
        texture_rgba_np_f32=texture_rgba_np_f32,
        ad_placement_descriptors=ad_placement_descriptors,
        camera_pose=camera_pose,
    )

    prii(
        original_rgb_np_u8,
        caption=f"this is the original video frame, {clip_id}_{frame_index:06d}_original.jpg:"
    )
   
    prii(overwritten_with_its_own_ad, caption="this is augmented with its own ad", out=Path("a.png"))


    print(f"flipflop {original_file_path} a.png")

     
if __name__ == "__main__":
    color_correction_demo()