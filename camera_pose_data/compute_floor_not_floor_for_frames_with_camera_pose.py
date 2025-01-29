from get_mask_hw_np_u8_from_clip_id_and_frame_index_and_convention_and_final_model_id import (
     get_mask_hw_np_u8_from_clip_id_and_frame_index_and_convention_and_final_model_id
)
from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
import numpy as np




def compute_floor_not_floor_for_frames_with_camera_pose():
    """
    hot computed floor_not_floor masks for things that do not have committed human annotations.
    """
    # This needs both tracking info and floor_not_floor annotations.
    # The hardest part is finding not just floor_not_floor annotations, but ones that have a camera pose.
    # we can compute the floor_not_floor annotations if we have a model that basically works.
    
    
    print_in_iterm2 = False
    clip_ids = [
        "slgame1",
        "slday2game1",
        "slday3game1",
        "slday4game1",
        "slday5game1",
        "slday6game1",
        "slday8game1",
        "slday9game1",
        "slday10game1",
    ]
    # somebody already staged inferences of these:
    frame_indices = list(range(150000, 300000 + 1, 1000))
    segmentation_convention = "floor_not_floor"
    final_model_id = "human"  # human is a sentinel value meaning not a model at all but human annotation.

    work_items = []
    for frame_index in frame_indices:
        for clip_id in clip_ids:
            camera_pose = get_camera_pose_from_clip_id_and_frame_index(
                clip_id=clip_id,
                frame_index=frame_index,
            )
            if camera_pose.f == 0.0:
                print("camera_pose.f is 0.0, so we are not going to be able to do anything.")
                continue
            
            original_rgb_np_u8 = get_original_frame_from_clip_id_and_frame_index(
                clip_id=clip_id,
                frame_index=frame_index,
            )

            floor_not_floor_hw_np_u8 = get_mask_hw_np_u8_from_clip_id_and_frame_index_and_convention_and_final_model_id(
                clip_id=clip_id,
                frame_index=frame_index,
                segmentation_convention=segmentation_convention,
                final_model_id=final_model_id,
            )
            work_item = dict(
                clip_id=clip_id,
                frame_index=frame_index,
                camera_pose=camera_pose,
                floor_not_floor_hw_np_u8=floor_not_floor_hw_np_u8,
                original_rgb_np_u8=original_rgb_np_u8,
            )
            work_items.append(
                work_item
            )
        
    return work_items


