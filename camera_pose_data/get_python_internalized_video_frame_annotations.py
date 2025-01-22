import time
from typing import Optional
from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image
)
from get_camera_pose_from_json_file_path import (
     get_camera_pose_from_json_file_path
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from get_local_file_paths_for_annotations import (
     get_local_file_paths_for_annotations
)
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



def get_python_internalized_video_frame_annotations(
    video_frame_annotations_metadata_sha256: str,
    limit: Optional[int] = None,
    # TODO: take in desired_labels?
):
    """
    This is supposed to be a pretty general way to get all the video frame annotations that
    have certain labels, such as original, floor_not_floor, camera_pose, deth_map, etc.
    It highly depends on the availablity of the metadata file, which is given by its sha256.

    "python_internalized" here means that there are no file paths.
    For instance, the original RGB image is a hwc u8 numpy array, not a file path.
    The masks are hw u8 numpy arrays, not file paths.
    The camera_pose is a python object, not a file path to a json file.

    TODO: this should be a generator you can for loop over,
    rather than materializing all the numpy arrays in RAM and returning a list.
    At some point, we are going to have a lot of annotations,
    and loading them all into RAM is going to be a problem.
    """
    start_time = time.time()
    print("Starting get_work_items. Takes a while.")
    desired_labels = set(["camera_pose", "floor_not_floor", "original"])

    local_file_pathed_annotations = get_local_file_paths_for_annotations(
        video_frame_annotations_metadata_sha256=video_frame_annotations_metadata_sha256,
        desired_labels=desired_labels,
        max_num_annotations=None,
        print_in_iterm2=False,
        print_inadequate_annotations = False,
    )
    
    # np.random.shuffle(local_file_pathed_annotations)

    work_items = []
    for a in local_file_pathed_annotations:
        if limit is not None and len(work_items) >= limit:
            break

        clip_id = a["clip_id"]
        frame_index = a["frame_index"]
        local_file_paths = a["local_file_paths"]
        camera_pose_json_file_path = local_file_paths["camera_pose"]
        floor_not_floor_file_path = local_file_paths["floor_not_floor"]
        original_file_path = local_file_paths["original"]

        original_rgb_np_u8 = open_as_rgb_hwc_np_u8(
            image_path=original_file_path
        )
        camera_pose = get_camera_pose_from_json_file_path(
            camera_pose_json_file_path=camera_pose_json_file_path
        )

        floor_not_floor_hw_np_u8 = open_alpha_channel_image_as_a_single_channel_grayscale_image(
            abs_file_path=floor_not_floor_file_path
        )

        work_item = dict(
            clip_id=clip_id,
            frame_index=frame_index,
            camera_pose=camera_pose,
            camera_pose_json_file_path=camera_pose_json_file_path,
            floor_not_floor_hw_np_u8=floor_not_floor_hw_np_u8,
            original_rgb_np_u8=original_rgb_np_u8,
        )
        work_items.append(
            work_item
        )
    
    duration = time.time() - start_time
    print(f"Took {duration} seconds to get {len(work_items)} work items.")
    return work_items


def get_work_items_2():
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


