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
from get_local_file_pathed_annotations import (
     get_local_file_pathed_annotations
)


def get_python_internalized_video_frame_annotations(
    video_frame_annotations_metadata_sha256: str,
    limit: Optional[int] = None,
    # TODO: take in desired_labels?
):
    """
    Not sure we should do this.  It is convenient, but loads all the annotations into RAM.
    Better might be to internalize just one annotation at a time.

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

    local_file_pathed_annotations = get_local_file_pathed_annotations(
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
