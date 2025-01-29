from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image
)
from get_camera_pose_from_json_file_path import (
     get_camera_pose_from_json_file_path
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)


def make_python_internalized_video_frame_annotation(
    local_file_pathed_annotation: dict,
) -> dict:
    """
    This is supposed to convert a video frame annotation that points at local file paths
    to a python-internalized video frame annotation.

    "python_internalized" here means that there are no file paths, but rather the data is stored in RAM
    as python objects.

    For instance, the original RGB image is a hwc u8 numpy array, not a file path.
    The masks are hw u8 numpy arrays, not file paths.
    The camera_pose is a CameraParameters python object, not a file path to a json file.
    """

    clip_id = local_file_pathed_annotation["clip_id"]
    frame_index = local_file_pathed_annotation["frame_index"]
    local_file_paths = local_file_pathed_annotation["local_file_paths"]

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
        floor_not_floor_hw_np_u8=floor_not_floor_hw_np_u8,
        original_rgb_np_u8=original_rgb_np_u8,
        camera_pose_json_file_path=camera_pose_json_file_path,  # stowaway for debugging?
    )
    
    return work_item
