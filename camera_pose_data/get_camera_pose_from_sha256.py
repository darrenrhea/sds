from get_camera_pose_from_json_file_path import (
     get_camera_pose_from_json_file_path
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from CameraParameters import CameraParameters


def get_camera_pose_from_sha256(
    camera_pose_sha256
) -> CameraParameters:
    
    camera_pose_json_file_path = get_file_path_of_sha256(
        sha256=camera_pose_sha256
    )

    camera_pose = get_camera_pose_from_json_file_path(
        camera_pose_json_file_path
    )

    return camera_pose
