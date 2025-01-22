from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
import better_json as bj
from CameraParameters import CameraParameters



def get_camera_pose_from_json_file_path(
    camera_pose_json_file_path
) -> CameraParameters:
    camera_pose_jsonable = bj.load(
        camera_pose_json_file_path
    )
    
    camera_pose = CameraParameters(
        rod=camera_pose_jsonable["rod"],
        loc=camera_pose_jsonable["loc"],
        f=camera_pose_jsonable["f"],
        ppi=0.0,
        ppj=0.0,
        k1=camera_pose_jsonable["k1"],
        k2=camera_pose_jsonable["k2"],
    )

    return camera_pose
