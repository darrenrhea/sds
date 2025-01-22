
from CameraParameters import (
     CameraParameters
)


def convert_camera_pose_to_jsonable(
    camera_pose: CameraParameters
) -> dict:
    """
    We want to not mention the ppi, ppj, k3, p1, p2 which are so often zero.
    Let the convention be that if they are zero, they are not mentioned.
    """
    assert isinstance(camera_pose, CameraParameters), f"ERROR: {type(camera_pose)=}"
    camera_pose_jsonable = camera_pose.to_jsonable()
    assert camera_pose_jsonable["ppi"] == 0.0
    assert camera_pose_jsonable["ppj"] == 0.0
    assert camera_pose_jsonable["k3"] == 0.0
    assert camera_pose_jsonable["p1"] == 0.0
    assert camera_pose_jsonable["p2"] == 0.0
    del camera_pose_jsonable["ppi"]
    del camera_pose_jsonable["ppj"]
    del camera_pose_jsonable["k3"]
    del camera_pose_jsonable["p1"]
    del camera_pose_jsonable["p2"]
    return camera_pose_jsonable

