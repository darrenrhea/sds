from typing import Optional

import numpy as np

from CameraParameters import (
     CameraParameters
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from get_enough_landmarks_to_validate_camera_pose import (
     get_enough_landmarks_to_validate_camera_pose
)
from draw_named_3d_points import (
     draw_named_3d_points
)

from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)


def vovfa_validate_one_video_frame_annotation(
    annotation: dict,
    draw_a_certificate_image: bool = False,
) -> Optional[np.ndarray]:
    """
    To be well-organized, we need to have a wide variety of NBA basketball video frames where
    we know the floor_not_floor segmentation, the camera pose,
    which floor/court it is, which teams and what uniforms they are wearing, etc.
    If you set draw_a_certificate to True, then this function will draw a "certificate image"
    that would allow a human to determine if the data is corrupt:
    if the camera pose is wrong, things will not all the floor_not_floor segmentation is wrong, etc.
    """
    clip_id = annotation["clip_id"]
    assert isinstance(clip_id, str)

    frame_index = annotation["frame_index"]
    assert (
        isinstance(frame_index, int)
    ), f"frame_index is {frame_index}, which is not an int, but is of type {type(frame_index)}."

    camera_pose_jsonable = annotation["camera_pose"]

    camera_pose = CameraParameters(
        rod=camera_pose_jsonable["rod"],
        loc=camera_pose_jsonable["loc"],
        f=camera_pose_jsonable["f"],
        ppi=0.0,
        ppj=0.0,
        k1=camera_pose_jsonable["k1"],
        k2=camera_pose_jsonable["k2"],
    )


    # league = clip_id_to_league(clip_id=clip_id)
    league = annotation["clip_id_info"]["league"]
    assert league == "nba"

    original_file_path = get_file_path_of_sha256(
        sha256=annotation["label_name_to_sha256"]["original"]
    )

    original_rgb_np_u8 = open_as_rgb_hwc_np_u8(
        image_path=original_file_path
    )

    # camera_pose = get_camera_pose_from_clip_id_and_frame_index(
    #     clip_id=clip_id,
    #     frame_index=frame_index
    # ) 


    if not draw_a_certificate_image:
        return None

    landmark_name_to_xyz = get_enough_landmarks_to_validate_camera_pose(
        league=league
    )
    
    print(camera_pose)

    drawn_on = draw_named_3d_points(
        original_rgb_np_u8=original_rgb_np_u8,
        camera_pose=camera_pose,
        landmark_name_to_xyz=landmark_name_to_xyz
    )

    return drawn_on
