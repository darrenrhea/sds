from clip_id_to_league import (
     clip_id_to_league
)

from get_enough_landmarks_to_validate_camera_pose import (
     get_enough_landmarks_to_validate_camera_pose
)
from draw_named_3d_points import (
     draw_named_3d_points
)
from get_video_frame_path_from_clip_id_and_frame_index import (
     get_video_frame_path_from_clip_id_and_frame_index
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)


def validate_one_camera_pose(
    clip_id: str,
    frame_index: int,
):
    """
    See also vovfa_validate_one_video_frame_annotation.py
    This function will validate one camera pose.
    See also the for-loopification of this, validate_camera_poses
    """

    assert isinstance(clip_id, str)
    assert isinstance(frame_index, int), f"frame_index is {frame_index}, which is not an int, but is of type {type(frame_index)}."

    league = clip_id_to_league(clip_id=clip_id)

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

    landmark_name_to_xyz = get_enough_landmarks_to_validate_camera_pose(
        league=league
    )
    drawn_on = draw_named_3d_points(
        original_rgb_np_u8=original_rgb_np_u8,
        camera_pose=camera_pose,
        landmark_name_to_xyz=landmark_name_to_xyz
    )

    return drawn_on

    

