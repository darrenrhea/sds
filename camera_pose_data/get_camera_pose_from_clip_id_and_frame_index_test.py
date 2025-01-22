from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from color_print_json import (
     color_print_json
)
from CameraParameters import CameraParameters

def test_get_camera_pose_from_clip_id_and_frame_index_1():
    clip_id = "bos-mia-2024-04-21-mxf"
    frame_index = 734500

    camera_pose = get_camera_pose_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index
    )
    assert isinstance(camera_pose, CameraParameters)
    color_print_json(camera_pose.to_dict())
    

if __name__ == '__main__':
    test_get_camera_pose_from_clip_id_and_frame_index_1()