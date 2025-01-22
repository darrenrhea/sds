from typing import Dict, Tuple
from make_map_from_clip_id_and_frame_index_to_video_frame_annotation import (
     make_map_from_clip_id_and_frame_index_to_video_frame_annotation
)
from get_camera_pose_from_sha256 import (
     get_camera_pose_from_sha256
)
from CameraParameters import CameraParameters


def make_map_from_clip_id_and_frame_index_to_camera_pose(
    video_frame_annotations_metadata_sha256
) -> Dict[Tuple[str, int], CameraParameters]:
    """
    Chaz has put camera-poses in json files next to the image files,
    but we prefer to look up the camera-poses from the clip_id and frame_index.
    """
    map_from_clip_id_and_frame_index_to_video_frame_annotation = (
        make_map_from_clip_id_and_frame_index_to_video_frame_annotation(
            video_frame_annotations_metadata_sha256=video_frame_annotations_metadata_sha256
        )
    )
    
    map_from_clip_id_and_frame_index_to_camera_pose = dict()
    for key, value in map_from_clip_id_and_frame_index_to_video_frame_annotation.items():
        clip_id, frame_index = key
        label_name_to_sha256 = value["label_name_to_sha256"]

        camera_pose_sha256 = label_name_to_sha256.get("camera_pose", None)

        if camera_pose_sha256 is not None:
            camera_pose = get_camera_pose_from_sha256(camera_pose_sha256)
        
            map_from_clip_id_and_frame_index_to_camera_pose[
                (clip_id, frame_index)
            ] = camera_pose

    return map_from_clip_id_and_frame_index_to_camera_pose


if __name__ == "__main__":
    video_frame_annotations_metadata_sha256 = "99bc2c688a6bd35f08b873495d062604e0b954244e6bb20f5c5a76826ac53524"
    map_from_clip_id_and_frame_index_to_camera_pose = (
        make_map_from_clip_id_and_frame_index_to_camera_pose(
            video_frame_annotations_metadata_sha256=video_frame_annotations_metadata_sha256
        )
    )
    print(f"{len(map_from_clip_id_and_frame_index_to_camera_pose)=}")