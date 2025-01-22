from make_map_from_clip_id_and_frame_index_to_camera_pose import (
     make_map_from_clip_id_and_frame_index_to_camera_pose
)
from typing import Dict, Tuple
from CameraParameters import CameraParameters


def make_map_from_annotation_id_to_camera_pose(
    video_frame_annotations_metadata_sha256
) -> Dict[Tuple[str, int], CameraParameters]:
    """
    Chaz has put camera-poses in json files next to the image files,
    but we prefer to look up the camera-poses from the clip_id and frame_index.
    """
    map_from_clip_id_and_frame_index_to_camera_pose = (
        make_map_from_clip_id_and_frame_index_to_camera_pose(
            video_frame_annotations_metadata_sha256=video_frame_annotations_metadata_sha256
        )
    )
    
    map_from_annotation_id_to_camera_pose = dict()
    for key, value in map_from_clip_id_and_frame_index_to_camera_pose.items():
        clip_id, frame_index = key
        annotation_id = f"{clip_id}_{frame_index:06d}"
        
        camera_pose = map_from_clip_id_and_frame_index_to_camera_pose[
            (clip_id, frame_index)
        ]
        map_from_annotation_id_to_camera_pose[annotation_id] = camera_pose 


    return map_from_annotation_id_to_camera_pose


if __name__ == "__main__":
    video_frame_annotations_metadata_sha256 = "99bc2c688a6bd35f08b873495d062604e0b954244e6bb20f5c5a76826ac53524"
    map_from_annotation_id_to_camera_pose = (
        make_map_from_annotation_id_to_camera_pose(
            video_frame_annotations_metadata_sha256=video_frame_annotations_metadata_sha256
        )
    )
    print(f"{len(map_from_annotation_id_to_camera_pose)=}")