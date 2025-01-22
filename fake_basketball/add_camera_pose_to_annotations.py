from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from pathlib import Path
from typing import List, Dict


def add_camera_pose_to_annotations(
    annotations: List[Dict[str, Path]]
) -> None:
    for annotation in annotations:
        clip_id = annotation["clip_id"]
        frame_index = annotation["frame_index"]
        camera_pose = get_camera_pose_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index,
        )
        annotation["camera_pose"] = camera_pose
