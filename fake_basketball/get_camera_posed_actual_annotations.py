from get_approved_annotations_from_these_repos import (
     get_approved_annotations_from_these_repos
)
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from pathlib import Path
from typing import List, Dict


def get_camera_posed_actual_annotations(
    repo_ids_to_use: List[str],
    pull_again: bool = False
) -> List[Dict[str, Path]]:
    """
    see segmentation_data_utils/get_local_file_pathed_annotations_test.py
    for a different way that does not need the repos staged
    
    WARNING: these days we start with fake background images,
    not actual background images, see get_fake_background_images.py
    
    To make fake annotations, 
    we need to know where some real / actual annotations are,
    See the test.
    """
    
    approved_annotations = get_approved_annotations_from_these_repos(
        repo_ids_to_use=repo_ids_to_use,
        pull_again=pull_again,
    )

    for approved_annotation in approved_annotations:
        clip_id = approved_annotation["clip_id"]
        frame_index = approved_annotation["frame_index"]
        camera_pose = get_camera_pose_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index,
        )
        approved_annotation["camera_pose"] = camera_pose
    
    return approved_annotations
    