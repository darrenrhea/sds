from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from is_valid_clip_id import (
     is_valid_clip_id
)
from pathlib import Path
from typing import List, Dict

import better_json as bj

from CameraParameters import CameraParameters


def get_camera_posed_people_free_backgrounds(
    subdir_name: str
) -> List[Dict[str, Path]]:
    """
    As long as you have a people and ball free background with camera poses,
    you can paste players and balls and referees and coaches onto them
    to make fake important people annotations.

    This aims to give you all that you need to paste players on to.
    Returns a list of dictionaries with keys:
    actual_annotation_original_file_path,
    clip_id,
    frame_index,
    camera_pose.

    Given the subdir_name, we will look for the background annotations
    in shared_dir / subdir_name.
    
    An people free background annotation is a single file:
    
        1. the original image and ends in "_original.png",
    
    """
    
    # dotflat will make an .approved directory:
    shared_dir = get_the_large_capacity_shared_directory()
    actual_dir = shared_dir / subdir_name
    print(f"actual_dir is {actual_dir}")
    assert actual_dir.exists(), f"ERROR: {actual_dir=} does not exist"
    original_paths = [x for x in actual_dir.glob("*_original.png")]
    
    actual_annotations = []
    for original_path in original_paths:
        annotation_id = original_path.name[:-len("_original.png")]
        annotation_id_for_clip_id_and_frame_index = original_path.name[:-len("_fake565285006612152_original.png")]
        sixdigits = annotation_id_for_clip_id_and_frame_index[-6:]
        for digit in sixdigits:
            assert digit in "0123456789", f"ERROR: {sixdigits=} is not all digits!"
        assert annotation_id_for_clip_id_and_frame_index[-7] == "_", f"ERROR: {annotation_id=} not parseable"
        clip_id = annotation_id_for_clip_id_and_frame_index[:-7]
        assert is_valid_clip_id(clip_id=clip_id), f"ERROR: invalid {clip_id=}"
        assert annotation_id_for_clip_id_and_frame_index == f"{clip_id}_{sixdigits}", f"ERROR: {clip_id=} and {sixdigits=} dont recreate {annotation_id_for_clip_id_and_frame_index=}"
        
        frame_index = int(sixdigits)

     
    


        actual_annotation_original_file_path = original_path.parent / (annotation_id + "_original.png")
        camera_pose_file_path = original_path.parent / (annotation_id + "_camera_pose.json")

        jsonable = bj.load(camera_pose_file_path)
        
        camera_pose = CameraParameters.from_dict(jsonable)


        assert actual_annotation_original_file_path.exists(), f"ERROR: {actual_annotation_original_file_path=} does not exist"
        assert is_valid_clip_id(clip_id=clip_id), f"ERROR: invalid {clip_id=}"
        actual_annotations.append(
            dict(
                original_file_path=actual_annotation_original_file_path,
                camera_pose=camera_pose,
                clip_id=clip_id,
                frame_index=frame_index,
            )
        )
    return actual_annotations

