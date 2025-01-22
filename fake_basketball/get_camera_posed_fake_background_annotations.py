from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from is_valid_clip_id import (
     is_valid_clip_id
)
from pathlib import Path
from typing import List, Dict


def get_camera_posed_fake_background_annotations(
    subdir_name: str,
    shard_id: int,
) -> List[Dict[str, Path]]:
    """
    As long as you have backgrounds with camera poses and masks,
    you can paste players and balls and referees and coaches onto them
    to make fake annotations.

    This aims to give you all that you need to paste players on to.
    Returns a list of dictionaries with keys:
    actual_annotation_original_file_path,
    actual_annotation_mask_file_path,
    clip_id,
    frame_index,
    camera_pose.

    Given the subdir_name, we will look for the background annotations
    in shared_dir / subdir_name.
    
    An annotation is a pair of files:
    
        1. the original image and ends in "_original.png",
        2. the mask and ends in "_nonfloor.png".
    
    
    WARNING: these days we start with fake background images,
    not actual background images, see get_fake_background_annotations.py
    
    To make fake annotations, 
    we need to know where some real / actual annotations are,
    See the test.
    """
    
    # dotflat will make an .approved directory:
    shared_dir = get_the_large_capacity_shared_directory()
    actual_dir = shared_dir / subdir_name
    print(f"actual_dir is {actual_dir}")
    assert actual_dir.exists(), f"ERROR: {actual_dir=} does not exist"
    mask_paths = [x for x in actual_dir.glob("*_nonfloor.png")]
    
    actual_annotations = []
    for p in mask_paths:
        annotation_id = p.name[:-len("_original.png")]
        annotation_id_for_clip_id_and_frame_index = p.name[:-len("_fake565285006612152_original.png")]
        k = len(annotation_id_for_clip_id_and_frame_index)
        random_digit = p.name[k+5:k+6]
        assert random_digit in "0123456789", f"ERROR: {random_digit=} is not a digit"
        if shard_id != int(random_digit):
            continue
        sixdigits = annotation_id_for_clip_id_and_frame_index[-6:]
        for digit in sixdigits:
            assert digit in "0123456789", f"ERROR: {sixdigits=} is not all digits!"
        assert annotation_id_for_clip_id_and_frame_index[-7] == "_", f"ERROR: {annotation_id=} not parseable"
        clip_id = annotation_id_for_clip_id_and_frame_index[:-7]
        assert is_valid_clip_id(clip_id=clip_id), f"ERROR: invalid {clip_id=}"
        assert annotation_id_for_clip_id_and_frame_index == f"{clip_id}_{sixdigits}", f"ERROR: {clip_id=} and {sixdigits=} dont recreate {annotation_id_for_clip_id_and_frame_index=}"
        
        frame_index = int(sixdigits)

        camera_pose = get_camera_pose_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index,
        )

        actual_annotation_original_file_path = p.parent / (annotation_id + "_original.png")
        actual_annotation_mask_file_path = p.parent / (annotation_id + "_nonfloor.png")
        assert actual_annotation_original_file_path.exists(), f"ERROR: {actual_annotation_original_file_path=} does not exist"
        assert actual_annotation_mask_file_path.exists(), f"ERROR: {actual_annotation_mask_file_path=} does not exist"
        assert is_valid_clip_id(clip_id=clip_id), f"ERROR: invalid {clip_id=}"
        actual_annotations.append(
            dict(
                original_file_path=actual_annotation_original_file_path,
                mask_file_path=actual_annotation_mask_file_path,
                camera_pose=camera_pose,
                clip_id=clip_id,
                frame_index=frame_index,
            )
        )
    return actual_annotations


if __name__ == "__main__":

    annotations = get_camera_posed_fake_background_annotations(
        subdir_name="fake_backgrounds",
        shard_id=0
    )
    print(f"{len(annotations)=}")