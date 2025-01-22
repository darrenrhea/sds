from maybe_find_sister_original_path_of_this_mask_path import (
     maybe_find_sister_original_path_of_this_mask_path
)
from get_annotation_id_clip_id_and_frame_index_from_mask_file_name import (
     get_annotation_id_clip_id_and_frame_index_from_mask_file_name
)
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from is_valid_clip_id import (
     is_valid_clip_id
)
from pathlib import Path
from typing import List, Dict, Tuple


def get_camera_posed_human_annotations_from_list_of_paths(
    list_of_original_and_mask_paths: List[Tuple[Path, Path]]
) -> List[Dict[str, Path]]:
    """
    An annotation is a pair of files,
    so give use a list of such pairs.

    This aims to give you all that you need to paste players on to.
    Returns a list of dictionaries with keys:
    actual_annotation_original_file_path,
    actual_annotation_mask_file_path,
    clip_id,
    frame_index,
    camera_pose.
    """
    assert isinstance(list_of_original_and_mask_paths, list), f"ERROR: {list_of_original_and_mask_paths=} is not a list"
    
    assert len(list_of_original_and_mask_paths) > 0, f"ERROR: {list_of_original_and_mask_paths=} is empty"

    for tupl in list_of_original_and_mask_paths:
        assert isinstance(tupl, tuple), f"ERROR: {tupl=} is not a tuple"
        assert len(tupl) == 2, f"ERROR: {tupl=} does not have length 2"
        original_file_path, mask_file_path = tupl
        assert isinstance(original_file_path, Path), f"ERROR: {original_file_path=} is not a Path"
        assert isinstance(mask_file_path, Path), f"ERROR: {mask_file_path=} is not a Path"
        assert original_file_path.exists(), f"ERROR: {original_file_path=} does not exist"
        assert mask_file_path.exists(), f"ERROR: {mask_file_path=} does not exist"
        assert original_file_path.is_absolute(), f"ERROR: {original_file_path=} is not absolute"
        assert mask_file_path.is_absolute(), f"ERROR: {mask_file_path=} is not absolute"
    
    actual_annotations = []
    for original_file_path, mask_file_path in list_of_original_and_mask_paths:
        (
            annotation_id,
            clip_id,
            frame_index
        ) = get_annotation_id_clip_id_and_frame_index_from_mask_file_name(
            mask_file_name=mask_file_path.name
        )


        camera_pose = get_camera_pose_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index,
        )

        assert (
            original_file_path.exists()
        ), f"ERROR: {original_file_path=} does not exist"

        assert (
            mask_file_path.exists()
        ), f"ERROR: {mask_file_path=} does not exist"

        assert is_valid_clip_id(clip_id=clip_id), f"ERROR: invalid {clip_id=}"
        
        actual_annotations.append(
            dict(
                original_file_path=original_file_path,
                mask_file_path=mask_file_path,
                camera_pose=camera_pose,
                clip_id=clip_id,
                frame_index=frame_index,
            )
        )
    return actual_annotations


if __name__ == "__main__":
    repo_dir = Path(
        "~/r/bay-zal-2024-03-15-mxf-yadif_led"
    ).expanduser()

    mask_paths = [
        repo_dir / "anna/bay-zal-2024-03-15-mxf-yadif_133917_nonfloor.png",
    ]

    list_of_original_and_mask_paths = []
    for mask_path in mask_paths:
        print(mask_path)
        original_path = maybe_find_sister_original_path_of_this_mask_path(
            mask_path=mask_path
        )
        pair = (original_path, mask_path)
        list_of_original_and_mask_paths.append(pair)

    

    get_camera_posed_human_annotations_from_list_of_paths(
        list_of_original_and_mask_paths=list_of_original_and_mask_paths
    )
    annotations = get_camera_posed_human_annotations_from_list_of_paths(
        list_of_original_and_mask_paths=list_of_original_and_mask_paths
    )
    print(f"{len(annotations)=}")