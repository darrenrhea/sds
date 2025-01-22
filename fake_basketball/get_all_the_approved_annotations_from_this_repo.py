from get_annotation_id_clip_id_and_frame_index_from_mask_file_name import (
     get_annotation_id_clip_id_and_frame_index_from_mask_file_name
)
from typing import Dict, List
from maybe_find_sister_original_path_of_this_mask_path import (
     maybe_find_sister_original_path_of_this_mask_path
)

from pathlib import Path
import subprocess
import better_json as bj


def get_all_the_approved_annotations_from_this_repo(
    repo_dir: Path,
    pull_again: bool = False,
) -> List[Dict]:
    """
    This function will return the segmentation annotations that 
    are approved in the given repository.  Returns a list of approved annotations.
    An annotation is a dictionary with these keys:
    original_file_path: Path
    mask_file_path: Path
    """
    assert repo_dir.is_dir(), f"ERROR: {repo_dir} is not even an extant directory"
    if pull_again:
        args = [
            "git",
            "pull",
            "--ff-only",
        ]
        print(f"Running {' '.join(args)} in {repo_dir}")
        subprocess.run(
            args=args,
            cwd=repo_dir,
        )

    approvals_file_path = repo_dir / "approvals.json5"
    assert (
        approvals_file_path.is_file()
    ), f"ERROR: {approvals_file_path} is not a file"

    jsonable = bj.load(approvals_file_path)

    assert "approved" in jsonable, f"ERROR: {approvals_file_path=} does not have the key approved"
    approved_rel_paths = jsonable["approved"]
    assert isinstance(approved_rel_paths, list), f"ERROR: {approved_rel_paths=} is not a list"
    for approved_rel_path in approved_rel_paths:
        assert isinstance(approved_rel_path, str), f"ERROR: {approved_rel_path=} is not a string"
    
    mask_paths = [
        repo_dir / approved_rel_path
        for approved_rel_path in approved_rel_paths
    ]
    
    annotators = [
        approved_rel_path.split("/")[0]
        for approved_rel_path in approved_rel_paths
    ]

    for mask_path in mask_paths:
        assert mask_path.is_file(), f"ERROR: {mask_path} is not a file"
        assert mask_path.name.endswith("_nonfloor.png"), f"ERROR: {mask_path=} does not end with _nonfloor.png"

    original_paths = []
    for mask_path in mask_paths:
        original_path = maybe_find_sister_original_path_of_this_mask_path(
            mask_path=mask_path
        )
        assert (
            original_path is not None
        ), f"ERROR: original_path in None corresponding to {mask_path=}"
        original_paths.append(original_path)
    
    approved_annotations = []
    for original_path, mask_path, annotator in zip(original_paths, mask_paths, annotators):
        (
            annotation_id,
            clip_id,
            frame_index
        ) = get_annotation_id_clip_id_and_frame_index_from_mask_file_name(
            mask_file_name=mask_path.name
        )
        
        dct = dict(
            original_file_path=original_path,
            mask_file_path=mask_path,
            annotation_id=annotation_id,
            annotator=annotator,
            clip_id=clip_id,
            frame_index=frame_index
        )

        approved_annotations.append(dct)
    
    return approved_annotations





