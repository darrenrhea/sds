from pathlib import Path
import pprint as pp
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def gpksafasf_get_primary_keyed_segmentation_annotations_from_a_single_folder(
    dataset_folder: Path,
    diminish_to_this_many: Optional[int]
) -> List[Dict[str, Any]]:
    """
    There is a need to still know the frame_index and clip_id of each segmentation annotation.

    Even in a lot of the fake segmentation data generation,
    you still meaningfully have a frame_index and clip_id for tracking purposes.

    Given a directory which at its top-level has
    either pairs of original image file and mask file
    or triplets of original image file, mask file, and relevance file,
    returns a python list of triplets of file paths
    
    "annotations".

    If there is no relevance mask, the third element of the triplet is None.

    TODO: somebody make this take in a list of folders
    for the usual use case of one folder per court.
    """
    assert (
        dataset_folder.is_dir()
    ), f"dataset_folder must be a directory, but {dataset_folder=} is not a directory"

    assert (
        diminish_to_this_many is None
        or (
        isinstance(diminish_to_this_many, int)
        and
        diminish_to_this_many > 0
        )
    ), f"diminish_to_this_many must be None or a positive integer, but {diminish_to_this_many=} is neither"

    original_paths1 = [
        original_path
        for original_path in dataset_folder.glob("*_original.jpg")
    ]
    original_paths2 = [
        original_path
        for original_path in dataset_folder.glob("*_original.png")
    ]
    original_paths3 = [
        original_path
        for original_path in dataset_folder.glob("*.jpg")
    ]
    original_paths = original_paths1 + original_paths2 + original_paths3

    pp.pprint(original_paths)

    annotations = []  # accumulator
    for original_path in original_paths:
        if str(original_path).endswith("original.jpg") or str(original_path).endswith("original.png"):
            annotation_id = original_path.stem[:-9]  # remove _original
        else:
            annotation_id = original_path.stem
        print(annotation_id)
        frame_index_str = annotation_id.split("_")[-1]
        
        frame_index = int(frame_index_str)
        clip_id = annotation_id[:-len(frame_index_str) - 1]
        
        assert clip_id == "brewcub", f"clip_id {clip_id} is not brewcub"

        mask_path = original_path.parent / (annotation_id + "_nonfloor.png")
        if not mask_path.exists():
            raise Exception(f"mask_path {mask_path} does not exist")
        relevance_path = original_path.parent / (annotation_id + "_relevance.png")
        if not relevance_path.exists():
            maybe_relevance_path = None
        else:
            maybe_relevance_path = relevance_path
        
        annotation = dict(
            clip_id=clip_id,
            frame_index=frame_index,
            annotation_id=annotation_id,
            original_path=original_path,
            mask_path=mask_path,
            maybe_relevance_path=maybe_relevance_path
        )

        annotations.append(
            annotation
        )
    
    if diminish_to_this_many is not None:
        np.random.shuffle(annotations)
        annotations = annotations[:diminish_to_this_many]
    
    annotations =sorted(
        annotations,
        key=lambda x: x["frame_index"]
    )
    return annotations

  