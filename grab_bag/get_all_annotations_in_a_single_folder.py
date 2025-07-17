from pathlib import Path
import pprint as pp
from typing import List, Optional, Tuple
import numpy as np


def get_all_annotations_in_a_single_folder(
    dataset_folder: Path,
    diminish_to_this_many: Optional[int]
) -> List[Tuple[Path, Path, Optional[Path]]]:
    """
    Given a directory which at its top-level has
    either pairs of original image file and mask file
    or triplets of original image file, mask file, and relevance file,
    returns a python list of triplets of file paths
    
    "datapoint_path_tuples".

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

    datapoint_path_tuples = []
    # This is covered by *.jpg already:
    # original_paths1 = [
    #     original_path
    #     for original_path in dataset_folder.glob("*_original.jpg")
    # ]
    original_paths2 = [
        original_path
        for original_path in dataset_folder.glob("*_original.png")
    ]
    original_paths3 = [
        original_path
        for original_path in dataset_folder.glob("*.jpg")
    ]
    original_paths = original_paths2 + original_paths3

    pp.pprint(original_paths)

    for original_path in original_paths:
        if str(original_path).endswith("original.jpg") or str(original_path).endswith("original.png"):
            annotation_id = original_path.stem[:-9]  # remove _original
        else:
            annotation_id = original_path.stem
        print(annotation_id)
        mask_path = original_path.parent / (annotation_id + "_nonfloor.png")
        if not mask_path.exists():
            raise Exception(f"mask_path {mask_path} does not exist")
        relevance_path = original_path.parent / (annotation_id + "_relevance.png")
        if not relevance_path.exists():
            maybe_relevance_path = None
        else:
            maybe_relevance_path = relevance_path
        datapoint_path_tuples.append(
            (original_path, mask_path, maybe_relevance_path)
        )
    
    if diminish_to_this_many is not None:
        np.random.shuffle(datapoint_path_tuples)
        datapoint_path_tuples = datapoint_path_tuples[:diminish_to_this_many]
    
    return datapoint_path_tuples

  