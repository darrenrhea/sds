from get_all_annotations_in_a_single_folder import (
     get_all_annotations_in_a_single_folder
)
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


def get_all_annotations_in_these_folders(
    dataset_folders: List[Path],
    diminish_to_this_many: Optional[int]
) -> List[Tuple[Path, Path, Optional[Path]]]:
    """
    This takes in a list of folders, e.g.
    for the usual use case of one folder per court.

    Assumes that each directory at its top-level has
    either pairs of original image file and mask file
    or triplets of original image file, mask file, and relevance file,
    returns a python list of triplets of file paths.
    
    "datapoint_path_tuples".

    If there is no relevance mask, the third element of the triplet is None.
    """

   
    for dataset_folder in dataset_folders:
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
    for dataset_folder in dataset_folders:

        summand = get_all_annotations_in_a_single_folder(
            dataset_folder=dataset_folder,
            diminish_to_this_many=None
        )
        datapoint_path_tuples += summand
    
    # we want to shuffle regardless:
    np.random.shuffle(datapoint_path_tuples)
    
    if diminish_to_this_many is not None and len(datapoint_path_tuples) > diminish_to_this_many:
        datapoint_path_tuples = datapoint_path_tuples[:diminish_to_this_many]
    
    return datapoint_path_tuples

  