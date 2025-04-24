import textwrap
from colorama import Fore, Style
from maybe_find_sister_original_path_of_this_mask_path import (
     maybe_find_sister_original_path_of_this_mask_path
)
from pathlib import Path
from typing import List, Tuple, Optional


def get_datapoint_path_tuples_from_list_of_dataset_folders(
    dataset_folders: List[Path],
) -> List[
        Tuple[
            Path,  # absolute original_path,
            Path,  # absolute target_mask_path,
            Optional[Path]  # absolute weight_mask_path / relevance_mask_path or None, i.e. it is optional
        ]
    ]:
    """
    
    One way to kind-of specify a training dataset is as a list of directories
    that contain the training data.  This makes that happen
    given a **VERY SPECIFIC** directory structure,
    and VERY SPECIFIC structure of the filenames within those directories.

    If you have a different directory structure or filenames you will need to write a different function that returns

    .. code-block:: python

        List[
            Tuple[
                Path,  # original_path,
                Path,  # target_mask_path,
                Optional[Path]  # weight_mask_path / relevance_mask_path, which is optional
            ]
        ]

    Every training datapoint may consist of a large number of image files.
    Right now 2 or 3 files: an original image file, a target_mask file,
    and maybe a weight_mask file to make the loss function
    only care about relevant regions of the image.

    But later more complicated things, like multiple target_masks for multiple segmentation conventions,
    multiple weight_masks to match the multiple target_masks, etc.

    Who knows what we might use as a dataset in the future, but
    right now a dataset is a finite list of directories that contain frames and masks of the same name.
    This function is meant to be a single point of entry for all datasets.
    """

    assert isinstance(dataset_folders, list)
    for dataset_folder in dataset_folders:
        assert isinstance(dataset_folder, Path), f"ERROR: {dataset_folder} is not a Path"
        assert dataset_folder.is_dir(), f"dataset_folder must be a directory, but {dataset_folder=} is not a directory"
 

    print("Gathering all training points from these directories recursively:")
    for folder in dataset_folders:
        print(f"    {folder}")

 
    datapoint_path_tuples = []
    for dataset_folder in dataset_folders:
        for mask_path in dataset_folder.rglob("*_nonfloor.png"):
            original_path = maybe_find_sister_original_path_of_this_mask_path(
                mask_path=mask_path
            )
            if original_path is None:
                print(
                    textwrap.dedent(
                        f"""\
                        {Fore.RED}
                        ERROR: {mask_path=} does not seem to have a sister original path.
                        Skipping this mask.
                        {Style.RESET_ALL}
                        """
                    )
                )
                continue
            relevance_path = original_path.parent / (mask_path.stem[:-9] + "_relevance.png")
            if not relevance_path.exists():
                maybe_relevance_path = None
            else:
                maybe_relevance_path = relevance_path
                print(
                    textwrap.dedent(
                        f"""\
                        {Fore.YELLOW}
                        WARNING: {relevance_path=} exists.  Very unusual for floor_not_floor right now.
                        {Style.RESET_ALL}
                        """
                    )
                )
            
            datapoint_path_tuples.append(
                (original_path, mask_path, maybe_relevance_path)
            )

    # BEGIN check sanity of datapoint_path_tuples before returning it:
    for path_tuple in datapoint_path_tuples:
        assert len(path_tuple) == 3, f"ERROR: {path_tuple} does not have 3 elements"
        for path in path_tuple[:2]:
            assert isinstance(path, Path), f"ERROR: {path} is not a Path"
            assert path.is_file()
        weight_mask_path = path_tuple[2]
        assert (
            weight_mask_path is None
            or
            (
                isinstance(weight_mask_path, Path)
                and
                weight_mask_path.is_file()
            )
        ), f"ERROR: {weight_mask_path} is not an extant Path, nor is is None"
      
    return datapoint_path_tuples

   

