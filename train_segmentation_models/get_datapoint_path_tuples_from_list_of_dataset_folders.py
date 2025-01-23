from pathlib import Path
from typing import List, Tuple, Optional
from colorama import Fore, Style


def convert_original_path_to_target_mask_path(original_path: Path):
    assert isinstance(original_path, Path)
    assert original_path.is_file()
    assert original_path.is_absolute()
    assert original_path.name.endswith('.jpg')

    target_mask_path = original_path.parent / (original_path.name.replace('.jpg', '_nonfloor.png'))
    assert target_mask_path.is_file(), f"ERROR: {target_mask_path} does not exist despite that {original_path} does"
    return target_mask_path


def convert_original_path_to_weight_mask_path(original_path: Path):
    assert isinstance(original_path, Path)
    assert original_path.is_file()
    assert original_path.is_absolute()
    assert original_path.name.endswith('.jpg')

    weight_mask_path = original_path.parent / (original_path.name.replace('.jpg', '_relevance.png'))
    return weight_mask_path


def get_datapoint_path_tuples_from_list_of_dataset_folders(
    list_of_dataset_folders: List[Path],
    dataset_kind: str,
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
    given a VERY SPECIFIC directory structure,
    and VERY SPECIFIC structure of the filenames within those directories.

    If you have a different directory structure or filenames you will need to write a different function that returns

    List[
        Tuple[
            Path,  # original_path,
            Path,  # target_mask_path,
            Optional[Path]  # weight_mask_path / relevance_mask_path, which is optional
        ]
    ]

    Used to be called
    get_datapoint_path_tuples_from_list_of_dataset_folders

    Every training datapoint may consist of a large number of image files.
    Right now 2 or 3 files: an original image file, a target_mask file,
    and maybe a weight_mask file to make the loss function
    only care about relevant regions of the image.

    But later more complicated things like multiple target_masks for multiple segmentation conventions,
    multiple weight_masks to match the multiple target_masks, etc.

    Who knows what we might use as a dataset in the future, but
    right now a dataset if a finite list of directories that contain frames and masks of the same name.
    This function is meant to be a single point of entry for all datasets.
    """

    assert isinstance(list_of_dataset_folders, list)
    for d in list_of_dataset_folders:
        assert isinstance(d, Path)
        assert d.is_dir(), "ERROR: {d} is not an extant directory"

    assert dataset_kind in ['nonfloor', 'nonwood']
    all_path_tuples = []

    print("Gathering all training points from these directories:")
    for folder in list_of_dataset_folders:
        print(f"    {folder}")

    for folder in list_of_dataset_folders:
        print(f'loading files from {folder}')
        original_paths = [p for p in Path(folder).rglob(f'*.jpg')]
        target_mask_paths = [convert_original_path_to_target_mask_path(p) for p in original_paths]
        maybe_weight_mask_paths = [convert_original_path_to_weight_mask_path(p) for p in original_paths]
        weight_mask_paths = [p if p.is_file() else None for p in maybe_weight_mask_paths]
        some_path_tuples = list(zip(original_paths, target_mask_paths, weight_mask_paths))
 
        all_path_tuples.extend(some_path_tuples)

    for path_tuple in all_path_tuples:
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
      
    return all_path_tuples


