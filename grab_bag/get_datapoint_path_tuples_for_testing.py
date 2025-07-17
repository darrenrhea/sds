from pathlib import Path
from typing import List, Optional, Tuple
from get_all_annotations_in_these_folders import (
     get_all_annotations_in_these_folders
)
from colorama import Fore, Style


def get_datapoint_path_tuples_for_testing() -> List[Tuple[Path, Path, Optional[Path]]]:
    """
    For testing stuff, we don't care what the training set is.
    Just a bunch of flatled convention segmented images in from of State Farm.
    """
    dataset_folder_strs = [
        "/shared/flattened_fake_game5/state_farm_state_farm",
    ]

    dataset_folders = [
        Path(dataset_folder_str).expanduser().resolve()
        for dataset_folder_str in dataset_folder_strs
    ]

    datapoint_path_tuples = get_all_annotations_in_these_folders(
        dataset_folders=dataset_folders,
        diminish_to_this_many=10000,
    )

    num_training_points = len(datapoint_path_tuples)
    print(f"{Fore.YELLOW}{num_training_points=}{Style.RESET_ALL}")
    return datapoint_path_tuples

