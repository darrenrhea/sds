from print_yellow import (
     print_yellow
)
from print_green import (
     print_green
)
from get_datapoint_path_tuples_from_list_of_dataset_folders import (
     get_datapoint_path_tuples_from_list_of_dataset_folders
)
from pathlib import Path



def test_get_datapoint_path_tuples_from_list_of_dataset_folders_1():
    """
    Make this grab a test fixture rather than prestaged stuff.
    """
    dataset_folders = [
        Path("~/r/nfl-59773-skycam-ddv3_floor/.approved").expanduser(),
    ]

    datapoint_path_tuples = get_datapoint_path_tuples_from_list_of_dataset_folders(
        dataset_folders=dataset_folders
    )

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
   
    for t in datapoint_path_tuples:
        original, mask, relevance = t
        print_green(f"{original=!s}")
        print_green(f"{mask=!s}")
        print_green(f"{relevance=!s}\n")

    num_training_points = len(datapoint_path_tuples)
    print_yellow(f"{num_training_points=}")


if __name__ == "__main__":
    test_get_datapoint_path_tuples_from_list_of_dataset_folders_1()
    print("get_datapoint_path_tuples_from_list_of_dataset_folders.py: all tests pass")