from pathlib import Path
from get_datapoint_path_tuples_from_list_of_dataset_folders import get_datapoint_path_tuples_from_list_of_dataset_folders 


def test_get_datapoint_path_tuples_from_list_of_dataset_folders():
    folder = Path("~/alpha_mattes_temp/").expanduser()
    list_of_dataset_folders = [folder,]

    path_tuples = get_datapoint_path_tuples_from_list_of_dataset_folders(
        list_of_dataset_folders=list_of_dataset_folders,
        dataset_kind="nonfloor"
    )

    for path_tuple in path_tuples:
        assert isinstance(path_tuple, tuple), f"ERROR: {path_tuple} is not a tuple"
        assert len(path_tuple) == 3, f"ERROR: {path_tuple} does not have 3 elements"
        for path in path_tuple:
            assert isinstance(path, Path), f"ERROR: {path} is not a Path"
            assert path.is_file()
    
    print("PASSED")

if __name__ == "__main__":
   test_get_datapoint_path_tuples_from_list_of_dataset_folders()