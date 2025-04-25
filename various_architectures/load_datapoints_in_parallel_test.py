from load_datapoints_in_parallel import (
     load_datapoints_in_parallel
)
from pathlib import Path


def test_load_datapoints_in_parallel_1():
    """
    This is a test function to test the load_datapoints_in_parallel function.
    It loads a few datapoints in parallel and checks that they are loaded correctly.
    """
    from get_datapoint_path_tuples_from_list_of_dataset_folders import (
        get_datapoint_path_tuples_from_list_of_dataset_folders
    )

    dataset_folders = [
        # Path("~/r/bal_game2_bigzoom_floor_10bit").expanduser(),
        Path("~/r/bal_game2_bigzoom_floor/.approved").expanduser(),
    ]

    datapoint_path_tuples = get_datapoint_path_tuples_from_list_of_dataset_folders(
        dataset_folders=dataset_folders,
    )

    channel_stacks = load_datapoints_in_parallel(
        datapoint_path_tuples=datapoint_path_tuples,
        preprocessor=None,
        preprocessor_params=None
    )

    print(f"{type(channel_stacks)=}")


if __name__ == "__main__":
    test_load_datapoints_in_parallel_1()