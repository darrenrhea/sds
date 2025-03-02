from save_jsonable import (
     save_jsonable
)
from get_a_temp_file_path import (
     get_a_temp_file_path
)
from store_file_by_sha256 import (
     store_file_by_sha256
)
from get_datapoint_path_tuples_from_list_of_dataset_folders import (
     get_datapoint_path_tuples_from_list_of_dataset_folders
)
from print_yellow import (
     print_yellow
)
from print_green import (
     print_green
)
from pathlib import Path


def stsditfajos_store_the_segmentation_datapoints_in_these_folders_as_json_of_sha256s(
    dataset_folders
) -> str:
    
    datapoint_path_tuples = get_datapoint_path_tuples_from_list_of_dataset_folders(
        dataset_folders=dataset_folders
    )

    records = []
    for triple in datapoint_path_tuples:
        original, mask, relevance = triple
        print_green(f"{original=!s}")
        print_green(f"{mask=!s}")
        print_green(f"{relevance=!s}\n")
        
        original_sha256 = store_file_by_sha256(original)
        mask_sha256 = store_file_by_sha256(mask)
        if relevance is not None:
            relevance_sha256 = store_file_by_sha256(relevance)
        else:
            relevance_sha256 = None

        record = dict(
            original_sha256=original_sha256,
            mask_sha256=mask_sha256,
            relevance_sha256=relevance_sha256
        )
        records.append(record)
    
    temp_file_path = get_a_temp_file_path(suffix=".json")
    save_jsonable(fp=temp_file_path, obj=records)

    sha256 = store_file_by_sha256(temp_file_path)

    return sha256


if __name__ == "__main__":
    dataset_folders = [
        Path("~/r/nfl-59773-skycam-ddv3_floor/.approved").expanduser(),
    ]

    sha256 = stsditfajos_store_the_segmentation_datapoints_in_these_folders_as_json_of_sha256s(  
        dataset_folders=dataset_folders
    )

    print_yellow(f"The {sha256=}")