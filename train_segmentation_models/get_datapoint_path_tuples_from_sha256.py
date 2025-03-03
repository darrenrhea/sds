from is_sha256 import (
     is_sha256
)
from download_the_files_with_these_sha256s import (
     download_the_files_with_these_sha256s
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from ltjfthts_load_the_jsonlike_file_that_has_this_sha256 import (
     ltjfthts_load_the_jsonlike_file_that_has_this_sha256
)
from print_yellow import (
     print_yellow
)
from pathlib import Path
from typing import Optional, Tuple, List


def get_datapoint_path_tuples_from_sha256(
    sha256: str
) -> List[Tuple[Path, Path, Optional[Path]]]:

    datapoints = ltjfthts_load_the_jsonlike_file_that_has_this_sha256(sha256)
    
    shas_to_download = set()
    
    for datapoint in datapoints:
        assert isinstance(datapoint, dict)
        assert "original_sha256" in datapoint
        assert "mask_sha256" in datapoint
        assert "relevance_sha256" in datapoint
        original_sha256 = datapoint["original_sha256"]
        mask_sha256 = datapoint["mask_sha256"]
        maybe_relevance_sha256 = datapoint["relevance_sha256"]
        assert is_sha256(original_sha256)
        assert is_sha256(mask_sha256)
        
        shas_to_download.add(original_sha256)
        shas_to_download.add(mask_sha256)

        if maybe_relevance_sha256 is not None:
            assert is_sha256(maybe_relevance_sha256)
            shas_to_download.add(maybe_relevance_sha256)

    download_the_files_with_these_sha256s(shas_to_download)

    # now that they are all downloaded, we can get the local paths:
    datapoint_path_tuples = []
    for datapoint in datapoints:
        original_sha256 = datapoint["original_sha256"]
        mask_sha256 = datapoint["mask_sha256"]
        maybe_relevance_sha256 = datapoint["relevance_sha256"]
        original_local_path = get_file_path_of_sha256(original_sha256)
        mask_local_path = get_file_path_of_sha256(mask_sha256)
        if maybe_relevance_sha256 is not None:
            maybe_relevance_local_path = get_file_path_of_sha256(maybe_relevance_sha256)
        else:
            maybe_relevance_local_path = None
        datapoint_path_tuples.append(
            (original_local_path, mask_local_path, maybe_relevance_local_path)
        )
    
    return datapoint_path_tuples
    
    
if __name__ == "__main__":
    datapoint_path_tuples = get_datapoint_path_tuples_from_sha256(
        sha256="c0d38c24dc78b4fc714279ae5c80ae6e8d5580898503b63fc7fde129fcdd0a35"
    )
    for t in datapoint_path_tuples:
        original, mask, relevance = t
        print(f"{original=!s}")
        print(f"{mask=!s}")
        print(f"{relevance=!s}")
        print()

    num_training_points = len(datapoint_path_tuples)
    print_yellow(f"{num_training_points=}")

       