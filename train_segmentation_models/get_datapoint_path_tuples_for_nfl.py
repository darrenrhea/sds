from print_yellow import (
     print_yellow
)
from colorama import Fore, Style
from maybe_find_sister_original_path_of_this_mask_path import (
     maybe_find_sister_original_path_of_this_mask_path
)
import textwrap
from pathlib import Path
from typing import Optional, Tuple, List


def get_datapoint_path_tuples_for_nfl(
) -> List[Tuple[Path, Path, Optional[Path]]]:
        
    dataset_folders = [
        Path("~/r/nfl-59773-skycam-ddv3_floor/.approved").expanduser()
    ]

    for dataset_folder in dataset_folders:
        assert dataset_folder.is_dir(), f"dataset_folder must be a directory, but {dataset_folder=} is not a directory"

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
            multiplicity = 1
            if original_path.name.startswith("nfl-59773-skycam-ddv3"):
                multiplicity = 1
            for m in range(multiplicity):    
                datapoint_path_tuples.append(
                    (original_path, mask_path, maybe_relevance_path)
                )
    
    return datapoint_path_tuples


if __name__ == "__main__":
    datapoint_path_tuples = get_datapoint_path_tuples_for_nfl()
    for t in datapoint_path_tuples:
        original, mask, relevance = t
        print(f"{original=!s}")
        print(f"{mask=!s}")
        print(f"{relevance=!s}")
        print()

    num_training_points = len(datapoint_path_tuples)
    print_yellow(f"{num_training_points=}")

       