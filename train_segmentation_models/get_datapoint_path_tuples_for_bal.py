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


def get_datapoint_path_tuples_for_bal(
) -> List[Tuple[Path, Path, Optional[Path]]]:
    
    mother_dir = Path("/shared/r")

    dataset_folders = [
        mother_dir / "bal2024_egypt_floor/.approved",
        mother_dir / "bal2024_southafrica_floor/.approved",
        mother_dir / "bal2024_senegal_floor/.approved",
        mother_dir / "bal2024_rwanda_floor/.approved",
        mother_dir / "stadepart2_floor/.approved",
        mother_dir / "fus-aia-2025-04-05_floor/.approved",
        mother_dir / "rabat_floor/.approved",
        
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
             
            datapoint_path_tuples.append(
                (original_path, mask_path, maybe_relevance_path)
            )
    
    return datapoint_path_tuples


if __name__ == "__main__":
    datapoint_path_tuples = get_datapoint_path_tuples_for_bal()
    for t in datapoint_path_tuples:
        original, mask, relevance = t
        print(f"{original=!s}")
        print(f"{mask=!s}")
        print(f"{relevance=!s}")
        print()

    num_training_points = len(datapoint_path_tuples)
    print_yellow(f"{num_training_points=}")

       