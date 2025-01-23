import textwrap
from colorama import Fore, Style
from maybe_find_sister_original_path_of_this_mask_path import (
     maybe_find_sister_original_path_of_this_mask_path
)
from pathlib import Path
from typing import Optional, Tuple, List

def get_datapoint_path_tuples_for_rockets_core(
) -> List[Tuple[Path, Path, Optional[Path]]]:
        
    dataset_folders = [
        # all kinds of floors for some generalizability esp for uniforms:
        Path("/shared/all_training_data/floor_not_floor").expanduser(),
        # Path("/shared/fake_nba/22-23_ATL_CORE").expanduser(),
        # Path("/shared/fake_nba/22-23_CHI_CORE").expanduser(),
        # Path("/shared/fake_nba/22-23_WAS_CORE").expanduser(),
        
        # This is fake then with player cutouts pasted on top:
        # Path("/shared/all_training_data/fake_nba/pasted").expanduser(),  # 634
        # Path("/shared/all_training_data/fake_nba/24-25_HOU_CORE/chunk0"),
        # Path("/shared/all_training_data/fake_nba/24-25_HOU_CORE/chunk1"),
        # Path("/shared/all_training_data/fake_nba/24-25_HOU_CORE/chunk2"),
        # Path("/shared/all_training_data/fake_nba/24-25_HOU_CORE/chunk3"),
        # Path("/shared/all_training_data/fake_nba/24-25_HOU_CORE/chunk0_edition2"),
        # Path("/shared/all_training_data/fake_nba/24-25_HOU_CORE/chunk1_edition2"),
        # Path("/shared/all_training_data/fake_nba/24-25_HOU_CORE/chunk2_edition2"),
        # Path("/shared/all_training_data/fake_nba/24-25_HOU_CORE/chunk3_edition2"),
        
        # This is fake/synthetic Statement Edition court, i.e. the bloodbath red Cup Court
        # Path("/shared/all_training_data/fake_nba/chunk0_24-25_HOU_STMT").expanduser(),
        # Path("/shared/all_training_data/fake_nba/chunk1_24-25_HOU_STMT").expanduser(),
        # Path("/shared/all_training_data/fake_nba/chunk2_24-25_HOU_STMT").expanduser(),
        # Path("/shared/all_training_data/fake_nba/chunk3_24-25_HOU_STMT").expanduser(),
        # Path("/shared/all_training_data/fake_nba/24-25_HOU_STMT/chunk0").expanduser(),
        # Path("/shared/all_training_data/fake_nba/24-25_HOU_STMT/chunk1").expanduser(),
        # Path("/shared/all_training_data/fake_nba/24-25_HOU_STMT/chunk2").expanduser(),
        # Path("/shared/all_training_data/fake_nba/24-25_HOU_STMT/chunk3").expanduser(),
        
        # this should help to reinforce that netcam should work for the HOU_CORE court:
        # Path("/shared/allnetcams").expanduser(),  # 68 right now
        # Path("/shared/all_training_data/exotic_viewpoints_hou_core").expanduser(),  # 274 right now
        # Path("/shared/all_training_data/exotic_viewpoints_hou_city").expanduser(),  # 12 right now


        # These are fake/synthetic City Edition court, i.e. Houston Summit / H-Town
        # Path("/shared/all_training_data/fake_nba/a1_24-25_HOU_CITY").expanduser(),
        # Path("/shared/all_training_data/fake_nba/a2_24-25_HOU_CITY").expanduser(),
        # Path("/shared/all_training_data/fake_nba/b1_24-25_HOU_CITY").expanduser(),
        # Path("/shared/all_training_data/fake_nba/b2_24-25_HOU_CITY").expanduser(),
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
            if "fake" in original_path.name:     
                datapoint_path_tuples.append(
                    (original_path, mask_path, maybe_relevance_path)
                )
            elif (
                "exotic_viewpoints_hou_core" in str(dataset_folder)
                or
                "exotic_viewpoints_hou_city" in str(dataset_folder)
            ):
                for i in range(9): 
                    datapoint_path_tuples.append(
                        (original_path, mask_path, maybe_relevance_path)
                    )
            elif (
                original_path.name.startswith("hou-det-2025-01-20-sdi")
            ):
                for i in range(10): 
                    datapoint_path_tuples.append(
                        (original_path, mask_path, maybe_relevance_path)
                    )
            elif (
                original_path.name.startswith("hou-gsw-2024-11-02-sdi")
                or
                original_path.name.startswith("hou-sas-2024-10-17-sdi")
                or
                original_path.name.startswith("hou-was-2024-11-11-sdi")
            ):
                for i in range(12): 
                    datapoint_path_tuples.append(
                        (original_path, mask_path, maybe_relevance_path)
                    )
            else:
                datapoint_path_tuples.append(
                    (original_path, mask_path, maybe_relevance_path)
                )
    return datapoint_path_tuples