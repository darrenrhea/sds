from delete_all_files_in_this_folder_that_are_not_in_this_folder import (
     delete_all_files_in_this_folder_that_are_not_in_this_folder
)

import argparse
from pathlib import Path


def delete_all_files_in_this_folder_that_are_not_in_this_folder_cli_tool():
    """
    delete_all_files_in_this_folder_that_are_not_in_this_folder \
    ~/fake/baseball/brewers/gray_uniform/pitchers/originals/ \
    ~/fake/baseball/brewers/gray_uniform/pitchers/masked/
    """

    argp = argparse.ArgumentParser()
    argp.add_argument("victim_folder", type=Path)
    argp.add_argument("limiting_folder", type=Path)
    argp.add_argument("--dry-run", action="store_true")  # dry_run will be false unless you specify this flag

    args = argp.parse_args()
    victim_folder = Path(args.victim_folder).resolve()
    limiting_folder = Path(args.limiting_folder).resolve()
    dry_run = args.dry_run
    
    delete_all_files_in_this_folder_that_are_not_in_this_folder(
        victim_folder=victim_folder,
        limiting_folder=limiting_folder,
        dry_run=dry_run
    )