from pathlib import Path
from colorama import Fore, Style
import argparse


def delete_all_files_in_this_folder_that_are_not_in_this_folder(
    victim_folder: Path,
    limiting_folder: Path,
    dry_run: bool
):
    assert isinstance(dry_run, bool)
    assert isinstance(victim_folder, Path)
    assert isinstance(limiting_folder, Path)
    assert victim_folder.is_dir(), f"{victim_folder} does not exist or isnt a directory"
    assert limiting_folder.is_dir(), f"{limiting_folder} does not exist or isnt a directory"

    if dry_run:
        for p in victim_folder.rglob("*"):
            if not p.is_file():
                continue
            rel_path = p.relative_to(victim_folder)
            corresponding = limiting_folder / rel_path
            print(p)
            print(corresponding)
            if not corresponding.exists():
                print(f"  {Fore.RED}WARNING: {rel_path} does not exist in {limiting_folder}{Style.RESET_ALL}")
                print(f"  {Fore.RED}WARNING: therefore {p} will be deleted{Style.RESET_ALL}")
                print("")
            print("")
    else:
        for p in victim_folder.rglob("*"):
            if not p.is_file():
                continue
            rel_path = p.relative_to(victim_folder)
            corresponding = limiting_folder / rel_path
            
            if not corresponding.exists():
                p.unlink()  # delete the file
                print(f"  {Fore.RED}{rel_path} does not exist in {limiting_folder}{Style.RESET_ALL}")
                print(f"  {Fore.RED}therefore we deleted {p}{Style.RESET_ALL}")
                print("")
            print("")
