from pathlib import Path
import sys
from colorama import Fore, Style
from typing import Callable


def find_most_recently_modified_file_in_this_directory(
        directory: Path,
        predicate: Callable
    ) -> Path:
    """
    returns the absolute Path of the most recently modified file within the given directory
    WHICH IS NOT PART OF THE .git subfolders.
    """
    assert directory.is_dir(), f"ERROR: {directory} is not a directory"

    best_so_far = -1e100
    argmax = None
    # for p in directory.iterdir():
    # https://stackoverflow.com/questions/6639394/what-is-the-python-way-to-walk-a-directory-tree
    for p in directory.rglob("*"):
        
        p = p.resolve()  # make it absolute
        
        if not p.is_file():
            continue

        is_git_shit = False
        for part in p.parts:
            if part == ".git":
                is_git_shit = True
                break

        if p.name == '.DS_Store':
            continue
        
        if is_git_shit:
            continue

        if not predicate(p):
            continue

        mod_timestamp = p.lstat().st_mtime
        # print(datetime.datetime.fromtimestamp(mod_timestamp))
        if mod_timestamp >= best_so_far:
            if mod_timestamp == best_so_far:
                print(f"{Fore.YELLOW}WARNING: there is an exact tie in the modified time, down to the microsecond, between {p} and {argmax}{Style.RESET_ALL}")
            best_so_far = mod_timestamp
            argmax = p
    
    if argmax is None:
        print("ERROR: maybe the directory {directory} has no files in it???")
        sys.exit(1)
    
    print(f"{Fore.YELLOW}Most recent file in {directory} is {argmax}{Style.RESET_ALL}")
    return argmax
