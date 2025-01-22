from pathlib import Path
import sys
import textwrap

from colorama import Fore, Style


def find_most_recently_modified_immediate_subdirectory_of_this_glob_form_in_this_directory(
    directory: Path,
    glob_form: str
) -> Path:
    """
    Sometimes you want to know something like:
    of all the directories which are immediate subdirectories of the directory 
    which one was most recently modified?    
    returns the absolute Path of the most recently modified subdirectory of the stated directory
    TODO: considering https://stackoverflow.com/questions/3620684/directory-last-modified-date
    we might want to consider recursively searching for the most recently modified file within the directories.
    """
    assert (
        isinstance(directory, Path)
    ), f"directory must be a Path, but you gave {directory} which is a {type(directory)}"

    assert directory.is_dir(), f"ERROR: {directory} is not a directory"

    best_so_far = -1e100
    argmax = None
    for p in directory.glob(glob_form):
        
        p = p.resolve()  # make it absolute
        
        if not p.is_dir():
            continue

        mod_timestamp = p.lstat().st_mtime
        # print(datetime.datetime.fromtimestamp(mod_timestamp))
        if mod_timestamp >= best_so_far:
            if mod_timestamp == best_so_far:
                print(
                    textwrap.dedent(
                        f"""\
                        {Fore.YELLOW}
                        WARNING: there is an exact tie in the modified time, down to the microsecond, between
                        {p}
                        and
                        {argmax}
                        {Style.RESET_ALL}
                        """
                    )
                )
            best_so_far = mod_timestamp
            argmax = p
    
    if argmax is None:
        print(f"ERROR: maybe the directory {directory} has no files in it???")
        sys.exit(1)
    
    print(f"{Fore.YELLOW}Most recent file in {directory} is {argmax}{Style.RESET_ALL}")
    return argmax


    