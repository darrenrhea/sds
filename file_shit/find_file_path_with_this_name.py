from pathlib import Path
from typing import Optional


def find_file_path_with_this_name(
    dir_path_to_search: Path,
    file_name: str,
    skip_hidden_dirs: bool = True
) -> Optional[Path]:
    """
    Finds a file with the given name in the given directory path recursively.
    If there are multiple files with the same name, the first one found is returned.
    """
    print(f"Searching for {file_name=} in {dir_path_to_search=}")
    assert isinstance(dir_path_to_search, Path), f"{dir_path_to_search} is not a Path object."
    assert isinstance(file_name, str), f"{file_name=} is not a string."
    assert dir_path_to_search.is_dir(), f"{dir_path_to_search} is not a directory."
    for p in dir_path_to_search.resolve().rglob(file_name):
        if skip_hidden_dirs and p.parent.name.startswith("."):
            continue
        if p.is_file():
            return p
    
    return None
