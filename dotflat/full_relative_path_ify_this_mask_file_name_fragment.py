from print_error import (
     print_error
)

from pathlib import Path
import sys
import re

def full_relative_path_ify_this_mask_file_name_fragment(
    file_name_fragment: str,
    repo_dir: Path
)-> Path:
    """
    People who are manipulating the approvals.json5 file should be allowed to mention files by a file_name_fragment,
    like:
    
    DSCF0241_000047
    or
    ben/DSCF0241_000047

    instead of ben/DSCF0241_000047_nonfloor.png.
    This function will take a file_name_fragment and, to the extent it is possible,
    identify the full relative path of the file in the repo_dir that contains the file_name_fragment.
    """
    # search file_name_fragment for the regex [0-9]{6}. If the pattern is not found, exit:
    
    if not re.search(r'[0-9]{6}', file_name_fragment):
        print_error(f'file_name_fragment {file_name_fragment} does not contain a 6-digit number, so we think something is wrong. Exiting.')
        sys.exit(1)
    
    num_times_found = 0
    finds = []
    for mask_path in repo_dir.rglob("*"):
        assert mask_path.is_relative_to(repo_dir)
        if not mask_path.is_file():
            continue
        if ".flat" in mask_path.parts:
            continue
        if ".approved" in mask_path.parts:
            continue
        if ".rejected" in mask_path.parts:
            continue
        if ".unknown" in mask_path.parts:
            continue
        if ".git" in mask_path.parts:
            continue
        if ".flat" in mask_path.parts:
            continue
        if not mask_path.name.endswith("_nonfloor.png"):
            continue
        rel_path = mask_path.relative_to(repo_dir)
        if file_name_fragment not in str(rel_path):
            continue
        # seems like we found a match:
        answer = rel_path
        finds.append(answer)
        num_times_found += 1
    
    if num_times_found == 0:
        print_error(f'ERROR: No _nonfloor.png file whose path relative to {repo_dir} contains {file_name_fragment} was found in {repo_dir}. Exiting.')
        sys.exit(1)
    if num_times_found > 1:
        print_error(f"ERROR: There is more than one _nonfloor.png file containing {file_name_fragment} was found in {repo_dir}:")
        for find in finds:
            print_error(find)
        print(f"It may be that {file_name_fragment} is not specific enough to identify a single file, or")
        print("maybe multiple people get assigned the same annotation? Pick one as the answer and delete the other")
    answer = finds[0]

    assert isinstance(answer, Path)
    return answer