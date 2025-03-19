import argparse
import pprint as pp
from print_error import (
     print_error
)
from colorama import Fore, Style
from print_warning import (
     print_warning
)

import shutil
from pathlib import Path
import sys
import better_json as bj

from full_relative_path_ify_this_mask_file_name_fragment import (
     full_relative_path_ify_this_mask_file_name_fragment
)



def approve_frame_range_cli_tool():
    argp = argparse.ArgumentParser()
    argp.add_argument("a", type=int, help="The first frame index to approve")
    argp.add_argument("b", type=int, help="The last frame index to approve")
    opt = argp.parse_args()
    a = opt.a
    b = opt.b


    repo_dir = Path.cwd()

    print(f"{Fore.GREEN}Making .flat hidden directory filled with all the annotations so that you can flipflop it:{Style.RESET_ALL}")

    dot_git_dir =  repo_dir / ".git"

    assert (
        dot_git_dir.exists()
    ), "ERROR: .git directory {dot_git_dir} not found, are you sure you are in the top-level directory of a git repository or specified the top-level of a git repository?"

    dot_flat_dir_path = repo_dir / ".flat"
    assert dot_flat_dir_path.is_dir(), f"ERROR: {dot_flat_dir_path} not found! This should never happen!"
    
    L = []
    for mask_path in dot_flat_dir_path.rglob("*_nonfloor.png"):
        annotation_id = mask_path.name[:-len("_nonfloor.png")]
        frame_index_str = annotation_id.split("_")[-1]
        frame_index = int(frame_index_str)
        if a <= frame_index and frame_index <= b:
            
            rel_path = full_relative_path_ify_this_mask_file_name_fragment(
                file_name_fragment=annotation_id,
                repo_dir=repo_dir
            )
            print(f"{rel_path=}")
            L.append(str(rel_path))
    
    L = sorted(L)
    
    for s in L:
        print(s)

    approvals = bj.load("approvals.json5")
    approved = approvals["approved"]
    for s in L:
        if s not in approved:
            print(f"WARNING: {s} not in approvals.json5")
        else:
            print(f"{s} in approvals.json5")
    print("bye")



        
