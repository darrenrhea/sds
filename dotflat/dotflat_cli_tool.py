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

def full_relative_pathify_list(L, repo_dir):
    """
    given a string fragment of a repo_dir relative path,
    this makes it into a full relative path.
    """
    return [
        full_relative_path_ify_this_mask_file_name_fragment(
            file_name_fragment=file_name_fragment,
            repo_dir=repo_dir
        )
        for file_name_fragment in L
    ]


def dotflat_cli_tool():
    if len(sys.argv) == 1:
        repo_dir = Path.cwd()
    else:
        repo_dir = Path(sys.argv[1]).resolve()

    print(f"{Fore.GREEN}Making .flat hidden directory filled with all the annotations so that you can flipflop it:{Style.RESET_ALL}")

    dot_git_dir =  repo_dir / ".git"

    assert (
        dot_git_dir.exists()
    ), "ERROR: .git directory {dot_git_dir} not found, are you sure you are in the top-level directory of a git repository or specified the top-level of a git repository?"

    dot_flat_dir_path = repo_dir / ".flat"

    

    # delete the dotflat directory:
    if dot_flat_dir_path.exists():
        assert dot_flat_dir_path.name == ".flat", "ERROR: trying to delete something other than .flat"
        shutil.rmtree(dot_flat_dir_path)

    # create the dotflat directory, but not its parents:
    dot_flat_dir_path.mkdir(exist_ok=True, parents=False)

    dot_originals = repo_dir / ".originals"
    shutil.rmtree(dot_originals, ignore_errors=True)
    dot_originals.mkdir(exist_ok=True, parents=False)

    select_subfolders = [
        "anna",
        "baynzo",
        "ben",
        "carla",
        "chris",
        "darren",
        "grace",
        "justan",
        "katie",
        "rebecca",
        "ross",
        "ruby",
        "sarah",
        "seqouyah",
        "stephen",
        "thomas",
    ]
    mask_paths = []
    original_to_mask = {}
    mask_to_original = {}
    # better to interate over the mask paths, which are always _nonfloor.png,
    # and which suggest at least an attempt at labeling:
    for mask_path in repo_dir.rglob("*_nonfloor.png"):
        # print(f"{mask_path.parts[-2]=}")
        if ".flat" in mask_path.parts:
            continue
        if ".approved" in mask_path.parts:
            continue
        if ".rejected" in mask_path.parts:
            continue
        if ".unknown" in mask_path.parts:
            continue
        if ".originals" in mask_path.parts:
            continue
        if mask_path.parts[-2] in select_subfolders:
            print(f"{mask_path=}")
            mask_paths.append(mask_path)
            # print(f"{mask_path.parent=}")

            annotation_id = mask_path.name[:-len("_nonfloor.png")]
            
            # try _original.png first:
            original_path_version1 = mask_path.parent / f"{annotation_id}_original.png"
            original_path = None
            if original_path_version1.is_file():
                original_path = original_path_version1
            else:
                original_path_version2 = mask_path.parent / f"{annotation_id}.jpg"
                if original_path_version2.is_file():
                    original_path = original_path_version2
                else:
                    original_path_version3 = mask_path.parent / f"{annotation_id}_original.jpg"
                    if original_path_version3.is_file():
                        original_path = original_path_version3
                    else:
                        print_error(f"Although\n    {mask_path=}\nexists,\nneither original file:\n\n    {original_path_version1=}\nnor\n     {original_path_version2=}\nnor\n     {original_path_version3=}\n was found!\n")
                        sys.exit(1)

            assert original_path is not None
            assert original_path.is_file(), f"ERROR: {original_path} not found! This should never happen!"

            rel_mask_path = mask_path.relative_to(repo_dir)
            rel_original_path = original_path.relative_to(repo_dir)
            original_to_mask[rel_original_path] = rel_mask_path
            mask_to_original[rel_mask_path] = rel_original_path
        
    for mask_path, original_path in mask_to_original.items():
        shutil.copy(
            src=mask_path,
            dst=dot_flat_dir_path
        )

        shutil.copy(
            src=original_path,
            dst=dot_flat_dir_path
        )

        shutil.copy(
            src=original_path,
            dst=dot_originals
        )


        

    print("Suggest you do:")
    print(f"flipflop {dot_flat_dir_path}")

    approvals_file = repo_dir / "approvals.json5"
    if not approvals_file.exists():
        print_warning(f"You don't have an approvals.json5 file in {repo_dir}! We will make you one with everything unknown quality.")
        jsonable = dict(
            unknown=sorted([str(p.relative_to(repo_dir)) for p in mask_paths]),
            approved=[],
            rejected=[],
        )
        bj.dump(fp=approvals_file, obj=jsonable)
        print(f"Made {approvals_file}")
    
    assert approvals_file.is_file(), f"ERROR: {approvals_file} still not found!  This should never happen!"
    
    # BEGIN check sanity of approvals.json5:

    print(f"Found {approvals_file}")
    approval_info = bj.load(approvals_file)
    for key in approval_info.keys():
        assert key in ["approved", "rejected", "unknown"], f"ERROR: {approvals_file} has a key {key} that is not one of approved, rejected, or unknown!"
    
    approved_strs = approval_info["approved"]
    rejected_strs = approval_info["rejected"]
    unknown_strs = []  # we regenerate runknown as what is not approved nor rejected.
    
    approved_rel_paths = full_relative_pathify_list(approved_strs, repo_dir)
    rejected_rel_paths = full_relative_pathify_list(rejected_strs, repo_dir)
    unknown_rel_paths = full_relative_pathify_list(unknown_strs, repo_dir)

    both_approved_and_rejected = set(approved_rel_paths).intersection(set(rejected_rel_paths))
    if both_approved_and_rejected:
        print("ERROR: approved and rejected lists have intersection! These files are both approved and rejected:")
        for path in sorted(list(both_approved_and_rejected)):
            print(f"    {path}")
            sys.exit(1)
    
    # both_approved_and_unknown = set(approved_rel_paths).intersection(set(unknown_rel_paths))
    # if both_approved_and_unknown:
    #     print_error("ERROR: approved and unknown lists have intersection! These files are both approved and unknown:")
    #     for path in sorted(list(both_approved_and_unknown)):
    #         print(f"    {path}")
    #         sys.exit(1)
    
    # both_rejected_and_unknown = set(rejected_rel_paths).intersection(set(unknown_rel_paths))
    # if both_rejected_and_unknown:
    #     print_error("ERROR: rejected and unknown lists have intersection! These files are both rejected and unknown:")
    #     for path in sorted(list(both_rejected_and_unknown)):
    #         print(f"    {path}")
    #         sys.exit(1)
    # ENDOF check sanity of approvals.json5.

    # add any masks that are not in the approvals.json5 file to the unknown list:
    for mask_path in mask_paths:
        rel_path = mask_path.relative_to(repo_dir)
        print(f"{rel_path=}")
        if (
            rel_path not in approved_rel_paths
            and
            rel_path not in rejected_rel_paths
            and
            rel_path not in unknown_rel_paths
        ):
            print(f"Adding {rel_path} to unknown list in {approvals_file} because it showed up in the repo and is neither approved nor rejected.json5 file.")
            unknown_rel_paths.append(rel_path)
    
    unknown_rel_paths = sorted(unknown_rel_paths)
    approved_rel_paths = sorted(approved_rel_paths)
    rejected_rel_paths = sorted(rejected_rel_paths)


    new_approvals_info = dict(
        approved=[str(x) for x in approved_rel_paths],
        rejected=[str(x) for x in rejected_rel_paths],
        unknown=[str(x) for x in unknown_rel_paths],
    )

    bj.dump(
        obj=new_approvals_info,
        fp=approvals_file
    )

    # BE CAREFUL: This will delete the folders:
    dot_approved = repo_dir / ".approved"
    dot_rejected = repo_dir / ".rejected"
    dot_unknown = repo_dir / ".unknown"
    
    shutil.rmtree(dot_approved, ignore_errors=True)
    shutil.rmtree(dot_rejected, ignore_errors=True)
    shutil.rmtree(dot_unknown, ignore_errors=True)

    
    dot_approved.mkdir(exist_ok=True, parents=False)
    dot_rejected.mkdir(exist_ok=True, parents=False)
    dot_unknown.mkdir(exist_ok=True, parents=False)

    for mask_rel_path in approved_rel_paths:
        mask_abs_path = repo_dir / mask_rel_path
        assert mask_abs_path.is_file(), f"ERROR: {mask_abs_path} not found!\nWhy is {mask_rel_path} in the approved list? There is no such file!"
        shutil.copy(
            src=mask_abs_path,
            dst=dot_approved
        )
        rel_original_path = mask_to_original[mask_rel_path]
        shutil.copy(
            src=rel_original_path,
            dst=dot_approved
        )

    
    for mask_rel_path in rejected_rel_paths:
        mask_abs_path = repo_dir / mask_rel_path
        assert mask_abs_path.is_file(), f"ERROR: {mask_abs_path} not found!\nWhy is {mask_rel_path} in the rejected list? There is no such file!"
        shutil.copy(
            src=mask_abs_path,
            dst=dot_rejected
        )
        rel_original_path = mask_to_original[mask_rel_path]
        shutil.copy(
            src=rel_original_path,
            dst=dot_rejected
        )

    for mask_rel_path in unknown_rel_paths:
        mask_abs_path = repo_dir / mask_rel_path
        assert mask_abs_path.is_file(), f"ERROR: {mask_abs_path} not found!\nWhy is {mask_rel_path} in the unknown list? There is no such file!"
        shutil.copy(
            src=mask_abs_path,
            dst=dot_unknown
        )
        rel_original_path = mask_to_original[mask_rel_path]
        shutil.copy(
            src=rel_original_path,
            dst=dot_unknown
        )
    
    print("Supposedly:")
    print(f"#approved: {len(approved_rel_paths)}")
    print(f"#Rejected: {len(rejected_rel_paths)}")
    print(f"#Unknown: {len(unknown_rel_paths)}")
    print("\n\n\n")

    print("Suggest you do:")
    print(f"flipflop {dot_unknown}")
    print(f"flipflop {dot_approved}")
    print(f"flipflop {dot_rejected}")

           

            