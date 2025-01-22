import subprocess
from get_set_of_bad_pngs import (
     get_set_of_bad_pngs
)
from get_approved_annotations_from_these_repos import (
     get_approved_annotations_from_these_repos
)
from maybe_find_sister_original_path_of_this_mask_path import (
     maybe_find_sister_original_path_of_this_mask_path
)
from pathlib import Path
import magic
import shutil

"""
brew install libmagic
"""

mask_file_path = Path(
    "/Users/darrenrhea/r/munich1080i_led/ben/munich2024-01-25-1080i-yadif_104000_nonfloor.png"
)

original_file_path = maybe_find_sister_original_path_of_this_mask_path(
    mask_path=mask_file_path
)

def determine_bit_depth(original_file_path: Path) -> int:

    t = magic.from_file(str(original_file_path))
    if "16-bit" in t:
        return 16
    elif "8-bit" in t:
        return 8
    else:
        raise ValueError(f"ERROR: {original_file_path=} is neither 8-bit nor 16-bit?")


def get_rid_of_16_bit():
    repo_ids_to_use = [
        "bay-czv-2024-03-01_led",
        "bay-efs-2023-12-20_led",
        "bay-mta-2024-03-22-mxf_led",
        "bay-mta-2024-03-22-part1-srt_led",
        "bay-zal-2024-03-15-yt_led",
        "maccabi_fine_tuning",
        "maccabi1080i_led",
        "munich1080i_led",
    ]

    actual_annotations = get_approved_annotations_from_these_repos(
        repo_ids_to_use=repo_ids_to_use
    )
    
    files_to_mutate = []
    repos_affected = set()
    for actual_annotation in actual_annotations:
        mask_file_path = actual_annotation["mask_file_path"]
        original_file_path = actual_annotation["original_file_path"]

        if mask_file_path.suffix == ".png":
            mask_depth = determine_bit_depth(original_file_path=mask_file_path)
            assert mask_depth is not None
            if mask_depth == 16:
                files_to_mutate.append(mask_file_path)
                repos_affected.add(mask_file_path.parent.parent.name)
        
        if original_file_path.suffix == ".png":
            original_depth = determine_bit_depth(original_file_path=original_file_path)
            assert original_depth is not None

            if original_depth == 16:
                files_to_mutate.append(original_file_path)
                repos_affected.add(original_file_path.parent.parent.name)

    # files_to_mutate = files_to_mutate[:1]

    for file_path in files_to_mutate:
        args = [
            "convert",
            str(file_path),
            "-depth",
            "8",
            "temp.png"
        ]
        print(" ".join(args))
        subprocess.run(args=args, check=True)
        shutil.move("temp.png", file_path)
    # actual_set = set([p.name for p in files_to_mutate])
    

    # should_be = set(get_set_of_bad_pngs())
    # print("New ones to fix:")
    # print(should_be.difference(actual_set))

    # print(actual_set.difference(should_be))
    print("Affected repos:")
    for r in list(repos_affected):
        print(r)
    
        
if __name__ == "__main__":
    get_rid_of_16_bit()