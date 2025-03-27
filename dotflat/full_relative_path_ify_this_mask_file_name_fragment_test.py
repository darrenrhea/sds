from full_relative_path_ify_this_mask_file_name_fragment import (
     full_relative_path_ify_this_mask_file_name_fragment
)

from pathlib import Path



def test_full_relative_path_ify_this_mask_file_name_fragment():
    
    repo_dir = Path("~/r/munich4k_importantpeople").expanduser()

    file_name_fragment = "DSCF0236_000475"

    answer = full_relative_path_ify_this_mask_file_name_fragment(
        repo_dir=repo_dir,
        file_name_fragment=file_name_fragment,
    )

    assert (
        answer == Path("grace/DSCF0236_000475_nonfloor.png")
    ), f"Got {answer} instead of grace/DSCF0236_000475_nonfloor.png"


if __name__ == "__main__":
    test_full_relative_path_ify_this_mask_file_name_fragment()
    print("full_relative_path_ify_this_mask_file_name_fragment_test.py: all tests passed")