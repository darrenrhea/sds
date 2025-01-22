from get_all_the_approved_annotations_from_this_repo import (
     get_all_the_approved_annotations_from_this_repo
)

from pathlib import Path


def test_get_all_the_approved_annotations_from_this_repo_1():
    repo_dir = Path("~/r/munich1080i_led").expanduser()
    approved_annotations = get_all_the_approved_annotations_from_this_repo(
        repo_dir=repo_dir
    )

    for approved_annotation in approved_annotations:
        original_file_path = approved_annotation["original_file_path"]
        mask_file_path = approved_annotation["mask_file_path"]
        annotation_id = approved_annotation["annotation_id"]
        clip_id = approved_annotation["clip_id"]
        frame_index = approved_annotation["frame_index"]

        print(f"{original_file_path=}")
        print(f"{mask_file_path=}")
        print(f"{annotation_id=}")
        print(f"{clip_id=}")
        print(f"{frame_index=}")


    assert len(approved_annotations) > 0, f"ERROR: {approved_annotations=} is empty"

    print(f"There are {len(approved_annotations)} approved annotations in {repo_dir}")


if __name__ == "__main__":
    test_get_all_the_approved_annotations_from_this_repo_1()