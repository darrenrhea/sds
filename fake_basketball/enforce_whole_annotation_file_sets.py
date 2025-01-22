from maybe_find_sister_original_path_of_this_mask_path import (
     maybe_find_sister_original_path_of_this_mask_path
)
from pathlib import Path
import sys


def ewafs_enforce_whole_annotation_file_sets():
    fake_dir = Path(sys.argv[1]).resolve()
      
    mask_paths = list(fake_dir.glob("*_nonfloor.png"))
    print(f"Found {len(mask_paths)} mask_paths")
    annotation_ids_to_destroy = set()
    annotation_ids = set()

    for mask_path in mask_paths:
        assert mask_path.is_file(), f"ERROR: {mask_path} is not a file"
        assert mask_path.name.endswith("_nonfloor.png"), f"ERROR: {mask_path=} does not end with _nonfloor.png"
        annotation_id = mask_path.stem[:-9]
        annotation_ids.add(annotation_id)
        original_path = maybe_find_sister_original_path_of_this_mask_path(
            mask_path=mask_path
        )
        if original_path is None:
            print(f"ERROR: {mask_path=} has no corresponding original")
            annotation_ids_to_destroy.add(annotation_id)
    
    num_original_paths = 0
    for original_path in fake_dir.glob("*_original.*"):
        annotation_id = original_path.stem[:-9]
        annotation_ids.add(annotation_id)
        num_original_paths += 1
    print(f"Found {num_original_paths} original_paths")

    for annotation_id in annotation_ids:
        # don't check for missing originals again, it is subtle:
        mask_path = fake_dir / f"{annotation_id}_nonfloor.png"

        if not mask_path.is_file():
            print(f"ERROR: {mask_path=} is missing for {annotation_id=}")
            annotation_ids_to_destroy.add(annotation_id)

        if "fake" in annotation_id:
            relevance_path = fake_dir / f"{annotation_id}_relevance.png"
            if not relevance_path.is_file():
                print(f"ERROR: {relevance_path=} is missing for a fake one.")
                annotation_ids_to_destroy.add(annotation_id)


    # erase all the files for the annotation_id:
    for annotation_id in annotation_ids_to_destroy:
        for suffix in ["_nonfloor.png", "_relevance.png", "_original.jpg", "_original.png"]:
            p =fake_dir / f"{annotation_id}{suffix}"
            if p.is_file():
                print(f"Deleting {p}")
                p.unlink()


if __name__ == "__main__":
    print("Hello")
    ewafs_enforce_whole_annotation_file_sets()