import textwrap
from pathlib import Path
import sys


def get_mask_path_from_original_path(
    original_path: Path
) -> Path:
    """
    Although our mask are always named lke {annotation_id}_nonfloor.png,
    the original images have different forms like:
    {annotation_id}_original.png
    {annotation_id}_original.jpg
    {annotation_id}.jpg
    """
    annotation_id = None
    if original_path.name.endswith("_original.png"):
        annotation_id = original_path.name[:-len("_original.png")]
    elif original_path.name.endswith("_original.jpg"):
        annotation_id = original_path.name[:-len("_original.jpg")]
    elif original_path.suffix == ".jpg":
        annotation_id = original_path.stem
    else:
        print(
            textwrap.dedent(
                f"""\
                original_path {original_path=} must end with _original.png or _original.jpg or be a .jpg file!
                """
            )
        )
        sys.exit(1)
    
    mask_path = original_path.parent / f"{annotation_id}_nonfloor.png"

    assert (
        mask_path.is_file()
    ), f"ERROR: mask_path must be an extant file! but {mask_path=} does not exist!"

    return mask_path