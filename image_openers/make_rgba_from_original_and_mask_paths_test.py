from prii import (
     prii
)
from make_rgba_from_original_and_mask_paths import (
     make_rgba_from_original_and_mask_paths
)

from pathlib import Path


def test_make_rgba_from_original_and_mask_paths_1():
    """The trouble
    exiftool /Users/darrenrhea/r/munich1080i_led/.approved/munich2024-01-25-1080i-yadif_082000_nonfloor.png
    # this is a 16 bit png:
    exiftool /Users/darrenrhea/r/munich1080i_led/.approved/munich2024-01-25-1080i-yadif_082000_original.png
    """


    original_path = Path(
        "/Users/darrenrhea/r/munich1080i_led/.approved/munich2024-01-25-1080i-yadif_082000_original.png",
    ).expanduser()
    
    mask_path = Path(
        "/Users/darrenrhea/r/munich1080i_led/.approved/munich2024-01-25-1080i-yadif_082000_nonfloor.png",
    )

    actual_annotation_rgba_np_u8 = make_rgba_from_original_and_mask_paths(
        original_path=original_path,
        mask_path=mask_path,
        flip_mask=False,
        quantize=False,
    )
    
    prii(
        actual_annotation_rgba_np_u8
    )


if __name__ == "__main__":
    test_make_rgba_from_original_and_mask_paths_1()