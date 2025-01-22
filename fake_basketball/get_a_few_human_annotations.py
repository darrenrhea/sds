from get_annotations_from_list_of_paths import (
     get_annotations_from_list_of_paths
)

from maybe_find_sister_original_path_of_this_mask_path import (
     maybe_find_sister_original_path_of_this_mask_path
)


from pathlib import Path


def get_a_few_human_annotations():
    r_dir = Path("~/r").expanduser()

    # many tests use just the first of these:
    these = [
        "munich1080i_led/sarah/munich2024-01-09-1080i-yadif_010937_nonfloor.png",
        "munich1080i_led/sarah/munich2024-01-25-1080i-yadif_343409_nonfloor.png",
        "bay-zal-2024-03-15-mxf-yadif_led/anna/bay-zal-2024-03-15-mxf-yadif_133917_nonfloor.png",
        "bay-zal-2024-03-15-mxf-yadif_led/anna/bay-zal-2024-03-15-mxf-yadif_101856_nonfloor.png",
    ]

    mask_paths = [
        r_dir / x
        for x in these
    ]

    list_of_original_and_mask_paths = []
    for mask_path in mask_paths:
        print(mask_path)
        original_path = maybe_find_sister_original_path_of_this_mask_path(
            mask_path=mask_path
        )
        pair = (original_path, mask_path)
        list_of_original_and_mask_paths.append(pair)

    
    annotations = get_annotations_from_list_of_paths(
        list_of_original_and_mask_paths=list_of_original_and_mask_paths
    )
    return annotations
