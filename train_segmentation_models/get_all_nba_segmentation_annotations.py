import numpy as np
from print_yellow import (
     print_yellow
)

from ltjfthts_load_the_jsonlike_file_that_has_this_sha256 import (
     ltjfthts_load_the_jsonlike_file_that_has_this_sha256
)


def get_all_nba_segmentation_annotations(
    all_segmentation_annotations_sha256: str
):
    real_data = ltjfthts_load_the_jsonlike_file_that_has_this_sha256(
        sha256_of_the_jsonlike_file=all_segmentation_annotations_sha256,
        check=True
    )
    appropriate_real_data = []
    clip_ids = set()
    for x in real_data:
        info = x["info"]
        clip_id = x["clip_id"]
        home_team = info["home_team"]
        league = info["league"]

        # #if home_team == "hou" or clip_id
        #     # print_red(f"skipping {x} because it is houston")
        #     continue
        if league != "nba":
            # print_yellow(f"skipping {clip_id} because it is not nba")
            continue
        appropriate_real_data.append(x)
        clip_ids.add(clip_id)
    # print_yellow(
    #     sorted(list(clip_ids))
    # )
    print_yellow(
        f"num NBA segmentation annotations = {len(appropriate_real_data)}"
    )
    
    np.random.shuffle(appropriate_real_data)
    
    return appropriate_real_data


if __name__ == "__main__":
    # Example usage
    all_segmentation_annotations = (
        get_all_nba_segmentation_annotations(
            all_segmentation_annotations_sha256=(
                "6d7074c40a5aa53286f14e8127d2822f9e5ccb68bee112fa6e43f10f4c6a8485"
            )
        )
    )