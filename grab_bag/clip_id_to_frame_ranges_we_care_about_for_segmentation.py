from pathlib import Path
import better_json as bj
import subprocess
from typing import List, Tuple, Union


def clip_id_to_frame_ranges_we_care_about_for_segmentation(
    clip_id: str
) -> List[
    Union[
        int,
        Tuple[int, int],
        Tuple[int, int, int]
    ]
]:
    """
    Given a clip_id, return the frame ranges we care about
    for segmentation.
    Tends to cut out any timeouts, commercials, half-time, etc.
    """

    subprocess.run(
        [
            "git",
            "pull",
        ],
        cwd=Path("~/r/final_models").expanduser()
    )

    clip_id_to_frame_ranges = bj.load(
        Path("~/r/clip_ids/clip_id_to_frame_ranges_we_care_about_for_segmentation.json5").expanduser()
    )
    frame_ranges = clip_id_to_frame_ranges[clip_id]["frame_ranges"]

    return frame_ranges
