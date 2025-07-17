"""
It may be better to have this work on just one GPU
and then have a separate script that
determines which GPUs aren't too busy then does parallelization
by calling this as a subprocess once per available GPU
with an assignment of work
appropriate to that GPU.
"""
from pathlib import Path
from colorama import Fore, Style
from typing import List, Tuple

def make_input_output_file_path_pairs(
    model_id_suffix: str,
) -> List[Tuple[Path, Path]]:
    """
    So long as we are going to take in a frame from an image file
    and save the result as an image file, this will work.
    """

    frames_dir = Path("/media/drhea/muchspace/clips/hou-lac-2023-11-14/frames")
    out_dir = Path("/media/drhea/muchspace/inferences")
    assert frames_dir.is_dir(), f"{frames_dir} is not a directory"
    assert out_dir.is_dir(), f"{out_dir} is not a directory"

    clip_id = "hou-lac-2023-11-14"
    tuples = []
    frame_ranges = [
        (152000, 159000),
        (222000, 223000),
        (242000, 246000),
    ]

    frame_indices = []
    for first_frame_index, last_frame_index in frame_ranges:
        frame_indices += list(range(first_frame_index, last_frame_index + 1))

    for frame_index in frame_indices:
        input_image_file_path = frames_dir / f"{clip_id}_{frame_index:06d}.jpg"
        output_image_file_path = out_dir / f"{clip_id}_{frame_index:06d}{model_id_suffix}.png"
        tuples.append(
            (input_image_file_path, output_image_file_path)
        )
    return tuples
