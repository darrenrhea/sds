
from pathlib import Path

import better_json as bj


def get_clip_ranges_from_json_file_path(
    json_file_path: Path
):
    """
    When evaluating segmentation models, we want to make small movies.
    You can specify one small movie with a clip_id and a frame range.
    This function reads a json file that contains a list of small movies.
    [
        ["bos-dal-2024-06-09-mxf", 434554, 435075],
        ["bos-dal-2024-06-09-mxf", 558167, 558652],
        ["bos-mia-2024-04-21-mxf", 366400, 367300],
        ["bos-mia-2024-04-21-mxf", 440413, 442520],
        ["bos-mia-2024-04-21-mxf", 467900, 468100],
        ["bos-mia-2024-04-21-mxf", 632567, 632838],
        ["bos-mia-2024-04-21-mxf", 737936, 738022],
    ]
    """
    assert json_file_path.exists(), f"{json_file_path=} does not exist"

    triplets = bj.load(json_file_path)
    assert isinstance(triplets, list), f"{triplets=} is not a list"

    for triplet in triplets:
        assert isinstance(triplet, list), f"{triplet=} is not a list"
        assert len(triplet) == 3, f"{triplet=} does not have 3 elements"
        clip_id = triplet[0]
        first_frame_index = triplet[1]
        last_frame_index = triplet[2]
        assert isinstance(clip_id, str), f"{clip_id=} is not a str"
        assert isinstance(first_frame_index, int), f"{first_frame_index=} is not an int"
        assert isinstance(last_frame_index, int), f"{last_frame_index=} is not an int"
        assert first_frame_index <= last_frame_index, f"{first_frame_index=} is not less than {last_frame_index=}"

    return triplets


