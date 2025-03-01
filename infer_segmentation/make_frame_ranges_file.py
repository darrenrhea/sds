from get_a_temp_file_path import (
     get_a_temp_file_path
)
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
import better_json as bj
from pathlib import Path
from typing import List, Tuple, Union, Optional


def make_frame_ranges_file(
    clip_id: str,
    original_suffix: str,
    frame_ranges: List[
        Union[
            int,
            Tuple[int, int],
            Tuple[int, int, int]
        ]
    ],
    list_of_input_and_output_file_paths: Optional[List[Tuple[Path, Path]]] = None,
) -> Path:
    """
    Either this is straight up given the list of input and output file paths,
    
    OR:
    Make a temporary frame ranges file for the given
    clip_id and frame_ranges.
    """
   
    frame_ranges_file_path = get_a_temp_file_path(
        suffix=".json5"
    )

    if list_of_input_and_output_file_paths is None:
        shared_dir = Path("/hd2") # get_the_large_capacity_shared_directory()
        obj = {
            "original_suffix": original_suffix,
            "input_dir": f"{shared_dir}/clips/{clip_id}/frames",
            "clip_id": clip_id,
            "frame_ranges": frame_ranges
        }
    else:
        for thing in list_of_input_and_output_file_paths:
            assert len(thing) == 2, f"{thing=} should be a list of two absolute file paths"
            input_path = thing[0]
            output_path = thing[1]
            assert input_path.is_file(), f"{input_path} should be a file"
            assert output_path.parent.is_dir(), f"{output_path.parent} is not an extant directory"
           
        obj = [
            (str(input_file_path), str(output_file_path))
            for input_file_path, output_file_path in
            list_of_input_and_output_file_paths
        ]

    bj.dump(
        fp=frame_ranges_file_path,
        obj=obj
    )
    assert frame_ranges_file_path.is_file()
    return frame_ranges_file_path

   
