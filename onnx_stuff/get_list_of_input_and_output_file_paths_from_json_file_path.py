from get_list_of_input_and_output_file_paths_from_jsonable import (
     get_list_of_input_and_output_file_paths_from_jsonable
)
from pathlib import Path
import better_json as bj
from typing import List, Tuple


def get_list_of_input_and_output_file_paths_from_json_file_path(
    json_file_path: Path,
    out_dir: Path,
    model_id_suffix: str,
) -> List[Tuple[Path, Path]]:
    """
    A "frame_ranges" file can be used for inference.
    It can also be a straight list of input file output file pairs.

    All inferers will take in a list of pairs of input and output file paths
    like what this function returns.

    The list is a list of workitems, where each workitem is a 2-tuple,
    which is what image to infer on and where to write the output.

    We sometimes want to infer on just some special subset of frames,
    so we have a json file that specifies which frames to infer on.

    list_of_input_and_output_file_paths is a list of 2-tuples,
    where the 0-ith coordinate of the tuple is the file Path of
    the image to infer on, and the 1-ith coordinate if the file Path
    to write the inferred mask to.
    """
    assert isinstance(json_file_path, Path), "json_path should be a Path"
    assert json_file_path.is_file(), f"{json_file_path} is not a file"

    jsonable = bj.load(json_file_path)

    list_of_input_and_output_file_paths = \
    get_list_of_input_and_output_file_paths_from_jsonable(
        jsonable=jsonable,
        out_dir=out_dir,
        model_id_suffix=model_id_suffix
    )
    return list_of_input_and_output_file_paths
