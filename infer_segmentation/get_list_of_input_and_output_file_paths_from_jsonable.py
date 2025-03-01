from pathlib import Path
import better_json as bj
from typing import Any, Dict, List, Tuple


def get_list_of_input_and_output_file_paths_from_jsonable(
    jsonable: Dict | List,
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
    assert isinstance(model_id_suffix, str), "model_id_suffix should be a string"
    assert isinstance(out_dir, Path), "out_dir should be a Path"
    assert out_dir.is_dir(), f"{out_dir} is not an extant directory"


    if isinstance(jsonable, list):
        # this is a straight list of input and output file paths
        for thing in jsonable:
            assert len(thing) == 2, f"{thing=} should be a list of two absolute file paths"

        list_of_input_and_output_file_paths = []
        for input_file_path, output_file_path in jsonable:
            input_file_path = Path(input_file_path).expanduser().resolve()
            output_file_path = Path(output_file_path).expanduser().resolve()
            list_of_input_and_output_file_paths.append(
                (input_file_path, output_file_path)
            )
    else:
        assert "frame_ranges" in jsonable, f"{json_path} does not have a frame_ranges key"
        frame_ranges = jsonable["frame_ranges"]
        original_suffix = jsonable.get("original_suffix", ".jpg")

        input_dir = Path(jsonable["input_dir"])
        assert input_dir.is_dir(), f"{input_dir=} is not an extant directory"

        clip_id = jsonable["clip_id"]
        assert isinstance(clip_id, str), f"{clip_id=} should be a string"

        input_file_paths = []
        for frame_range in frame_ranges:
            if isinstance(frame_range, list):
                assert (
                    len(frame_range) in [2, 3]
                ), f"{frame_range=} should be a list of length 2 or 3, either [first, last] or [first, last, step]"
                start = frame_range[0]
                end = frame_range[1]
                if len(frame_range) == 2:
                    step = 1
                else:
                    step = frame_range[2]
            elif isinstance(frame_range, int):
                start = frame_range
                end = frame_range
                step = 1
            else:
                raise Exception(f"{frame_range=} should be a list or an int")

            assert isinstance(start, int), f"{start=} should be an int"
            assert isinstance(end, int), f"{end=} should be an int"
            assert isinstance(step, int), f"{step=} should be an int"
            assert start <= end, f"{start=} should be <= {end=}"

            for frame_number in range(start, end + 1, step):
                input_file_path = input_dir / f"{clip_id}_{frame_number:06d}{original_suffix}"
                assert input_file_path.is_file(), f"{input_file_path} is not a file"
                input_file_paths.append(input_file_path)


        list_of_input_and_output_file_paths = []
        for input_file_path in input_file_paths:
            assert input_file_path.name.endswith(original_suffix), f"All inputs should end with {original_suffix=}, but {input_file_path} does not??!!"
            annotation_id = input_file_path.name[:-len(original_suffix)]

            output_file_path = out_dir / f"{annotation_id}{model_id_suffix}.png"
            assert input_file_path.is_file(), f"{input_file_path} is not a file"

            list_of_input_and_output_file_paths.append(
                (input_file_path, output_file_path)
            )

    # check sanity:
    for input_file_path, output_file_path in list_of_input_and_output_file_paths:
        assert input_file_path.is_file(), f"{input_file_path} is not a file"
        assert (
            output_file_path.parent.is_dir()
        ), f"{output_file_path.parent} is not an extant directory, and we do not make directories for you"

    return list_of_input_and_output_file_paths

