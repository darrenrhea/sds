from pathlib import Path
from make_frame_ranges_file import (
     make_frame_ranges_file
)
from infer_from_id import (
     infer_from_id
)
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from typing import List, Tuple


def infer_arbitrary_frames(
    final_model_id: str,
    list_of_input_and_output_file_paths:List[Tuple[Path, Path]]
):
    """
    Given a final_model_id
    give a list of input file paths and corresponding output file paths
    and it will infer that model on each input path and write the grayscale to the corresponding output path.
    """
    if len(list_of_input_and_output_file_paths) == 0:
        print("No frames to infer, so skipping.")
        return
    
    frame_ranges_file_path = make_frame_ranges_file(
        list_of_input_and_output_file_paths=list_of_input_and_output_file_paths,
        clip_mother_dir=None,
        clip_id=None,
        original_suffix=None,
        frame_ranges=None,
    )

    shared_dir = get_the_large_capacity_shared_directory()
    output_dir = shared_dir / "inferences"
    output_dir.mkdir(exist_ok=True)

    infer_from_id(
        final_model_id=final_model_id,
        model_id_suffix=final_model_id,
        frame_ranges_file_path=frame_ranges_file_path,
        output_dir=output_dir
    )
  
