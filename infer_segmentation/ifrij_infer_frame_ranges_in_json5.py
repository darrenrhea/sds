from get_original_path import (
     get_original_path
)
from get_flat_mask_path import (
     get_flat_mask_path
)

from pathlib import Path
import pprint as pp
import textwrap
from infer_arbitrary_frames import (
     infer_arbitrary_frames
)
import os
import argparse
import better_json as bj


def infer_clip_id_from_a_to_b(
    final_model_id: str,
    clip_id: str,
    first_frame_index: int,
    last_frame_index: int,
    step: int,
):
    """
    Infers the already staged frames of a clip_id from first_frame_index to last_frame_index.
    """

    output_dir = Path(
        "/shared/inferences"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    list_of_input_and_output_file_paths = []
    for frame_index in range(first_frame_index, last_frame_index + 1, step):

        input_path = get_original_path(
            clip_id=clip_id,
            frame_index=frame_index,
        )

        assert input_path.exists(), f"{input_path=} does not exist"

        output_path = output_dir / f"{clip_id}_{frame_index:06d}_{final_model_id}.png"
        
        # if output_path.exists():
        #     print(f"Skipping {output_path=!s} because it already exists")
        #     continue
        list_of_input_and_output_file_paths.append(
            (input_path, output_path)
        )
    
    if len(list_of_input_and_output_file_paths) == 0:
        print(f"No frames which are not yet inferred for {clip_id=} from {first_frame_index=} to {last_frame_index=}, so skipping.")
        return

    infer_arbitrary_frames(
        final_model_id=final_model_id,
        list_of_input_and_output_file_paths=list_of_input_and_output_file_paths
    )


def ifrij_infer_frame_ranges_in_json5():
    final_model_id = os.environ["m"]

    argp = argparse.ArgumentParser(
        description="",
        usage=textwrap.dedent(
            """
            You need a json5 file that is a list of 3-tuples of (clip_id, a, b) where a and b are first and last frame_indices.
            export m=notvisiblelabelledblack1730
            python ifatj_infer_flattened_according_to_json5.py fr.json5
            """
        )
    )
    argp.add_argument(
        "json_file_path",
        help="The json5 file describing which (clip_id, a, b) frame range triplets to flatten"
    )
    
   


    args = argp.parse_args()
    json_file_path = Path(args.json_file_path).resolve()
    
    assert json_file_path.exists(),f"{json_file_path=} does not exist"

    triplets = bj.load(json_file_path)
    assert isinstance(triplets, list), f"{triplets=} is not a list"

    for triplet in triplets:
        assert isinstance(triplet, list), f"{triplet=} is not a list"
        assert len(triplet) in [3, 4], f"{triplet=} does not have 3 or 4 elements"
        clip_id = triplet[0]
        first_frame_index = triplet[1]
        last_frame_index = triplet[2]
        step = triplet[3] if len(triplet) == 4 else 1
        assert isinstance(clip_id, str), f"{clip_id=} is not a str"
        assert isinstance(first_frame_index, int), f"{first_frame_index=} is not an int"
        assert isinstance(last_frame_index, int), f"{last_frame_index=} is not an int"
        assert first_frame_index <= last_frame_index, f"{first_frame_index=} is not less than {last_frame_index=}"
    
    for triplet in triplets:
        clip_id = triplet[0]
        first_frame_index = triplet[1]
        last_frame_index = triplet[2]
        step = triplet[3] if len(triplet) == 4 else 1

        infer_clip_id_from_a_to_b(
            final_model_id=final_model_id,
            clip_id=clip_id,
            first_frame_index=first_frame_index,
            last_frame_index=last_frame_index,
            step=step,
        )
    

        
if __name__ == "__main__":
    ifrij_infer_frame_ranges_in_json5()
