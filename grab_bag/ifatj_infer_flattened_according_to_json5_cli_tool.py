from pathlib import Path
import textwrap
import os
import argparse
import better_json as bj
from infer_flattened_clip_id_from_a_to_b import (
     infer_flattened_clip_id_from_a_to_b
)


def ifatj_infer_flattened_according_to_json5_cli_tool():
    final_model_id = os.environ["m"]

    argp = argparse.ArgumentParser(
        description="",
        usage=textwrap.dedent(
            """
            export m=notvisiblelabelledblack1730
            ifatj_infer_flattened_according_to_json5 fr.json5
            """
        )
    )
    argp.add_argument(
        "json_file_path",
        help="The json5 file describing which (clip_id, a, b) frame range triplets to flatten"
    )
    
    # TODO: Make configurable?
    rip_height = 256
    rip_width = 4268


    args = argp.parse_args()
    json_file_path = Path(args.json_file_path).resolve()
    
    assert json_file_path.exists(),f"{json_file_path=} does not exist"

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
    
    for triplet in triplets:
        clip_id = triplet[0]
        first_frame_index = triplet[1]
        last_frame_index = triplet[2]

        infer_flattened_clip_id_from_a_to_b(
            final_model_id=final_model_id,
            clip_id=clip_id,
            first_frame_index=first_frame_index,
            last_frame_index=last_frame_index,
            rip_height=rip_height,
            rip_width=rip_width,
            board_id="board0",
        )
