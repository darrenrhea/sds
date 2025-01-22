from get_clip_ranges_from_json_file_path import (
     get_clip_ranges_from_json_file_path
)
from flatten_clip_id_from_a_to_b import (
     flatten_clip_id_from_a_to_b
)
import argparse
import better_json as bj

from pathlib import Path


def ffrfi_flatten_frame_ranges_for_inference():
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "json_file_path",
        help="The json5 file describing which (clip_id, a, b) frame ranges to flatten"
    )
    
    # TODO: Make configurable?
    rip_height = 256
    rip_width = 1856


    args = argp.parse_args()
    json_file_path = Path(args.json_file_path).resolve()
    
    triplets = get_clip_ranges_from_json_file_path(
        json_file_path=json_file_path
    )
  
    for triplet in triplets:
        clip_id = triplet[0]
        first_frame_index = triplet[1]
        last_frame_index = triplet[2]

        flatten_clip_id_from_a_to_b(
            clip_id=clip_id,
            first_frame_index=first_frame_index,
            last_frame_index=last_frame_index,
            rip_height=rip_height,
            rip_width=rip_width,
        )
    

        
if __name__ == "__main__":
    ffrfi_flatten_frame_ranges_for_inference()
