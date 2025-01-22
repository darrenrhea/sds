import textwrap
from flatten_clip_id_from_a_to_b import (
     flatten_clip_id_from_a_to_b
)
import argparse


def sflbfi_stage_flattened_led_boards_for_inference():
    usage = textwrap.dedent(
        """\
        sflbfi_stage_flattened_led_boards_for_inference.py -c slday8game1 -a 0 -b 3000 -s 1000
        """
    )
    argp = argparse.ArgumentParser(
        description="Stage flattened LED boards, say for inference.",
        usage=usage,
    )
    argp.add_argument(
        "-c", "--clip_id",
        required=True,
        help="The clip_id to flatten."
    )
    argp.add_argument(
        "-a", "--first_frame_index",
        required=True,
        type=int,
        help="The first frame index to flatten."
    )
    argp.add_argument(
        "-b", "--last_frame_index",
        required=True,
        type=int,
        help="The last frame index to flatten."
    )
    argp.add_argument(
        "-s", "--step",
        type=int,
        required=True,
        help="The step to take between each frame."
    )

    args = argp.parse_args()
    clip_id = args.clip_id
    first_frame_index = args.first_frame_index
    last_frame_index = args.last_frame_index
    step = args.step

    rip_height = 256
    rip_width = 4268 # 1856

    flatten_clip_id_from_a_to_b(
        clip_id=clip_id,
        first_frame_index=first_frame_index,
        last_frame_index=last_frame_index,
        step=step,
        rip_height=rip_height,
        rip_width=rip_width,
    )
    

        
if __name__ == "__main__":
    sflbfi_stage_flattened_led_boards_for_inference()