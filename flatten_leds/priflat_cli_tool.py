from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)
import argparse
from prii import (
     prii
)

from fasf_flatten_a_single_frame import (
     fasf_flatten_a_single_frame
)

def priflat_cli_tool():
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "clip_id",
        type=str,
    )
    argp.add_argument(
        "frame_index",
        type=int,
    )
    # TODO: it looks as if you can change the rip height and width,
    # but it currently does not really work correctly.
    argp.add_argument(
        "--rip_height",
        type=int,
        required=False,
        default=256,
    )
    argp.add_argument(
        "--rip_width",
        type=int,
        required=False,
        default=2560, # used to be  4268, for summer_league_2024
    )
    opt = argp.parse_args()
    clip_id = opt.clip_id
    frame_index = opt.frame_index
    rip_height = opt.rip_height
    rip_width = opt.rip_width

    original_rgb_hwc_np_u8 = get_original_frame_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )

    prii(original_rgb_hwc_np_u8)

    (
        flattened_rgb,
        visibility_mask
    ) = fasf_flatten_a_single_frame(
        clip_id=clip_id,
        frame_index=frame_index,
        board_id="board0",
        rip_height=rip_height,
        rip_width=rip_width,
    )

    prii(flattened_rgb)
    prii(visibility_mask)

