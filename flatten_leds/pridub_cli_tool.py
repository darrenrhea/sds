from edub_extract_discovered_unoccluded_background import (
     edub_extract_discovered_unoccluded_background
)
import argparse


def pridub_cli_tool():
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


    edub_extract_discovered_unoccluded_background(
        clip_id=clip_id,
        frame_index=frame_index,
        rip_height=rip_height,
        rip_width=rip_width,
        min_width=512,
    )


