from a2dc_annotate_2d_correspondences import (
     a2dc_annotate_2d_correspondences
)
import argparse

usage_str = """
Get work items via:
bat ~/r/nba_ads/slgame1.json5

Then do something like this:

a2dc_annotate_2d_correspondences 2024-SummerLeague_Courtside_2520x126_TM_STILL slgame1 1544

to annotate 2D correspondences between the ad they sent us called ~/r/nba_ads/sl/2024-SummerLeague_Courtside_2520x126_TM_STILL.png
and the video frame from the video clip with clip_id slgame1 and frame_index 1544.
"""


def a2dc_annotate_2d_correspondences_cli_tool():
    """
    Use the a2dc... command line tool to spacially align an ad image that they sent us with a video frame.
    This type of spacial alignment is a prerequisite to self-reproducing ad insertions,
    which must also be color correct, blur correct, and noise correct,
    including video codec artifact correct, and reflections and shadows correct.

    The implementation is shallow in that it is
    only a CLI tool wrapper around the real implementation a2dc_annotate_2d_correspondences.
    """
    argparser = argparse.ArgumentParser(
        description="a2dc is for annotating/registering 2D-2D correspondences between an ad they sent us and a video frame.",
        usage=usage_str,
    )
    argparser.add_argument(
        "ad_id",
        type=str,
        help="The ad_id of the LED image",
    )
    argparser.add_argument(
        "clip_id",
        type=str,
        help="The clip_id of the video frame you are aligning with the ad.",
    )
    argparser.add_argument(
        "frame_index",
        type=int,
        help="The frame_index of the video frame you are aligning with the ad.",
    )
    args = argparser.parse_args()

    ad_id = args.ad_id
    clip_id = args.clip_id
    frame_index = args.frame_index

    a2dc_annotate_2d_correspondences(
        ad_id=ad_id,
        clip_id=clip_id,
        frame_index=frame_index,
    )
    