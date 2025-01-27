from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)
from prii import (
     prii
)
import argparse
import textwrap



def priframe_cli_tool():
    """
    For iterm2 users,
    print the video frame in the terminal.
    """

    argument_parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """\
            Shows a particular video frame of a video.
            """
        ),
        usage=textwrap.dedent(
            """\
            priframe <clip_id> <frame_index>
            """
        )
    )

    argument_parser.add_argument(
        "clip_id",
        type=str,
        help="The clip_id of the video."
    )

    argument_parser.add_argument(
        "frame_index_str",
        type=str,
        help="The frame_index you want to show."
    )

    args = argument_parser.parse_args()
    frame_index_str = args.frame_index_str
    clip_id = args.clip_id

    frame_index = int(frame_index_str)

    rgb_hwc_np_u8 = get_original_frame_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index
    )

    print(f"Showing frame {clip_id}_{frame_index:06d}_original:")
    prii(rgb_hwc_np_u8)


