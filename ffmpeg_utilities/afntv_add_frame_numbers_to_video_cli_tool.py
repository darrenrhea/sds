import textwrap
from add_frame_numbers_to_video import (
     add_frame_numbers_to_video
)
from pathlib import Path
import argparse


def afntv_add_frame_numbers_to_video_cli_tool():
    usage = textwrap.dedent(
        """\
        \n
        afntv_add_frame_numbers_to_video \\
        -i /media/drhea/muchspace/s3/awecomai-original-videos/bos-dal-2024-06-06-unaugmented_srt_fullgame.mp4 \\
        -f 0 \\
        -o /media/drhea/muchspace/s3/awecomai-original-videos/bos-dal-2024-06-06-unaugmented_srt_fullgame_with_frame_numbers.mp4


        afntv_add_frame_numbers_to_video \\
        -i /media/drhea/muchspace/s3/awecomai-test-videos/nba/Mathieu/game2-3_summer_league.ts \\
        -f 0 \\
        -o /media/drhea/muchspace/s3/awecomai-test-videos/nba/Mathieu/game2-3_summer_league.ts_with_frame_numbers.mp4
        
        """
    )
    
    argp = argparse.ArgumentParser(
        description="add frame numbers to a video, given you know the frame number of the first/0-ith frame",
        usage=usage,
    )

    argp.add_argument(
        "-i",
        "--in_video",
        type=str,
        required=True,
        help="the video file that you want to add frames numbers to"
    )

    argp.add_argument(
        "-f",
        "--first_frame_index",
        type=int,
        required=True,
        help="what integer is the first frame considered to be we usually use zero/0 for this"
    )

    argp.add_argument(
        "-o",
        "--out_video",
        type=str,
        required=False,
        help="where to save the video file that has frame numbers"
    )

    opt = argp.parse_args()
    in_video_file_path = Path(opt.in_video).resolve()
    first_frame_index = opt.first_frame_index
    if opt.out_video is None:
        out_video_file_path = in_video_file_path.parent / f"{in_video_file_path.stem}_with_frame_numbers{in_video_file_path.suffix}"
    else:
        out_video_file_path = Path(opt.out_video).resolve()
    

    add_frame_numbers_to_video(
        in_video_file_path=in_video_file_path,
        first_frame_index=first_frame_index,
        out_video_file_path=out_video_file_path,
    )
