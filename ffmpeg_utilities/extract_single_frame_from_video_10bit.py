import subprocess
from pathlib import Path
import numpy as np
import argparse


def extract_single_frame_from_video_10bit(
        input_video_abs_file_path,
        frame_index_to_extract,
        out_frame_abs_file_path,
        fps
):
    """
    The 10-bit version of extract_single_frame_from_video.
    Given a frame_index to extract (starting at 0)
    and the absolute path to video (better be constant frame rate),
    extracts that frame and saves it to out_frame_abs_file_path
    """
    assert fps in [59.94, 29.97, 50.0], f"ERROR: {fps} is not a valid fps"
    assert input_video_abs_file_path.is_absolute(), f"ERROR: {input_video_abs_file_path} must be absolute"
    n = frame_index_to_extract

    if n == 0:
        SS ="0"
    else:
        answer = (n - 0.25) / fps
        SS = f"{answer:.3f}"

    # startts=max(firstframe-0.25,0)/60000*1001
    # ffmpeg -y -accurate_seek -ss $startts -i $infile -map 0:v:0 -f image2 -vsync 0 -pix_fmt rgb48be -start_number $firstframe -vframes 1 ${infile}_%06d.png

    args = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-accurate_seek",
        "-ss",
        SS,
        "-i",
        str(input_video_abs_file_path),
        "-f",
        "image2",
        "-vsync",
        "0",
        "-pix_fmt",
        "rgb48be",
        "-frames:v",
        "1",
        "-update",
        "1",
        "-loglevel",
        "error",
        str(out_frame_abs_file_path)
    ]
    # print(" \\\n".join(args))
    subprocess.run(args=args)
    print(f"pri {out_frame_abs_file_path}")
