from get_nonbroken_ffmpeg_file_path import (
     get_nonbroken_ffmpeg_file_path
)
from get_font_file_path import (
     get_font_file_path
)
from pathlib import Path
import subprocess

from colorama import Fore, Style


def make_plain_video(
    original_suffix: str,
    frames_dir: Path,
    first_frame_index: int,
    last_frame_index: int,
    clip_id: str,
    fps: float,
    out_video_file_path: Path,
) -> None:  
    """
    Suppose you already have video frames blown out into a directory called
    :python:`frames_dir` saved as
    as JPEGs with file extension .jpg and naming format

    .. code-block:: python

         frames_dir / f"{clip_id}_%06d.jpg"

    Makes a plain video out of the RGB frames.
    """

    assert (
        frames_dir.is_dir()
    ), f"ERROR: frames_dir {frames_dir} is not an extant directory!"

    font_file_path = get_font_file_path()
    ffmpeg = get_nonbroken_ffmpeg_file_path()
    
    assert (
        out_video_file_path.parent.is_dir()
    ), f"ERROR: out_dir {out_video_file_path.parent} is not an extant directory!  This program does not make directories. You will have to make it yourself."

    num_frames = last_frame_index - first_frame_index + 1

    args = [
        str(ffmpeg),
        "-nostdin",
        "-y",
        "-start_number",
        str(first_frame_index),
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / f"{clip_id}_%06d{original_suffix}"),
        "-frames",
        str(num_frames),
        "-vf",
        f"drawtext=fontfile={font_file_path}: text='%{{frame_num}}': start_number={first_frame_index}: x=w-tw: y=3*lh: fontcolor=yellow: fontsize=50: box=1: boxcolor=black: boxborderw=5",
        #f"drawtext=fontfile={font_file_path}: text='%{{frame_num}}': start_number={first_frame_index}: x=0: y=3*lh: fontcolor=red: fontsize=50: box=1: boxcolor=black: boxborderw=5",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuvj420p",
        "-crf",
        "18",
        str(out_video_file_path),
    ]
    print(f"{Fore.GREEN}")
    print(" \\\n".join(args))
    print(f"{Style.RESET_ALL}")

    subprocess.run(
        args=args
    )
    
    return None
