import sys
from get_nonbroken_ffmpeg_file_path import (
     get_nonbroken_ffmpeg_file_path
)
from extract_single_frame_from_interlaced_video import (
     extract_single_frame_from_interlaced_video
)
from crazy_fps_calculation import (
     crazy_fps_calculation
)
import subprocess
from pathlib import Path
import argparse
from print_yellow import print_yellow
from print_green import print_green
from print_red import print_red


def extract_single_frame_from_video(
    input_video_abs_file_path: Path,
    deinterlace: bool,
    frame_index_to_extract: int,
    fps: float,
    pix_fmt: str,
    png_or_jpg: str,
    out_frame_abs_file_path: Path,
    verbose: bool,
):
    """
    Given a frame_index to extract (canonical, starting at 0 with first frame)
    and the absolute path to video, extracts that frame
    and saves it to out_frame_abs_file_path.
    """

    non_broken_ffmpeg_file_path = get_nonbroken_ffmpeg_file_path()
    # non_broken_ffmpeg_file_path = Path(
    #     "~/ffmpeg-7.0.1-amd64-static/ffmpeg"
    # ).expanduser()
    if deinterlace:
        return extract_single_frame_from_interlaced_video(
            input_video_abs_file_path=input_video_abs_file_path,
            frame_index_to_extract=frame_index_to_extract,
            fps=fps,
            pix_fmt=pix_fmt,
            png_or_jpg=png_or_jpg,
            out_frame_abs_file_path=out_frame_abs_file_path,
        )
    
    assert (
        png_or_jpg in ["png", "jpg"]
    ), f"ERROR: {png_or_jpg=} is not a valid png_or_jpg value, only png and jpg are valid"

    valid_pix_fmts = ["rgb48be", "rgb24", "yuvj420p", "yuvj422p"]

    assert (
         isinstance(deinterlace, bool)
    ), "ERROR: deinterlace must be a bool"

    assert (
        pix_fmt in valid_pix_fmts
    ), f"ERROR: {pix_fmt=} is not a valid pix_fmt. Must be one of {valid_pix_fmts}"

    if png_or_jpg == "png":
        assert (
            out_frame_abs_file_path.suffix == ".png"
        ), f"Confusion: you asked for png, but you asked us to save to {out_frame_abs_file_path}"

    elif png_or_jpg == "jpg":
        assert (
            pix_fmt in ["yuvj420p", "yuvj422p"]
        ), f"ERROR: {pix_fmt=} is not a valid pix_fmt for jpg. Must be one of yuvj420p or yuvj422p"

        assert (
            out_frame_abs_file_path.suffix == ".jpg"
        ), f"Confusion: you asked for jpg, but you asked us to save to {out_frame_abs_file_path}"
    else:
        raise ValueError(f"ERROR: {png_or_jpg=} is not a valid png_or_jpg value, only png and jpg are valid")

    assert(
        fps in [59.94, 29.97, 50.0, 25.0]
    ), f"ERROR: {fps} is not a likely fps"
    
    assert (
        input_video_abs_file_path.is_absolute()
    ), f"ERROR: {input_video_abs_file_path} must be absolute"

    n = frame_index_to_extract

    if n == 0:
        SS = "0"
    else:
        answer = crazy_fps_calculation(n, fps=fps)
        SS = f"{answer:.3f}"

    if png_or_jpg == "jpg":
        assert pix_fmt in ["yuvj420p", "yuvj422p"]
        format_args = [
            "-pix_fmt",
            pix_fmt,
            "-q:v",
            "1",
            "-qmin",
            "1",
            "-qmax",
            "1",
        ]
    elif png_or_jpg == "png":
        assert pix_fmt in ["rgb24", "rgb48be"]
        format_args = [
            "-pix_fmt",
            pix_fmt
        ]
       
    else:
        raise Exception()

    args = [
        str(non_broken_ffmpeg_file_path),
        "-y",
        "-nostdin",
        "-accurate_seek",
        "-ss",
        SS,
        "-i",
        str(input_video_abs_file_path),
    ]

    if deinterlace:
        args += [
            "-vf",
            #"yadif=mode=1",
            "bwdif=mode=1",  # bwdif is better than yadif
        ]
      
    
    args += [
        "-f",
        "image2",
         "-vsync",
        "0",
    ]
    
    args += format_args
    
    args += [
        "-frames:v",
        "1",
        "-update",
        "1",
        "-loglevel",
        "error",
        str(out_frame_abs_file_path)
    ]
    if verbose:
        print_green("ffmpeg command:")
        print_green(" \\\n".join(args))
        print_yellow(f"pri {out_frame_abs_file_path}")

    subprocess.run(args=args)

