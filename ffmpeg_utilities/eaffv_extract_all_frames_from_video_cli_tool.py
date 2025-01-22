import textwrap
from extract_all_frames_from_video import (
     extract_all_frames_from_video
)
from pathlib import Path
import sys
import argparse
from colorama import Fore, Style


def extract_all_frames_from_video_cli_tool(): 
    """
    eaffv_extract_all_frames_from_video

    TODO: now that the clip_ids repo is a thing, this should use that
    """
    argp = argparse.ArgumentParser()
    argp.add_argument("--input", type=Path, required=True, help="Absolute path to the video file.")
    argp.add_argument("--clip_id", type=str, required=True, help="The clip id.")
    argp.add_argument("--output", required=True, type=Path, help="Absolute path to the output directory.")
    argp.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Don't ask for confirmation, just do it.",
    )
    argp.add_argument(
        "--pix_fmt",
        type=str,
        default="yuvj422p",
        help=textwrap.dedent(
            """\
            The pixel format to write out.
            For jpg, must be yuvj422p (default if you don't specify) or yuvj420p.
            For png, it must be rgb24 (i.e. 8 bits for each of r g and b, default) or rgb48be (big endian 16 bits per r g and b channel
            """
        )
    )
    argp.add_argument("--original_suffix", type=str, default="_original.jpg", help="The original suffix.")

    opt = argp.parse_args()
    input_video_abs_file_path = opt.input
    out_dir_abs_path = opt.output

    input_video_abs_file_path = Path(input_video_abs_file_path).resolve()
    clip_id = opt.clip_id
    out_dir_abs_path = Path(out_dir_abs_path).resolve()
    dont_ask_just_do_it = opt.yes
    pix_fmt = opt.pix_fmt
    original_suffix = opt.original_suffix
    original_extension = original_suffix[-4:]
    if pix_fmt is None:
        if original_extension == ".jpg":
            pix_fmt = "yuvj422p"
        elif original_extension == ".png":
            pix_fmt = "rgb24"
    
    if original_extension == ".png":
        assert opt.pix_fmt in ["rgb24", "rgb48be"], f"ERROR: {opt.pix_fmt=} is not a valid pix_fmt. Must be one of ['rgb24', 'rgb48be']"
    elif original_extension == ".jpg":
        assert opt.pix_fmt in ["yuvj422p", "yuvj420p"], f"ERROR: {opt.pix_fmt=} is not a valid pix_fmt. Must be one of ['yuvj422p', 'yuvj420p']"
    else:
        raise ValueError(f"ERROR: {original_extension=} is not a valid original_extension. Must be one of ['.jpg', '.png']")
    print(f"{Fore.YELLOW}{dont_ask_just_do_it=}{Style.RESET_ALL}")
    
    out_dir_abs_path.mkdir(exist_ok=True, parents=True)

    if not dont_ask_just_do_it:
        yes = input(
            textwrap.dedent(
                f"""\
                
                About to extract all frames from the video file:

                    {input_video_abs_file_path}
                
                as {pix_fmt=}

                named via the format string:
                
                {clip_id}_%06d{original_suffix}
                to:
                
                    {out_dir_abs_path}
                Press RETURN to continue, or anything something else to exit.



                """
            )
        )
    else:
        yes = ""

    if yes != "":
        print("Exiting")
        sys.exit(1)
    
    extract_all_frames_from_video(
        input_video_abs_file_path=input_video_abs_file_path,
        out_dir_abs_path=out_dir_abs_path,
        clip_id=clip_id,
        pix_fmt=pix_fmt,
        original_suffix=original_suffix,
    )
    
