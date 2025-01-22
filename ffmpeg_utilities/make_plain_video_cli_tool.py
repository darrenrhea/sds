from make_plain_video import (
     make_plain_video
)

from pathlib import Path
import argparse


def make_plain_video_cli_tool():
    """
    To run make_evaluation_video from the command line.
    """
    argp = argparse.ArgumentParser(description="used to evaluate the performance of a segmentations model by video")
    argp.add_argument("--frames_dir", type=Path, required=True, help="the directory where the original .jpg video frames are")
    argp.add_argument("--clip_id", type=str, required=True, help="the clip_id that forms part of the file names like asmczv or busch4k or chicago4k or gsw")
    argp.add_argument("--original_suffix", type=str, required=True, help=".jpg or .png or _original.png or _original.jpg")
    argp.add_argument("--first_frame_index", type=int, required=True, help="")                               
    argp.add_argument("--last_frame_index", type=int, required=True, help="")
    argp.add_argument("--fps", type=float, required=True, help="the frames per second of the video")
    argp.add_argument("--out", type=Path, required=True, help="where to dump it to")
    # TODO: add an argument for the explicit out_file_path
    opt = argp.parse_args()
    original_suffix = opt.original_suffix
    assert original_suffix in [".jpg", ".png", "_original.png", "_original.jpg"], f"ERROR: original_suffix must be one of [.jpg, .png, _original.png, original.jpg] but was {original_suffix}"
    frames_dir = Path(opt.frames_dir).resolve()

    first_frame_index = opt.first_frame_index
    last_frame_index = opt.last_frame_index
    clip_id = opt.clip_id
    fps = opt.fps
    out_video_file_path = Path(opt.out).resolve()

    assert fps in [50, 59.94, 29.97], f"ERROR: fps must be one of [50, 59.94, 29.97] but was {fps}"

    make_plain_video(
        original_suffix=original_suffix,
        frames_dir=frames_dir,
        first_frame_index=first_frame_index,
        last_frame_index=last_frame_index,
        clip_id=clip_id,
        fps=fps,
        out_video_file_path=out_video_file_path,
    )