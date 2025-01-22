from instruct_how_to_download_file_to_laptop import (
     instruct_how_to_download_file_to_laptop
)
from make_evaluation_video import (
     make_evaluation_video
)
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from pathlib import Path
import argparse


def mev_make_evaluation_video_cli_tool():
    """
    To run make_evaluation_video from the command line.
    """
    argp = argparse.ArgumentParser(description="used to evaluate the performance of a segmentations model by video")
    argp.add_argument("--frames_dir", type=Path, required=True, help="the directory where the original video frames are.")
    argp.add_argument("--original_suffix", type=str, required=True, help="like .jpg or _original.png")
    argp.add_argument("--masks_dir", type=Path, required=True, help="the directory where the grayscale .png masks are")
    argp.add_argument("--clip_id", type=str, required=True, help="the clip_id that forms part of the file names like asmczv or busch4k or chicago4k or gsw")
    argp.add_argument("--first_frame_index", type=int, required=True, help="")                               
    argp.add_argument("--last_frame_index", type=int, required=True, help="")
    argp.add_argument("--model_id", type=str, required=True, help="the directory to write the extracted frames to")
    argp.add_argument("--fps", type=float, required=True, help="the frames per second of the video")
    argp.add_argument('--background', default=False, action='store_true')
    argp.add_argument('--fill_color', type=str, default="black", help="the color to fill the background with.  black or green")
    argp.add_argument('--out_suffix', type=str, default=None, required=False, help="oftentimes used to specify the ad")
    argp.add_argument('--subdir', type=str, default=None, required=False)
    opt = argp.parse_args()

    frames_dir = Path(opt.frames_dir).resolve()
    masks_dir = Path(opt.masks_dir).resolve()

    first_frame_index = opt.first_frame_index
    last_frame_index = opt.last_frame_index
    original_suffix = opt.original_suffix
    clip_id = opt.clip_id
    model_id = opt.model_id
    fps = opt.fps
    out_suffix = opt.out_suffix
    subdir = opt.subdir
    fill_color = opt.fill_color

    assert fps in [25, 50, 59.94, 29.97], f"ERROR: fps must be one of [50, 59.94, 29.97] but was {fps}"

    what_is_normal_colors = ["foreground"]
    
    if opt.background:
        what_is_normal_colors=["background"]
    
    shared_dir = get_the_large_capacity_shared_directory()
    output_dir = shared_dir / "show_n_tell"

    assert (
        output_dir.is_dir()
    ), f"ERROR: output_dir {output_dir} is not an extant directory!  This program does not make directories. You will have to make it yourself."


    for what_is_normal_color in what_is_normal_colors:
        print(f"Making {what_is_normal_color} to green...")

        if subdir is not None:
            true_output_dir = output_dir / subdir
        else:
            true_output_dir = output_dir

        if out_suffix is None:
            out_video_file_path = true_output_dir / f"{clip_id}_from_{first_frame_index}_to_{last_frame_index}_{model_id}_{what_is_normal_color}_fill{fill_color}.mp4"
        elif out_suffix is not None:
            out_video_file_path = true_output_dir / f"{clip_id}_from_{first_frame_index}_to_{last_frame_index}_{model_id}_{what_is_normal_color}_fill{fill_color}_{out_suffix}.mp4"

        make_evaluation_video(
            original_suffix=original_suffix,
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            first_frame_index=first_frame_index,
            last_frame_index=last_frame_index,
            clip_id=clip_id,
            model_id=model_id,
            fps=fps,
            what_is_normal_color=what_is_normal_color,
            fill_color=fill_color,
            out_video_file_path=out_video_file_path
        )

        instruct_how_to_download_file_to_laptop(
            file_path=out_video_file_path
        )
