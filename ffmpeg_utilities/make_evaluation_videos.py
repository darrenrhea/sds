import better_json as bj
from pathlib import Path
import argparse
from make_evaluation_video import (
    make_evaluation_video
)

from instruct_how_to_download_file_to_laptop import (
     instruct_how_to_download_file_to_laptop
)


def make_evaluation_videos(
    clip_id: str,
    fps: float,
    model_id_suffix: str,
    frame_ranges: list
):
    assert isinstance(clip_id, str), f"ERROR: clip_id is not a string but is {type(clip_id)}"
    assert isinstance(model_id_suffix, str), f"ERROR: model_id_suffix is not a string but is {type(model_id_suffix)}"
    assert isinstance(frame_ranges, list), f"ERROR: frame_ranges is not a list but is {type(frame_ranges)}"
    for frame_range in frame_ranges:
        assert isinstance(frame_range, list), f"ERROR: frame_range is not a list but is {type(frame_range)}"
        assert len(frame_range) == 2, f"ERROR: frame_range is not of length 2 but is of length {len(frame_range)}"
        assert all(isinstance(x, int) for x in frame_range), f"ERROR: frame_range is not a list of integers but is {frame_range}"
    
    frames_dir = Path(f"/media/drhea/muchspace/clips/{clip_id}/frames")
    masks_dir = Path("/media/drhea/muchspace/inferences")
    model_id = "effs-led-238frames-1920x1088-cameraposedfake-epoch001100"
   
    
    assert fps in [50, 59.94, 29.97], f"ERROR: fps must be one of [50, 59.94, 29.97] but was {fps}"


    original_suffix = "_original.png"

    out_video_dir_path = Path(
        "/media/drhea/muchspace/show_n_tell"
    )
        
    out_video_dir_path.mkdir(exist_ok=True, parents=True)
    
    out_video_file_paths = []
    for first_frame_index, last_frame_index in frame_ranges:
        what_is_normal_color="foreground"
        out_video_file_path = out_video_dir_path / f"{clip_id}_from_{first_frame_index}_to_{last_frame_index}_{model_id}_{what_is_normal_color}.mp4"
        out_video_file_paths.append(out_video_file_path)

        make_evaluation_video(
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            first_frame_index=first_frame_index,
            last_frame_index=last_frame_index,
            clip_id=clip_id,
            model_id=model_id,
            fps=fps,
            what_is_normal_color=what_is_normal_color,
            out_video_file_path=out_video_file_path,
            original_suffix=original_suffix,
        )

        instruct_how_to_download_file_to_laptop(
            file_path=out_video_file_path
        )
    
    for video_file_path in out_video_file_paths:
        instruct_how_to_download_file_to_laptop(
            file_path=video_file_path
        )


def make_evaluation_video_cli_tool():
    """
    TODO: wrap it as a cli_tool
    """
    argp = argparse.ArgumentParser(description="used to evaluate the performance of a segmentations model by video")
    argp.add_argument("--frames_dir", type=Path, required=True, help="the directory where the original .jpg video frames are")
    argp.add_argument("--masks_dir", type=Path, required=True, help="the directory where the grayscale .png masks are")
    argp.add_argument("--clip_id", type=str, required=True, help="the clip_id that forms part of the file names like asmczv or busch4k or chicago4k or gsw")
    argp.add_argument("--first_frame_index", type=int, required=True, help="")                               
    argp.add_argument("--last_frame_index", type=int, required=True, help="")
    argp.add_argument("--model_id", type=str, required=True, help="the directory to write the extracted frames to")
    argp.add_argument("--fps", type=float, required=True, help="the frames per second of the video")
    argp.add_argument('--background', default=False, action='store_true')
    opt = argp.parse_args()
    clip_id = opt.clip_id
    model_id = opt.model_id
    fps = opt.fps

    frames_dir = Path(opt.frames_dir).resolve()
    masks_dir = Path(opt.masks_dir).resolve()


if __name__ == "__main__":
    # frame_range_to_ad_file_path = Path(
    #     f"~/r/frame_attributes/{clip_id}/frame_range_to_ad.json5"
    # ).expanduser()

    # start_stop_ad = bj.load(frame_range_to_ad_file_path)
    # assert isinstance(start_stop_ad, list), f"ERROR: start_stop_ad is not a list but is {type(start_stop_ad)}"
    
    # clip_id = "munich2024-01-25-1080i-yadif"

    clip_id = "munich2024-01-09-1080i-yadif"
    model_id_suffix = "effs-led-238frames-1920x1088-cameraposedfake-epoch001100"
    fps = 50.0
    if clip_id == "munich2024-01-25-1080i-yadif":

        frame_ranges = [
            [79000, 83000],
            [84400, 86300],
            [86700, 87500],
            [88000, 90100],
            [90500, 91300],
        ]
    
    if clip_id == "munich2024-01-09-1080i-yadif":
        frame_ranges = [
            [3600, 4300],
            [7000, 8800],
            [10400, 18500],
            [19000, 23200],
            [28800, 37000],
            [40300, 43200],
            [45000, 49600], 
        ]

    frames_dir = Path(f"/media/drhea/muchspace/clips/{clip_id}/frames")
    masks_dir = Path("/media/drhea/muchspace/inferences")

    make_evaluation_videos(
        clip_id=clip_id,
        fps=fps,
        model_id_suffix=model_id_suffix,
        frame_ranges=frame_ranges,
    )
