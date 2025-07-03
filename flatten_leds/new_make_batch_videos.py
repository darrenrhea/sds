from get_mother_dir_of_frames_dir_from_clip_id import (
     get_mother_dir_of_frames_dir_from_clip_id
)
import argparse
import os
from instruct_how_to_download_file_to_laptop import (
     instruct_how_to_download_file_to_laptop
)
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from make_evaluation_video import (
     make_evaluation_video
)
from pathlib import Path
import better_json as bj



def new_make_batch_videos():
    final_model_id = os.environ["m"]
    
    argp = argparse.ArgumentParser()

    argp.add_argument(
        "json_file_path",
        help="The json5 file describing which (clip_id, a, b) frame ranges to flatten"
    )
    argp.add_argument(
        "--fps",
        type=float,
        default=59.94,
        help="how many frames per second to make the video at, default 59.94"
    )
    
    args = argp.parse_args()
    json_file_path = Path(args.json_file_path).resolve()
    
    assert json_file_path.exists(),f"{json_file_path=} does not exist"

    triplets = bj.load(json_file_path)
    assert isinstance(triplets, list), f"{triplets=} is not a list"

    for triplet in triplets:
        assert isinstance(triplet, list), f"{triplet=} is not a list"
        assert len(triplet) == 3, f"{triplet=} does not have 3 elements"
        clip_id = triplet[0]
        first_frame_index = triplet[1]
        last_frame_index = triplet[2]
        assert isinstance(clip_id, str), f"{clip_id=} is not a str"
        assert isinstance(first_frame_index, int), f"{first_frame_index=} is not an int"
        assert isinstance(last_frame_index, int), f"{last_frame_index=} is not an int"
        assert first_frame_index <= last_frame_index, f"{first_frame_index=} is not less than {last_frame_index=}"

    shared_dir = get_the_large_capacity_shared_directory()
    # shared_dir = "/hd2"

    show_n_tell_path = Path(shared_dir).expanduser() / "show_n_tell"
    show_n_tell_clips_path = show_n_tell_path / f"{final_model_id}"
    show_n_tell_clips_path.mkdir(exist_ok=True)

    # fps = get_frame_rate_from_clip_id(
    #     clip_id=triplet[0]
    # )
    fps = args.fps
    assert fps in [25, 50, 59.94, 29.97], f"fps must be one of [25, 50, 59.94, 29.97] but was {fps}"
    out_video_file_paths = []
    for triplet in triplets:
        print(f"Making video for {triplet=}")
        clip_id = triplet[0]
        first_frame_index = triplet[1]
        last_frame_index = triplet[2]

        original_suffix = "_original.jpg"
        mother_dir_of_frames_dir = get_mother_dir_of_frames_dir_from_clip_id(
            clip_id=clip_id
        )
        frames_dir = mother_dir_of_frames_dir / "clips" / clip_id / "frames"
        masks_dir = shared_dir / "inferences"

        what_is_normal_color = "foreground"
        # what_is_normal_color = "background"
        fill_color = "green"
        
        out_video_file_path = shared_dir / "show_n_tell" / f"{clip_id}_from_{first_frame_index}_to_{last_frame_index}_{final_model_id}_{what_is_normal_color}_fill{fill_color}.mp4"


        make_evaluation_video(
            original_suffix=original_suffix,
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            first_frame_index=first_frame_index,
            last_frame_index=last_frame_index,
            clip_id=clip_id,
            model_id=final_model_id,
            fps=fps,
            what_is_normal_color=what_is_normal_color,
            fill_color=fill_color,
            out_video_file_path=out_video_file_path
        )

        out_video_file_paths.append(out_video_file_path)


        instruct_how_to_download_file_to_laptop(
            file_path=out_video_file_path
        )
    
    # repeat all the downloadable things at the end:
    
    print("Here is a repeat of all the downloadables")

    for out_video_file_path in out_video_file_paths:
        instruct_how_to_download_file_to_laptop(
            file_path=out_video_file_path
        )
            

if __name__ == "__main__":
    new_make_batch_videos()