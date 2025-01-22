from concatenate_videos import (
     concatenate_videos
)
from instruct_how_to_download_file_to_laptop import (
     instruct_how_to_download_file_to_laptop
)
from write_rgb_hwc_np_u8_to_png import (
     write_rgb_hwc_np_u8_to_png
)
from make_plain_video import (
     make_plain_video
)
import argparse
from mooss_make_original_over_segmentations_stack import (
     mooss_make_original_over_segmentations_stack
)
from get_clip_ranges_from_json_file_path import (
     get_clip_ranges_from_json_file_path
)
import os
from pathlib import Path


def mfvs_make_flat_video_stack():
    just_concatenate_videos_that_already_exist = False
    
    argp = argparse.ArgumentParser()

    argp.add_argument(
        "json_file_path",
        help="The json5 file describing which (clip_id, a, b) frame ranges to make stack movies from."
    )

    final_model_ids = [
            "effl280",
            "ltwo220",
    ]
    
    args = argp.parse_args()
     # part of the filename the resulting video will be saved as, to explain the stack order:
    suffix_str = "_".join(final_model_ids)

    json_file_path = Path(args.json_file_path).resolve()
    fps = 59.94  # TODO: make this an argument?
    
    assert json_file_path.exists(), f"{json_file_path=} does not exist"

    triplets = get_clip_ranges_from_json_file_path(json_file_path)

    rip_height = 256
    rip_width = 1856
    board_id = "board0"

    out_video_file_paths = []
    for triplet in triplets:
        clip_id = triplet[0]
        first_frame_index = triplet[1]
        last_frame_index = triplet[2]
        
       
        out_video_file_path = Path(
            f"/shared/show_n_tell/{clip_id}_{first_frame_index:06d}_{last_frame_index:06d}_{suffix_str}.mp4"
        )
        
        out_dir = Path(f"/shared/clips/{clip_id}/composition_temp_dir")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_video_file_paths.append(out_video_file_path)

        if not just_concatenate_videos_that_already_exist:
            for frame_index in range(first_frame_index, last_frame_index+1): 
                stack_hwc_np_u8 = mooss_make_original_over_segmentations_stack(
                    clip_id=clip_id,
                    frame_index=frame_index,
                    board_id=board_id,
                    final_model_ids=final_model_ids,
                    rip_height=rip_height,
                    rip_width=rip_width,
                    color=(0, 255, 0),
                )

                out_abs_file_path = out_dir / f"{clip_id}_{frame_index:06d}_stack.png"
                
                write_rgb_hwc_np_u8_to_png(
                    rgb_hwc_np_u8=stack_hwc_np_u8,
                    out_abs_file_path=out_abs_file_path,
                    verbose=True
                )

            make_plain_video(
                original_suffix="_stack.png",
                frames_dir=out_dir,
                first_frame_index=first_frame_index,
                last_frame_index=last_frame_index,
                clip_id=clip_id,
                fps=fps,
                out_video_file_path=out_video_file_path
            )
    
    # repeat all the downloadable things at the end:
    for out_video_file_path in out_video_file_paths:
        instruct_how_to_download_file_to_laptop(
            file_path=out_video_file_path
        )
    
    concatenated_file_path = Path(f"/shared/show_n_tell/original_{suffix_str}_ensemble_concatenated.mp4")

    concatenate_videos(
        video_abs_file_paths=out_video_file_paths,
        out_video_abs_file_path=concatenated_file_path,
        verbose=True
    )

    instruct_how_to_download_file_to_laptop(
        file_path=concatenated_file_path
    )


if __name__ == "__main__":
    mfvs_make_flat_video_stack()