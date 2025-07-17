from instruct_how_to_download_file_to_laptop import (
     instruct_how_to_download_file_to_laptop
)
from get_mother_dir_of_frames_dir_from_clip_id import (
     get_mother_dir_of_frames_dir_from_clip_id
)
from make_evaluation_video import (
     make_evaluation_video
)
from open_a_grayscale_png_barfing_if_it_is_not_grayscale import (
     open_a_grayscale_png_barfing_if_it_is_not_grayscale
)
from write_grayscale_hw_np_u8_to_png import (
     write_grayscale_hw_np_u8_to_png
)
import numpy as np
from print_green import (
     print_green
)
from pathlib import Path
from load_json_file import (
     load_json_file
)
from color_print_json import (
     color_print_json
)

def ensemble_models(
    clip_id: str,
    frame_index: int,
    final_model_ids: list[str],
    out_final_model_id: str,
):
    total = np.zeros((1080, 1920), dtype=np.float32)
    for final_model_id in final_model_ids:
        mask_path = Path(f"/shared/inferences/{clip_id}_{frame_index:06d}_{final_model_id}.png")
        assert mask_path.exists(), f"Mask path {mask_path} does not exist."
        hw_u8 = open_a_grayscale_png_barfing_if_it_is_not_grayscale(
            mask_path
        )
        total += hw_u8
    total /= len(final_model_ids)
    total_u8 = np.clip(np.round(total), 0, 255).astype(np.uint8)
    out_mask_path = Path(f"/shared/inferences/{clip_id}_{frame_index:06d}_{out_final_model_id}.png")
    verbose = (frame_index % 30 == 0)
    write_grayscale_hw_np_u8_to_png(
        grayscale_hw_np_u8=total_u8,
        out_abs_file_path=out_mask_path,
        verbose=verbose
    )

def ensemble_models_over_range_of_frames(
    final_model_ids: list[str],
    out_final_model_id: str,
    clip_id: str,
    first_frame_index: int,
    last_frame_index: int
):
    print_green(f"Doing {clip_id} from {first_frame_index} to {last_frame_index}")
    for frame_index in range(first_frame_index, last_frame_index+1):
        ensemble_models(
            clip_id=clip_id,
            frame_index=frame_index,
            final_model_ids=final_model_ids,
            out_final_model_id=out_final_model_id
        )

obj = load_json_file(
    Path("~/r/frame_attributes/summer_league_evaluationvideos.json5").expanduser()
)
color_print_json(obj)
clip_id, first_frame_index, last_frame_index = obj[0]
final_model_ids = [
    "summerleague2025restart1epoch720",
    "summerleague2025restart2epoch720",
    "summerleague2025batchsize2epoch700",
    # "summerleague2025batchsize2epoch340",
]
out_final_model_id = "ensemble"

# ensemble_models_over_range_of_frames(
#     final_model_ids=final_model_ids,
#     out_final_model_id=out_final_model_id,
#     clip_id=clip_id,
#     first_frame_index=first_frame_index,
#     last_frame_index=last_frame_index,
# )

print_green("Ensembled masks saved. Now making video.")

shared_dir = Path("/shared")
what_is_normal_color = "foreground"
fill_color = "green"
out_video_file_path = shared_dir / "show_n_tell" / f"{clip_id}_from_{first_frame_index}_to_{last_frame_index}_{out_final_model_id}_{what_is_normal_color}_fill{fill_color}.mp4"

fps = 59.94
original_suffix = "_original.jpg"
mother_dir_of_frames_dir = get_mother_dir_of_frames_dir_from_clip_id(
    clip_id=clip_id
)
frames_dir = mother_dir_of_frames_dir / "clips" / clip_id / "frames"
masks_dir = shared_dir / "inferences"
draw_frame_numbers = True

make_evaluation_video(
    original_suffix=original_suffix,
    frames_dir=frames_dir,
    masks_dir=masks_dir,
    first_frame_index=first_frame_index,
    last_frame_index=last_frame_index,
    clip_id=clip_id,
    model_id=out_final_model_id,
    fps=fps,
    what_is_normal_color=what_is_normal_color,
    fill_color=fill_color,
    out_video_file_path=out_video_file_path,
    draw_frame_numbers=draw_frame_numbers,
)


instruct_how_to_download_file_to_laptop(
    file_path=out_video_file_path
)