from print_yellow import (
     print_yellow
)
from open_a_grayscale_png_barfing_if_it_is_not_grayscale import (
     open_a_grayscale_png_barfing_if_it_is_not_grayscale
)
from print_red import (
     print_red
)
from infer_arbitrary_frames import (
     infer_arbitrary_frames
)
from get_video_frame_path_from_clip_id_and_frame_index import (
     get_video_frame_path_from_clip_id_and_frame_index
)
from pathlib import Path
from typing import List, Tuple


def m94y_infer_clip_id_frame_index_pairs_under_this_model(
    final_model_id: str,
    clip_id_frame_index_pairs: List[Tuple[str, int]],
):
    list_of_input_output_file_paths = []
    for clip_id, frame_index in clip_id_frame_index_pairs:
        input_file_path = get_video_frame_path_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index
        )
        output_file_path = Path(f"/shared/inferences/{clip_id}_{frame_index:06d}_{final_model_id}.png")
        if output_file_path.exists():
            valid = False
            try:
                open_a_grayscale_png_barfing_if_it_is_not_grayscale(output_file_path)
                valid = True
            except Exception as e:
                print_red(f"Output file {output_file_path} is not a valid grayscale PNG: {e}")
            
            if valid:
                print_yellow(f"Skipping {output_file_path} because it already exists as a valid grayscale PNG.")
                continue
        
        list_of_input_output_file_paths.append((input_file_path, output_file_path))

    infer_arbitrary_frames(
        final_model_id=final_model_id,
        list_of_input_and_output_file_paths=list_of_input_output_file_paths,
    )
