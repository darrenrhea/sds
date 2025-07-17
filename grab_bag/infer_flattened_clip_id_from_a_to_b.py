from get_flat_mask_path import (
     get_flat_mask_path
)
from get_flat_original_path import (
     get_flat_original_path
)
from pathlib import Path
import pprint as pp
from infer_arbitrary_frames import (
     infer_arbitrary_frames
)


def infer_flattened_clip_id_from_a_to_b(
    final_model_id: str,
    clip_id: str,
    first_frame_index: int,
    last_frame_index: int,
    rip_height: int,
    rip_width: int,
    board_id: str,
):
    """
    Infers the already staged flattened frames for a clip_id from first_frame_index to last_frame_index.
    Hopefully the notion of "staged" changes to s3 soon.
    """
     
    output_dir = Path(
        f"/shared/clips/{clip_id}/inferred_flattenings/{rip_width}x{rip_height}/"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    list_of_input_and_output_file_paths = []
    for frame_index in range(first_frame_index, last_frame_index + 1):

        input_path = get_flat_original_path(
            clip_id=clip_id,
            frame_index=frame_index,
            board_id=board_id,
            rip_width=rip_width,
            rip_height=rip_height,
        )

        assert input_path.exists(), f"{input_path=} does not exist"

        output_path = get_flat_mask_path(
            final_model_id=final_model_id,
            clip_id=clip_id,
            frame_index=frame_index,
            board_id="board0",
            rip_width=rip_width,
            rip_height=rip_height,
        )
        if frame_index == first_frame_index:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.exists():
            print(f"Skipping {output_path=!s} because it already exists")
            continue
        list_of_input_and_output_file_paths.append(
            (input_path, output_path)
        )
    
    pp.pprint(list_of_input_and_output_file_paths)
    if len(list_of_input_and_output_file_paths) == 0:
        print(f"No frames which are not yet inferred for {clip_id=} from {first_frame_index=} to {last_frame_index=}, so skipping.")
        return

    infer_arbitrary_frames(
        final_model_id=final_model_id,
        list_of_input_and_output_file_paths=list_of_input_and_output_file_paths
    )
