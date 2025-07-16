from make_rgba_from_original_and_mask_paths import (
     make_rgba_from_original_and_mask_paths
)
from get_video_frame_path_from_clip_id_and_frame_index import (
     get_video_frame_path_from_clip_id_and_frame_index
)
from print_green import (
     print_green
)
from wj81_infer_clip_id_frame_index_pairs_under_these_models import (
     wj81_infer_clip_id_frame_index_pairs_under_these_models
)
from prii import (
     prii
)
from pathlib import Path
import better_json as bj
import argparse


def jk59_infer_clip_id_frame_index_pairs_under_these_models_cli_tool():
    argp = argparse.ArgumentParser(
        description="Infer specific frames from a clip using specified models."
    )
    argp.add_argument(
        "-m", "--models",
        type=str,
        required=True,
        help="Path to the json5 file containing final model IDs."
    )
    argp.add_argument(
        "-f", "--frames",
        type=str,
        required=True,
        help="Path to the json5 file containing clip ID and frame index pairs."
    )

    args = argp.parse_args()
    final_model_ids_path = Path(args.models).resolve()
    frames_file_path = Path(args.frames).resolve()

    final_model_ids = bj.load(final_model_ids_path)
    clip_id_frame_index_pairs = bj.load(frames_file_path)

    wj81_infer_clip_id_frame_index_pairs_under_these_models(
        final_model_ids=final_model_ids,
        clip_id_frame_index_pairs=clip_id_frame_index_pairs
    )

    for clip_id, frame_index in clip_id_frame_index_pairs:
        print_green(f"{clip_id}_{frame_index:06d}")
        original_path = get_video_frame_path_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index
        )
        for final_model_id in final_model_ids:
            print_green(f"{final_model_id}:")
            mask_path = Path(f"/shared/inferences/{clip_id}_{frame_index:06d}_{final_model_id}.png")
           
            
            rgba = make_rgba_from_original_and_mask_paths(
                original_path=original_path,
                mask_path=mask_path,
                flip_mask=False,
                quantize=False,
            )
            prii(rgba)
    
    

