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


def test_wj81_infer_clip_id_frame_index_pairs_under_these_models_1():
    final_model_ids = [
        "summerleague2025batchsize2epoch340",
        "summerleague2025restart1epoch300",
        "summerleague2025restart2epoch280",
    ]

    clip_id_frame_index_pairs = [
        ("sl-2025-07-10-sdi", 22086),
        ("sl-2025-07-10-sdi", 22099),
        ("slday10game1", 1000),
    ]

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
    
    
if __name__ == "__main__":
    test_wj81_infer_clip_id_frame_index_pairs_under_these_models_1()
    print("Test completed successfully.")

