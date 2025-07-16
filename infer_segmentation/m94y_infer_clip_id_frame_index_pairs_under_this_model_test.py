from prii import (
     prii
)
from pathlib import Path
from m94y_infer_clip_id_frame_index_pairs_under_this_model import (
     m94y_infer_clip_id_frame_index_pairs_under_this_model
)


def test_m94y_infer_clip_id_frame_index_pairs_under_this_model_1():
    final_model_id = "summerleague2025batchsize2epoch340"

    clip_id_frame_index_pairs = [
        ("sl-2025-07-10-sdi", 22086),
        ("sl-2025-07-10-sdi", 22099),
        ("slday10game1", 1000),
    ]

    m94y_infer_clip_id_frame_index_pairs_under_this_model(
        final_model_id=final_model_id,
        clip_id_frame_index_pairs=clip_id_frame_index_pairs
    )

    for clip_id, frame_index in clip_id_frame_index_pairs:
        output_file_path = Path(f"/shared/inferences/{clip_id}_{frame_index:06d}_{final_model_id}.png")
        prii(output_file_path)
 
    
if __name__ == "__main__":
    test_m94y_infer_clip_id_frame_index_pairs_under_this_model_1()
    print("test_m94y_infer_clip_id_frame_index_pairs_under_this_model_1 passed")
