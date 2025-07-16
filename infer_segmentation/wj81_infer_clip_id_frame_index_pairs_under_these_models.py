from m94y_infer_clip_id_frame_index_pairs_under_this_model import (
     m94y_infer_clip_id_frame_index_pairs_under_this_model
)

from typing import List, Tuple

def wj81_infer_clip_id_frame_index_pairs_under_these_models(
    final_model_ids: List[str],
    clip_id_frame_index_pairs: List[Tuple[str, int]],
):
    """
    This function takes a list of (clip_id, frame_index) pairs and infers the frames
    using the specified models.
    """
    for final_model_id in final_model_ids:
        m94y_infer_clip_id_frame_index_pairs_under_this_model(
            final_model_id=final_model_id,
            clip_id_frame_index_pairs=clip_id_frame_index_pairs,
        )
       