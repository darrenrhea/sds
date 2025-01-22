from get_mask_hw_np_u8_from_clip_id_and_frame_index_and_convention_and_final_model_id import (
     get_mask_hw_np_u8_from_clip_id_and_frame_index_and_convention_and_final_model_id
)
from prii import (
     prii
)
import numpy as np


def test_get_mask_hw_np_u8_from_clip_id_and_frame_index_and_convention_and_final_model_id_1():
    clip_id = "bos-mia-2024-04-21-mxf"
    frame_index = 734500
    segmentation_convention = "led"
    final_model_id = "human"

    mask_hw_np_u8 = get_mask_hw_np_u8_from_clip_id_and_frame_index_and_convention_and_final_model_id(
        clip_id=clip_id,
        frame_index=frame_index,
        segmentation_convention=segmentation_convention,
        final_model_id=final_model_id,
    )
    assert mask_hw_np_u8 is not None
    assert isinstance(mask_hw_np_u8, np.ndarray)
    prii(mask_hw_np_u8)


if __name__ == "__main__":
    test_get_mask_hw_np_u8_from_clip_id_and_frame_index_and_convention_and_final_model_id_1()
    print("get_mask_hw_np_u8_from_clip_id_and_frame_index_and_convention_and_final_model_id has passed all tests.")