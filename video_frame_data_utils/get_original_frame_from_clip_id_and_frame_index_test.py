from prii import (
     prii
)
from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)

import numpy as np


def test_get_original_frame_from_clip_id_and_frame_index_1():
    clip_id = "brewcub"
    frame_index = 23094
    rgb_hwc_np_u8 = get_original_frame_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )
    assert isinstance(rgb_hwc_np_u8, np.ndarray)
    assert rgb_hwc_np_u8.dtype == np.uint8
    assert rgb_hwc_np_u8.shape[0] == 1080
    assert rgb_hwc_np_u8.shape[1] == 1920
    assert rgb_hwc_np_u8.shape[2] == 3
    prii(rgb_hwc_np_u8)


if __name__ == "__main__":
    test_get_original_frame_from_clip_id_and_frame_index_1()
    print("get_original_frame_from_clip_id_and_frame_index has passed all tests.")