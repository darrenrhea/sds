from get_flattened_original_hwc_np_u8 import (
     get_flattened_original_hwc_np_u8
)
from prii import (
     prii
)


def test_get_flattened_original_hwc_np_u8_1():
    
    
    clip_id = "brewcub"
    rip_width = 1024
    rip_height = 256

    frame_indices = [
        23094,
        88369
    ]
    for frame_index in frame_indices:
        for board_id in ["left", "right"]:
            rgb_hwc_np_u8 = get_flattened_original_hwc_np_u8(
                clip_id=clip_id,
                frame_index=frame_index,
                board_id=board_id,
                rip_width=rip_width,
                rip_height=rip_height,
            )
            
            prii(rgb_hwc_np_u8)

if __name__ == "__main__":
    test_get_flattened_original_hwc_np_u8_1()