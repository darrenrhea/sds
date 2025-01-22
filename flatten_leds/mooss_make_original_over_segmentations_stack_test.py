from mooss_make_original_over_segmentations_stack import (
     mooss_make_original_over_segmentations_stack
)

from prii import prii


def test_mooss_make_original_over_segmentations_stack_1():
    final_model_ids = [
            "effs94",
            "effl150",
            "effl280",
    ]
    clip_id = "bos-dal-2024-06-09-mxf"

    stack_hwc_np_u8 = mooss_make_original_over_segmentations_stack(
        clip_id=clip_id,
        frame_index=558600,
        final_model_ids=final_model_ids,
        board_id="board0",
        rip_height = 256,
        rip_width = 1856,
        color=(255, 0, 0),
    )
    prii(stack_hwc_np_u8)


if __name__ == "__main__":
    test_mooss_make_original_over_segmentations_stack_1()