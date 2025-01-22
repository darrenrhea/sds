from prii import (
     prii
)

from fasf_flatten_a_single_frame import (
     fasf_flatten_a_single_frame
)

def test_fasf_flatten_a_single_frame_1():
    clip_id = "slday3game1"
    frame_index = 0
    rip_height = 256
    rip_width = 4268
    board_id = "board0"

    (
        flattened_rgb,
        visibility_mask
    ) = fasf_flatten_a_single_frame(
        clip_id=clip_id,
        frame_index=frame_index,
        rip_height=rip_height,
        rip_width=rip_width,
        board_id=board_id,
    )

    prii(flattened_rgb)
    prii(visibility_mask)


if __name__ == "__main__":
    test_fasf_flatten_a_single_frame_1()