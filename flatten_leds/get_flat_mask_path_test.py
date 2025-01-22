from get_flat_mask_path import (
     get_flat_mask_path
)


def test_get_flat_mask_path():
    clip_id = "bos-mia-2024-04-21-mxf"
    frame_index = 442105
    rip_width = 1856
    rip_height = 256
    final_model_id = "effs94"
    board_id = "board0"
    flat_mask_path = get_flat_mask_path(
        clip_id=clip_id,
        frame_index=frame_index,
        rip_width=rip_width,
        rip_height=rip_height,
        final_model_id=final_model_id,
        board_id=board_id
    )
    assert flat_mask_path.exists(), f"{flat_mask_path=} does not exist"

    assert flat_mask_path.exists()
    print(f"{flat_mask_path=}")


if __name__ == "__main__":
    test_get_flat_mask_path()
    print("get_flat_mask_path PASSED")