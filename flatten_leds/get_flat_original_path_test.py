from get_flat_original_path import (
     get_flat_original_path
)


def test_get_flat_original_path():
    clip_id = "bos-mia-2024-04-21-mxf"
    frame_index = 442105
    rip_width = 1856
    rip_height = 256
    original_path = get_flat_original_path(
        clip_id=clip_id,
        frame_index=frame_index,
        rip_width=rip_width,
        rip_height=rip_height,
    )
    assert original_path.exists(), f"{original_path=} does not exist"


if __name__ == "__main__":
    test_get_flat_original_path()
    print("get_flat_original_path PASSED")