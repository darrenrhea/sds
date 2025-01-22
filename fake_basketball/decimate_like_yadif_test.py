from pathlib import Path

from decimate_like_yadif import (
    decimate_like_yadif
)

from open_as_hwc_rgb_np_uint8 import (
        open_as_hwc_rgb_np_uint8
)

from prii import prii

def decimate_like_yadif_test_0():
    image_path = Path(
        "~/r/munich1080i_led/sarah/munich2024-01-09-1080i-yadif_000500_nonfloor.png"
    ).expanduser()

    rgb_hwc_np_u8 = open_as_hwc_rgb_np_uint8(
        image_path=image_path
    )

    decimated = decimate_like_yadif(
        rgb_hwc_np_u8=rgb_hwc_np_u8,
        parity=0
    )
    
    prii(decimated)

    print("bye")

def decimate_like_yadif_test_1():
    image_path = Path(
        "~/r/munich1080i_led/sarah/munich2024-01-09-1080i-yadif_000500_nonfloor.png"
    ).expanduser()

    rgb_hwc_np_u8 = open_as_hwc_rgb_np_uint8(
        image_path=image_path
    )

    decimated = decimate_like_yadif(
        rgb_hwc_np_u8=rgb_hwc_np_u8,
        parity=1
    )
    
    prii(decimated)
    
    print("bye")

    
if __name__ == "__main__":
    decimate_like_yadif_test_0()
    decimate_like_yadif_test_1()


