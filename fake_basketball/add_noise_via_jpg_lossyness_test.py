from pathlib import (
     Path
)
from add_noise_via_jpg_lossyness import (
     add_noise_via_jpg_lossyness
)
from prii import (
     prii
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)


def test_add_noise_via_jpg_lossyness_1():
    for quality in [50, 30, 20, 10, 5]:
        image_path = Path(
            "~/r/nba_ads/NBA_ID.jpg"
        ).expanduser()
        image_path = Path(
            "~/r/bos-mia-2024-04-21-mxf_led/justan/bos-mia-2024-04-21-mxf_495000_original.jpg"
        ).expanduser()
        rgb_hwc_np_u8 = open_as_rgb_hwc_np_u8(
            image_path=image_path
        )
        
    

        rgb_hwc_noisy_np_u8 = add_noise_via_jpg_lossyness(
            rgb_hwc_np_u8=rgb_hwc_np_u8,
            quality=quality,
        )
        prii(rgb_hwc_noisy_np_u8, caption=f"with JPEG noise at quality level {quality}")
    
if __name__ == "__main__":
    test_add_noise_via_jpg_lossyness_1()