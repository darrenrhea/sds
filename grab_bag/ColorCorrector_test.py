from prii import (
     prii
)
from ColorCorrector import (
     ColorCorrector
)
from pathlib import Path
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
import numpy as np


def test_ColorCorrector_1():

    good_image_path = Path(
        "/media/drhea/muchspace/clips/munich2024-01-09-1080i-yadif/frames/munich2024-01-09-1080i-yadif_010450_original.jpg"
    )

    ugly_image_path = Path(
        "/media/drhea/muchspace/clips/youtubeAmiCuoupzPQ/frames/youtubeAmiCuoupzPQ_001600_original.jpg"
    )
    
    ugly_rgb_hwc_np_u8 = open_as_rgb_hwc_np_u8(
        image_path=ugly_image_path
    )
    print(f"{ugly_image_path}:")
    prii(ugly_rgb_hwc_np_u8)

    good_rgb_hwc_np_u8 = open_as_rgb_hwc_np_u8(
        image_path=good_image_path
    )
    print(f"{good_image_path}:")
    prii(good_rgb_hwc_np_u8)

    for gamma in np.linspace(start=1.0, stop=2.4, num=20):
        
        color_corrector = ColorCorrector(gamma=gamma)
        corrected_rgb_hwc_np_u8 = color_corrector.inverse_map(ugly_rgb_hwc_np_u8)
        print(f"Corrected via {gamma=}")
        prii(corrected_rgb_hwc_np_u8)
        print("does it look like this:")
        prii(good_rgb_hwc_np_u8)



if __name__ == "__main__":
    test_ColorCorrector_1()