from draw_marker_on_np_u8 import (
     draw_marker_on_np_u8
)
from prii import (
     prii
)
from translated_feathered_paste import (
     translated_feathered_paste
)

import PIL.Image
import numpy as np


def test_translated_feathered_paste():
    cutout_pil = PIL.Image.open("cutout.png").convert("RGBA")
    resized_cutout_pil = cutout_pil.resize(
        size=(1550, 1500),
    )

    cutout_rgba_np_uint8 = np.array(resized_cutout_pil)
    print(f"{cutout_rgba_np_uint8.shape}")

    bottom_layer_color_np_uint8 = np.array(
        PIL.Image.open("ball_000113.jpg")
    )

   
    h = bottom_layer_color_np_uint8.shape[0]
    w = bottom_layer_color_np_uint8.shape[1]
    bottom_xy = (
        int(np.random.randint(0, w)),
        int(np.random.randint(0, h)),
    )

    temp = bottom_layer_color_np_uint8.copy()
    draw_marker_on_np_u8(xy=bottom_xy, victim=temp)
    prii(temp, caption="bottom_layer_color_np_uint8")



    top_xy=(465, 146)


    temp = cutout_rgba_np_uint8.copy()
    draw_marker_on_np_u8(xy=(465, 146), victim=temp)
    prii(temp, caption="cutout_rgba_np_uint8")


   

    result_np_int8 = translated_feathered_paste(
        bottom_layer_color_np_uint8=bottom_layer_color_np_uint8,
        top_layer_rgba_np_uint8=cutout_rgba_np_uint8,
        top_xy=top_xy,
        bottom_xy=bottom_xy
    )

    prii(result_np_int8)

    PIL.Image.fromarray(result_np_int8).save("result.png")

if __name__ == "__main__":
    test_translated_feathered_paste()