from alpha_composite_for_images_of_the_same_size import (
     alpha_composite_for_images_of_the_same_size
)
from write_rgb_and_alpha_to_png import (
     write_rgb_and_alpha_to_png
)
from pathlib import Path
from make_rgb_hwc_np_u8_from_rgba_hwc_np_u8 import make_rgb_hwc_np_u8_from_rgba_hwc_np_u8
from open_as_rgba_hwc_np_u8 import (
     open_as_rgba_hwc_np_u8
)
import cv2
from prii import prii
import numpy as np


import PIL

def downsample_rgba(rgba):
    height = rgba.shape[0]
    width = rgba.shape[1]
    downsample_width = width // 4
    downsample_height = height // 4

    img_pil = PIL.Image.fromarray(rgba)
    smaller_pil = img_pil.resize((downsample_width, downsample_height))
    smaller_rgba = np.array(smaller_pil)
    return smaller_rgba

def get_original_and_mask():

    k = 7
    where_name = [
        "shorts",
        "shirt",
        "shoes",
        "jersey",
    ][3]
    image_path = Path(
        # "~/r/allstarsrising2-2025-02-14-sdi_floor/katie/allstarsrising2-2025-02-14-sdi_050400_nonfloor.png"
        f"~/r/chatgpt_cutouts_approved/apl/{k}_original.png"
    ).expanduser()

    where_to_change_mask_path = Path(
        # "~/r/allstarsrising2-2025-02-14-sdi_floor/katie/allstarsrising2-2025-02-14-sdi_050400_nonfloor.png"
        f"~/r/chatgpt_cutouts_approved/apl/{k}_{where_name}.png"
    ).expanduser()

    rgba = open_as_rgba_hwc_np_u8(
        image_path=image_path
    )
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    prii(rgba, caption="rgba")
    prii(alpha, caption="alpha")

    where_to_change_mask = (
        open_as_rgba_hwc_np_u8(where_to_change_mask_path)[:, :, 3]
    )
    
    # where_to_change_mask *= (where_to_change_mask > 16)
    
    prii(where_to_change_mask, caption="where_to_change_mask")
    return rgb, alpha, where_to_change_mask


def interactive(
    rgb: np.ndarray,
    alpha: np.ndarray,
    where_to_change_mask: np.ndarray,
):
    window_name = "change hue on where_to_change_mask; press any key to exit"
    cv2.namedWindow(window_name)

    def on_change(value):
        # print(f"Slider value: {value}")
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = value
        modified_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        # merged_image = np.where(
        #     where_to_change_mask[:, :, np.newaxis],
        #     modified_rgb,
        #     rgb,
        # )
        merged_image = alpha_composite_for_images_of_the_same_size(
            bottom_layer_color_np_uint8=rgb,
            top_layer_color_np_uint8=modified_rgb,
            top_opacity_np_uint8=where_to_change_mask,
        )
        rgba = np.dstack([merged_image, alpha])
        downsampled = downsample_rgba(rgba)

        collapsed = make_rgb_hwc_np_u8_from_rgba_hwc_np_u8(
            rgba_hwc_np_u8=downsampled
        )
        merged_image_bgr = cv2.cvtColor(collapsed, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, merged_image_bgr)

    cv2.createTrackbar('hue', window_name, 0, 255, on_change)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    rgb, alpha, where_to_change_mask = get_original_and_mask()

    interactive(
        rgb=rgb,
        alpha=alpha,
        where_to_change_mask=where_to_change_mask
    )