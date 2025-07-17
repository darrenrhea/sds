from prii_rgb_and_alpha import (
     prii_rgb_and_alpha
)
from prii import (
     prii
)
import numpy as np
import pprint as pp
from Segmenter import Segmenter


def infer_time_augmentation(
    rgb_hwc_np_u8: np.array,
    segmenter: Segmenter,
    augmentation,
    verbose: bool = False
) -> np.array:

    N = 3
    total_f32 = np.zeros(
        shape=(
            rgb_hwc_np_u8.shape[0],
            rgb_hwc_np_u8.shape[1]
        ),
        dtype=np.float32
    )

    for k in range(N):
        temp = augmentation(image=rgb_hwc_np_u8)
        aug_rgb_hwc_np_u8 = temp["image"]
        mask_hw_np_f32 = segmenter.infer_rgb_hwc_np_u8_to_hw_np_f32(
            aug_rgb_hwc_np_u8
        )
        mask_hw_np_u8 = np.round(mask_hw_np_f32 * 255.0).clip(0, 255).astype(np.uint8)
        
        if verbose:
            prii(aug_rgb_hwc_np_u8, caption="Infer time augmented:")
            print("resulting mask:")
            prii_rgb_and_alpha(
                rgb_hwc_np_u8=rgb_hwc_np_u8,
                alpha_hw_np_u8=mask_hw_np_u8
            )

        total_f32 += mask_hw_np_f32

    total_f32 /= N
    avg_mask_hw_np_u8 = np.round(total_f32 * 255.0).clip(0, 255).astype(np.uint8)


    return avg_mask_hw_np_u8


    


if __name__ == "__main__":
    infer_time_augmentation()