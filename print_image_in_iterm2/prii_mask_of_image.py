import numpy as np
from prii import (
     prii
)

def prii_mask_of_image(
    rgb,
    mask
):
    rgba_np = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    rgba_np[:, :, :3] = rgb
    rgba_np[:, :, 3] = mask
    prii(rgba_np)