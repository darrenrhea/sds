from pathlib import Path
from prii import (
     prii
)
import numpy as np
from typing import List, Tuple

def blur_both_original_and_mask_u8(
    rgba_np_u8: np.ndarray,
    ij_displacement_and_weight_pairs: List[Tuple[Tuple[int, int], float]]
):
    """
    Blur both the original image and the mask image.

    Args:
    - rgba_np_u8: np.ndarray, shape (H, W, 4), dtype np.uint8, the original image stacked with the mask
    - ij_displacement_and_weight_pairs: List[Tuple[Tuple[int, int], float]], the   displacement and weight pairs.
    Suggest you use create_ij_displacement_and_weight_pairs to create this.

    Returns:
    - blurred_rgba_np_u8: np.ndarray, shape (H, W, 4), dtype np.uint8, the blurred image.
    """
    # Convert the image to float64:
    height, width, num_channels = rgba_np_u8.shape
    assert num_channels == 4


    max_i_displacement = max(
        abs(ij_displacement[0])
        for ij_displacement, _ in ij_displacement_and_weight_pairs
    )
    max_j_displacement = max(
        abs(ij_displacement[1])
        for ij_displacement, _ in ij_displacement_and_weight_pairs
    )
    min_i_displacement = min(
        ij_displacement[0]
        for ij_displacement, _ in ij_displacement_and_weight_pairs
    )
    min_j_displacement = min(
        ij_displacement[1]
        for ij_displacement, _ in ij_displacement_and_weight_pairs
    )
    padded = np.zeros(
        shape=(
            height + max_i_displacement - min_i_displacement,
            width + max_j_displacement - min_j_displacement,
            num_channels
        ),
        dtype=np.uint8
    )
    # print(f"{max_j_displacement=}")
    # print(f"{min_j_displacement=}")
    # print(f"{max_i_displacement=}")
    # print(f"{min_i_displacement=}")
    for i in range(max_i_displacement):
        padded[i, :, :] = rgba_np_u8[0, :, :]

    for j in range(max_j_displacement):
        padded[:, j, :] = rgba_np_u8[:, 0, :]
    
    padded[
        max_i_displacement:max_i_displacement+height,
        max_j_displacement:max_j_displacement+width,
        :
    ] = rgba_np_u8

    for i in range(height + max_i_displacement, height + max_i_displacement - min_i_displacement):
        padded[
            i,
            max_j_displacement:max_j_displacement+width,
            :
        ] = rgba_np_u8[-1, :, :]

    for j in range(width + max_j_displacement, width + max_j_displacement - min_j_displacement):
        padded[
            max_i_displacement:max_i_displacement+height,
            j,
            :
        ] = rgba_np_u8[:, -1, :]

    # prii(
    #     padded[:,:,:3],
    #     caption="padded:",
    #     out=Path("padded.png").resolve()
    # )

    rgba_np_f64 = padded.astype(np.float64)

    total = np.zeros(
        shape=padded.shape,
        dtype=np.float64
    )

    total_of_weights = 0.0
    for ij_displacement, weight in ij_displacement_and_weight_pairs:
        total_of_weights += weight
        total += weight * np.roll(
            a=rgba_np_f64,
            shift=ij_displacement,
            axis=(0, 1)
        ) 
    total /= total_of_weights
    ans = np.round(
        total[
            max_i_displacement:max_i_displacement+height,
            max_j_displacement:max_j_displacement+width,
        ]
    ).clip(0, 255).astype(np.uint8)
    # prii(ans[:,:,:3])
    # prii(ans[:,:,3])

    assert ans.shape == (height, width, num_channels)
    return ans