import numpy as np


def augment_the_floor_texture_for_bal(
    rgb_np_linear_f32: np.ndarray,
) -> np.array:
    """
    This function is used to augment the floor texture for
    BAL / Basketball Africa League.
    """
    r = rgb_np_linear_f32[:, :, 0]
    g = rgb_np_linear_f32[:, :, 1]
    b = rgb_np_linear_f32[:, :, 2]
   
    bright_factor = np.random.uniform(low=0.6, high=1.0)
    print(f"brightening the floor texture with {bright_factor=}")
    R = r * bright_factor
    G = g * bright_factor
    B = b * bright_factor
    
    rgb_np_linear_f32[:, :, 0] = R
    rgb_np_linear_f32[:, :, 1] = G
    rgb_np_linear_f32[:, :, 2] = B
    return rgb_np_linear_f32
