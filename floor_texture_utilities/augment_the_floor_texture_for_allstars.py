import numpy as np


def augment_the_floor_texture_for_allstars(
    rgb_np_linear_f32: np.ndarray,
) -> np.array:
    """
    This function is used to augment the floor texture for allstars.
    """
    r = rgb_np_linear_f32[:, :, 0]
    g = rgb_np_linear_f32[:, :, 1]
    b = rgb_np_linear_f32[:, :, 2]
    if np.random.rand() < 0.5:
        print("Using indicators to determine the floor texture color")
        is_yellow = (r > 0.5) * (g > 0.5) * (b < 0.5)
        is_blue = (r < 0.3) * (g < 0.3)
        is_other = (1 - is_yellow) * (1 - is_blue)
        f = 0.45
        R = r + 0.001
        G = is_yellow * g * 1.2 + is_blue * f * g + is_other * g
        B = is_blue * (f * b) + is_yellow * 0 + is_other * b
    else:
        bright_factor = np.random.uniform(low=0.6, high=1.0)
        print(f"brightening the floor texture with {bright_factor=}")
        R = r * bright_factor
        G = g * bright_factor
        B = b * bright_factor
    
    rgb_np_linear_f32[:, :, 0] = R
    rgb_np_linear_f32[:, :, 1] = G
    rgb_np_linear_f32[:, :, 2] = B
    return rgb_np_linear_f32
