from blur_rgb_hwc_np_linear_f32 import (
     blur_rgb_hwc_np_linear_f32
)

import numpy as np


def add_shadows(rgb_hwc_np_linear_f32):
    """
    Add shadows to an image.

    Args:
        rgb_hwc_np_linear_f32: A numpy array of shape (H, W, 3) and dtype float32.

    Returns:
        A numpy array of shape (H, W, 3) and dtype float32.
    """
    x, y = np.meshgrid(
        np.linspace(0, 1, rgb_hwc_np_linear_f32.shape[0], dtype=np.float32),
        np.linspace(0, 2, rgb_hwc_np_linear_f32.shape[1], dtype=np.float32),
        indexing='ij'
    )
    assert x.shape == rgb_hwc_np_linear_f32.shape[:2]
    assert y.shape == rgb_hwc_np_linear_f32.shape[:2]
    dx = np.random.uniform(0, 1)
    dy = np.random.uniform(0, 1)
    theta = np.random.uniform(0, 2 * np.pi)

    temp = np.random.randint(0, 3)
    if temp == 0:
        ur = np.random.uniform(5, 17)
        vr = np.random.uniform(5, 17)
    elif temp == 1:
        ur = np.random.uniform(2, 5)
        vr = np.random.uniform(2, 5)
    else:
        ur = 1
        vr = 1

    
    ux = ur * np.cos(theta)
    uy = ur * np.sin(theta)

    
    vx = - vr * np.sin(theta)
    vy =   vr * np.cos(theta)

    avg_light = np.random.uniform(low=0.8, high=1.2)
    light_vary = np.random.uniform(low=0, high=0.6)
    min_light = avg_light - light_vary
    max_light = avg_light + light_vary
    r = (max_light - min_light) / 2
    c = (max_light + min_light) / 2
    # print(f"{avg_light=} {light_vary=} {min_light=} {max_light=} {r=} {c=}")
    light = c + r * np.sin(2 * np.pi * (ux * x + uy * y + dx)) * np.sin(2 * np.pi * (vx * x + vy * y + dy))

    light_and_shadows = rgb_hwc_np_linear_f32 * light[..., np.newaxis]

    if np.random.randint(0, 2) == 0:
        sigma_x = np.random.uniform(0.0, 10)
        sigma_y = np.random.uniform(0.0, 3)
    else:
        sigma_x = 0
        sigma_y = 0

    blurred = blur_rgb_hwc_np_linear_f32(
        rgb_hwc_np_linear_f32=light_and_shadows,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
    )

    cropped = blurred.clip(0, 1)
    
    return cropped


