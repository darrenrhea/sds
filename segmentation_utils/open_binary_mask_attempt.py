
import PIL
import PIL.Image
import numpy as np
from pathlib import Path

def open_binary_mask_attempt(attempt_path):
    attempt_pil = PIL.Image.open(str(attempt_path))
    attempt_np = np.array(attempt_pil)
    assert attempt_np.ndim == 2, "We ask you to give us attempts that are truly grayscale i.e one channel only, not RGB nor RGBA"
    print(attempt_np.shape)
    if not np.all(
        np.logical_or(
            attempt_np == 0,
            attempt_np == 255
        )
    ):
       print(f"The classification attempt image:\n\n{attempt_path}\n\nhas shades of gray in it.  Just Sayin...")
    attempt_binary = (attempt_np >= 128)
    return attempt_binary

