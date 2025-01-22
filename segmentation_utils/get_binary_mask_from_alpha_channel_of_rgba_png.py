
import PIL
import PIL.Image
import numpy as np
from pathlib import Path
import sys


def get_binary_mask_from_alpha_channel_of_rgba_png(image_path, save_path, threshold=127):
    assert isinstance(threshold, int)
    assert 0 <= threshold and threshold <= 254
    assert isinstance(image_path, Path)
    image_pil = PIL.Image.open(str(image_path))
    image_rgba_np = np.array(image_pil)
    assert image_rgba_np.shape[2] == 4, f"{image_path} must be an RGBA PNG file."
    alpha_np = image_rgba_np[:, :, 3]  # get just the alpha channel
    if not np.all(
        np.logical_or(
            alpha_np == 0,
            alpha_np == 255
        )
    ):
        print(f"The alpha channel of image:\n\n    {image_path}\n\nhas shades of gray in it.  Just Sayin...")
    binarized_alpha_np = (alpha_np > threshold)
    save_pil = PIL.Image.fromarray(binarized_alpha_np)
    save_pil.save(save_path)
    return binarized_alpha_np


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            """
            Usage:
            python get_binary_mask_from_alpha_channel_of_rgba_png <source_of_rgba_info.jpg_or_png> <where_to_black_and_white_mask.png>
        """
        )
        sys.exit(1)

    image_path = Path(sys.argv[1])
    save_path = Path(sys.argv[2])
    assert image_path.is_file(), f"No such file {image_path}"

    if save_path.is_file():
        ans = input(
            f"{save_path} already exists, are you sure you want to overwrite it? "
        )
        if ans not in ["yes", "Yes", "y", "Y"]:
            print("Stopping")
            sys.exit(1)

    get_binary_mask_from_alpha_channel_of_rgba_png(image_path, save_path, threshold=127)


