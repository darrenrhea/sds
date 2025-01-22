import sys
import PIL
import PIL.Image
import numpy as np
from pathlib import Path



def transfer_flip_alpha_channel(alpha_path, rgb_path, save_path):
    """
    This is good for when the mask is correct in the alpha channel of one png, but
    """
    print(
        f"Taking the alpha channel from:\n"
        f"{alpha_path}\n"
        f"and the rgb info from:\n"
        f"{rgb_path}\n"
        f"and combining them to form {save_path}\n"
    )
    alpha_pil = PIL.Image.open(str(alpha_path))  # the source of the good alpha channel
    alpha_source_np = np.array(alpha_pil)
    assert alpha_source_np.dtype == np.uint8, f"Why is the datatype {alpha_source_np.dtype}"
    if alpha_source_np.ndim == 3:
        assert alpha_source_np.shape[2] == 4, f"{alpha_path} doesn't have an alpha channel!"
        alpha_np = alpha_source_np[:, :, 3]
    elif alpha_source_np.ndim == 2:  # if the source of alpha information is a grayscale image, it will be only 2 dim:
        alpha_np = alpha_source_np
    
    rgba_pil = PIL.Image.open(str(rgb_path)).convert(
        "RGBA"
    )  # the source of the rgb information. Even if it doesn't have an alpha channel, make one
    rgba_np = np.array(rgba_pil)
    rgba_np[:, :, 3] = 255 - alpha_np
    save_pil = PIL.Image.fromarray(rgba_np)
    save_pil.save(save_path)


def main():
    if len(sys.argv) < 4:
        print(
            """
Usage:
    transfer_flip_alpha <source_of_alpha.png> <source_of_rgb_info.(png|jpg|bmp)> <where_to_output_combined_rgba.png>
"""
        )
        sys.exit(1)

    alpha_path = Path(sys.argv[1])
    rgb_path = Path(sys.argv[2])
    save_path = Path(sys.argv[3])
    assert alpha_path.is_file(), f"No such file {alpha_path}"
    assert rgb_path.is_file(), f"No such file {rgb_path}"

    # if save_path.is_file():
    #     ans = input(
    #         f"{save_path} already exists, are you sure you want to overwrite it? "
    #     )
    #     if ans not in ["yes", "Yes", "y", "Y"]:
    #         print("Stopping")
    #         sys.exit(1)

    transfer_flip_alpha_channel(
        alpha_path=alpha_path, rgb_path=rgb_path, save_path=save_path
    )

if __name__ == "__main__":
    main()
