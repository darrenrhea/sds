import sys
import PIL
import PIL.Image
import numpy as np
from pathlib import Path



def add_alphas(alpha1_path, alpha2_path, rgb_path, save_path):
    """
    adding or unioning together two masks is often useful.

    Use case one:

    The nonfloor-style-mask is appoximately equal to 
    the important-people-style-mask
    unioned
    with the gross-off-the-court-style-mask

    """
    print(
        f"Taking the alpha channela from {alpha1_path} and from {alpha2_path} and the rgb info from {rgb_path} and combining them to form {save_path}"
    )
    alpha1_pil = PIL.Image.open(str(alpha1_path))  # the source of the good alpha channel
    alpha1_np = np.array(alpha1_pil)
    alpha2_pil = PIL.Image.open(str(alpha2_path))  # the source of the good alpha channel
    alpha2_np = np.array(alpha2_pil)
    

    rgb_pil = PIL.Image.open(str(rgb_path)).convert(
        "RGBA"
    )  # the source of the rgb information. Even if it doesn't have an alpha channel, make one
    rgba_np = np.array(rgb_pil)
    if alpha1_np.dtype == np.bool:
        alpha1_np = alpha1_np.astype("uint8") * 255
    if alpha1_np.ndim == 3:
        assert alpha1_np.shape[2] == 4, f"{alpha1_path} doesn't have an alpha channel!"
        alpha1_channel = alpha1_np[:, :, 3]
    elif alpha1_np.ndim == 2:
        alpha1_channel = alpha1_np[:, :]
    else:
        raise Exception("No idea how we got here")

    if alpha2_np.dtype == np.bool:
        alpha2_np = alpha2_np.astype("uint8") * 255
    if alpha2_np.ndim == 3:
        assert alpha2_np.shape[2] == 4, f"{alpha2_path} doesn't have an alpha channel!"
        alpha2_channel = alpha2_np[:, :, 3]
    elif alpha2_np.ndim == 2:
        alpha2_channel = alpha2_np[:, :]
    else:
        raise Exception("No idea how we got here.")
    
    alpha_channel = np.fmax(alpha1_channel, alpha2_channel)

    rgba_np[:,:,3] =alpha_channel
    save_pil = PIL.Image.fromarray(rgba_np)
    save_pil.save(save_path)


def main():
    if len(sys.argv) < 5:
        print(
            """
Usage:
    add_alphas <source_of_alpha1.png> <source_of_alpha2.png> <source_of_rgb_info.(png|jpg|bmp)> <where_to_output_combined_rgba.png>
"""
        )
        sys.exit(1)

    alpha1_path = Path(sys.argv[1])
    alpha2_path = Path(sys.argv[2])
    rgb_path = Path(sys.argv[3])
    save_path = Path(sys.argv[4])
    assert alpha1_path.is_file(), f"No such file {alpha1_path}"
    assert alpha2_path.is_file(), f"No such file {alpha2_path}"
    assert rgb_path.is_file(), f"No such file {rgb_path}"

    if save_path.is_file():
        ans = input(
            f"{save_path} already exists, are you sure you want to overwrite it? "
        )
        if ans not in ["yes", "Yes", "y", "Y"]:
            print("Stopping")
            sys.exit(1)

    add_alphas(
        alpha1_path=alpha1_path,
        alpha2_path=alpha2_path,
        rgb_path=rgb_path,
        save_path=save_path
    )

if __name__ == "__main__":
    main()
