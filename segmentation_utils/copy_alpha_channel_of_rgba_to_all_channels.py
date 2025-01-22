import sys
import PIL
import PIL.Image
import numpy as np
from pathlib import Path
from numpy.core.fromnumeric import searchsorted
from numpy.lib.npyio import save


def quadruplicate_alpha_channel(alpha_path, save_path):
    """
    This is good for when the mask is correct in the alpha channel of one png, but
    """
    print(
        f"Taking the alpha channel from:\n"
        f"{alpha_path}\n"
        f"and quadruplicating to form {save_path}\n"
    )
    alpha_pil = PIL.Image.open(str(alpha_path))  # the source of the good alpha channel
    alpha_source_np = np.array(alpha_pil)
    assert alpha_source_np.dtype == np.uint8, f"Why is the datatype {alpha_source_np.dtype}"
    alpha_np = np.zeros((alpha_source_np.shape[0], alpha_source_np.shape[1], alpha_source_np.shape[2]), dtype=np.uint8)
    alpha_np[:, :, 0] = alpha_source_np[:, :, 3]
    alpha_np[:, :, 1] = alpha_source_np[:, :, 3]
    alpha_np[:, :, 2] = alpha_source_np[:, :, 3]
    alpha_np[:, :, 3] = alpha_source_np[:, :, 3]
    save_pil = PIL.Image.fromarray(alpha_np)
    save_pil.save(save_path)


def main():
    if len(sys.argv) < 3:
        print(
            """
Usage:
    quadruplicate_alpha_channel <source_of_alpha.png>  <where_to_output_combined_rgba.png>
"""
        )
        sys.exit(1)

    alpha_path = Path(sys.argv[1])
    save_path = Path(sys.argv[2])
    assert alpha_path.is_file(), f"No such file {alpha_path}"

    if save_path.is_file():
        ans = input(
            f"{save_path} already exists, are you sure you want to overwrite it? "
        )
        if ans not in ["yes", "Yes", "y", "Y"]:
            print("Stopping")
            sys.exit(1)

    quadruplicate_alpha_channel(
        alpha_path=alpha_path, save_path=save_path
    )

if __name__ == "__main__":
    main()
