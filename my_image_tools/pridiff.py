from open_as_rgba_hwc_np_u8 import (
     open_as_rgba_hwc_np_u8
)
from colorama import Fore, Style
import PIL
import PIL.Image
import numpy as np
from print_image_in_iterm2 import print_image_in_iterm2
import matplotlib.pyplot as plt




def pridiff(path1, path2):
    """
    This powers the pridiff command.
    Compares the RGB data of two images in iterm2.
    Alpha channels are ignored.
    Sometimes you need to diff two jpegs or pngs.
    """
    # cannot do these checks as they are, because it does not think of /dev/fd/63 as a file:
     
    # assert path1.is_file(), f"No such file {path1}"
    # assert path2.is_file(), f"No such file {path2}"

    # if save_path.is_file():
    #     ans = input(
    #         f"{save_path} already exists, are you sure you want to overwrite it? "
    #     )
    #     if ans not in ["yes", "Yes", "y", "Y"]:
    #         print("Stopping")
    #         sys.exit(1)
    print(f"{Fore.YELLOW}I ran{Style.RESET_ALL}")
    rgb_np_1 = open_as_rgba_hwc_np_u8(path1)[:, :, :3]
    rgb_np_2 = open_as_rgba_hwc_np_u8(path2)[:, :, :3]
    assert rgb_np_1.shape == rgb_np_2.shape, "ERROR: Images are not the same size!"
    diff = rgb_np_1.astype(np.int32) - rgb_np_2.astype(np.int32)
    sqrdiff = diff**2
    l2 = np.sum(sqrdiff, axis=2).astype(np.float64) ** 0.5
    a = l2.ravel()
    
    plt.hist(a, bins=[x for x in range(40+1)])
    plt.savefig("out.png", dpi=200)

    difference = np.zeros(
        shape=(
            rgb_np_1.shape[0],
            rgb_np_1.shape[1],
            rgb_np_1.shape[2]
        ),
        dtype=np.float32
    )
    difference += rgb_np_1
    difference -= rgb_np_2

    if np.max(difference) == np.min(difference):
        normalized_into_0_1 = np.abs(difference) / 255.0
    else:
        normalized_into_0_1 = (
            (difference - np.min(difference))
            /
            (np.max(difference) - np.min(difference))
        )
    quantized = np.clip(255 * normalized_into_0_1, 0, 255).astype(np.uint8)
    final_pil = PIL.Image.fromarray(quantized)
    print_image_in_iterm2(image_pil=final_pil)


