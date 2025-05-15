usage_message = """
Usage:

    confirm_binary <mask.png>
 
If someone makes an attempt to foreground-mask an image,
currently we want it to be binarized, i.e. fully opaque or fully transparent.

"""

from print_image_in_iterm2 import print_image_in_iterm2
import sys
import PIL
import PIL.Image
import numpy as np
from pathlib import Path
from colorama import Fore, Style

def main():
    if len(sys.argv) < 2:
        print(usage_message)
        sys.exit(1)

    image_path = Path(sys.argv[1])

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
        print(f"{Fore.RED}BINARYNESS FAILED: The alpha channel of image:\n\n    {image_path}\n\nhas fractional transparency in it:\n\n{Style.RESET_ALL}")

        total = np.sum(
            np.logical_and(
                alpha_np > 0,
                alpha_np < 255
            )
        )
        
        image_np = 255 * np.logical_and(
            alpha_np > 0,
            alpha_np < 255
        ).astype(np.uint8)
        image_pil = PIL.Image.fromarray(image_np)
        print(f"here are the {total} pixels whose alpha value is strictly between 0 and 255:")
        print_image_in_iterm2(image_pil)

        exit(1)  # exit code 1 means failure
    else:
        print(f"{Fore.GREEN}BINARYNESS PASSED: The alpha channel of image:\n\n    {image_path}\n\nhas only values 0 and 255.{Style.RESET_ALL}")
        print(f"\n\n{Fore.YELLOW}COLOR INFO INCLUDED??? If this does not show the full original image as it was prior to segmenting, the color info is not in there:{Style.RESET_ALL}")
        rgb_image_pil = PIL.Image.fromarray(image_rgba_np[:, :, :3])
        print_image_in_iterm2(rgb_image_pil)
        exit(0)  #  exit code 0 means success


if __name__ == "__main__":
    main()
