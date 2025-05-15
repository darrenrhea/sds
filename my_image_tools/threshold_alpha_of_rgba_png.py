usage_message = """
Usage:

    threshold_alpha <mask.png> <threshold> [where_to_write_result.png]
 
This thresholds the alpha channel of the RGBA image you gave
to be strictly one of the two extreme values 0 or 255, depending
on whether alpha < threshold or alpha >= threshold.
The threshold you specify must be between 1 and 255 inclusive.

The resulting RGBA will be shown to you given that the terminal client is iterm2.

Optionally, once you have tried various thresholds, you can write out the result
by specifying an output file.

Examples:

    # anything below 255 will be made fully transparent, 255 stays fully opaque:
    $ threshold_alpha gsw1_150331_full.png 255

    # anything 1 or above will be made fully opaque, alpha=0 pixels stay fully transparent: 
    $ threshold_alpha gsw1_150331_full.png 1

"""

from print_image_in_iterm2 import print_image_in_iterm2
import sys
import PIL
import PIL.Image
import numpy as np
from pathlib import Path
from colorama import Fore, Style

def main():
    if len(sys.argv) < 3:
        print(usage_message)
        sys.exit(1)

    image_path = Path(sys.argv[1])
    threshold = int(sys.argv[2])
    if len(sys.argv) >= 4:
        out_path = Path(sys.argv[3])
    else:
        out_path = None
    
    assert 1 <= threshold and threshold <= 255, "The threshold must be between 1 and 255 inclusive."

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
        print(f"The alpha channel of the original image:\n\n    {image_path}\n\nhas fractional transparency in it.")
    else:
        print(f"The alpha channel of the original image:\n\n    {image_path}\n\nis already all 0 or 255.")


    
    thresholded_np = 255 * (
        alpha_np >= threshold
    ).astype(np.uint8)
    image_rgba_np[:,:,3] = thresholded_np
    new_image_pil = PIL.Image.fromarray(image_rgba_np)
    print_image_in_iterm2(new_image_pil)

    if out_path is not None:
        if out_path.is_file():
            ans = input(f"{out_path} already exists?  Destroy it by overwriting it with result?")
            if ans not in ["y", "Y", "yes"]:
                sys.exit(1)
        new_image_pil.save(out_path)
        print(f"The result of thresholding has been saved to:\n    {out_path}")


if __name__ == "__main__":
    main()
