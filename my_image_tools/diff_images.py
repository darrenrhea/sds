"""
Sometimes you need to diff two jpegs or pngs.
"""
import sys
import PIL
import PIL.Image
import numpy as np
from pathlib import Path
from print_image_in_iterm2 import print_image_in_iterm2
import matplotlib.pyplot as plt
from image_openers import (
    open_as_hwc_rgb_np_uint8,
    open_as_hwc_rgba_np_uint8,
    open_alpha_channel_image_as_a_single_channel_grayscale_image
)



def open_as_grayscale_and_yet_still_hwc_rgb_np_uint8(image_path):
    """
    Opens an image file path to be grayscale and yet still 3 channeled RGB np.uint8 H x W x C

    We will be starting off with a grayscale variant of the image, but
    then we plan on mutating it to have colorful parts, so it needs to have
    all three channels R G and B, despite that those
    3 channels contain identical data listed three times over.
    """
    pil = PIL.Image.open(str(image_path))
    pil = pil.convert("Y").convert("RGB")
    image_np_uint8 = np.array(pil)
    assert image_np_uint8.ndim == 3
    assert image_np_uint8.shape[2] == 3
    return image_np_uint8


def pridiff(path1, path2):
    """
    This powers the pridiff command.
    Compares the RGB data of two images in iterm2.
    Alpha channels are ignored.
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
    rgb_np_1 = open_as_hwc_rgba_np_uint8(path1)[:, :, :3]
    rgb_np_2 = open_as_hwc_rgba_np_uint8(path2)[:, :, :3]
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



def pridiffexact(path1, path2):
    """
    This powers the pridiffexact command.
    Compares the RGB data of two images in iterm2.
    Alpha channels are ignored.
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
    rgb_np_1 = open_as_hwc_rgb_np_uint8(path1)[:, :, :3]
    rgb_np_2 = open_as_hwc_rgb_np_uint8(path2)[:, :, :3]
    assert rgb_np_1.shape == rgb_np_2.shape, "ERROR: Images are not the same size!"
    diff = np.any(
        np.abs(rgb_np_1.astype(np.int32) - rgb_np_2.astype(np.int32)) >= 1, axis=2)
    visualization = rgb_np_1.copy()
    visualization[diff, :] = [255, 0, 0]
    final_pil = PIL.Image.fromarray(visualization)
    print_image_in_iterm2(image_pil=final_pil)






def main():
    """
    This is the "main" of the command line utility pridiff.
    """
    if len(sys.argv) < 3:
        print(
            """
Prints the diff between the rgb parts of two images in iterm2.

Usage:
   pridiff image1.(jpg|png) image2.(jpg|png) 
"""
        )
        sys.exit(1)

    path1 = Path(sys.argv[1])
    path2 = Path(sys.argv[2])
    pridiff(
        path1=path1,
        path2=path2
    )

def mainalpha():
    """
    This is the "main" of the command line utility pridiffalpha.
    """
    if len(sys.argv) < 3:
        print(
            """
Prints the diff between the alpha parts of two images in iterm2.


Usage:
   pridiffalpha old_version_of_image.(jpg|png) new_version_of_image.(jpg|png)

Pieces that got added as we changed from old to new shown in green.
Pieces that got removed shown in red.
"""
        )
        sys.exit(1)

    path1 = Path(sys.argv[1])
    path2 = Path(sys.argv[2])
    pridiffalpha(
        path1=path1,
        path2=path2
    )


def exact_main():
    """
    This is the "main" of the command line utility pridiffexact.
    """
    if len(sys.argv) < 3:
        print(
            """
Prints the exact diff between the rgb parts of two images in iterm2

Usage:
   pridiffexact image1.(jpg|png) image2.(jpg|png) 
"""
        )
        sys.exit(1)

    path1 = Path(sys.argv[1])
    path2 = Path(sys.argv[2])
    pridiffexact(
        path1=path1,
        path2=path2
    )

if __name__ == "__main__":
 
  path1 = Path(
    "fixtures/BOSvMIA_2022-05-23_PGM_ESP_415500.jpg"
  )
  path2 = Path(
    "fixtures/BOSvMIA_2022-05-23_PGM_ESP_415500_ad_insertion.png"
  )

  pridiffexact(path1, path2)