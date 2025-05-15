from pridiff import pridiff
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
)


def pridiff_cli_tool():
    """
    This is the "main" or entry-point of the command line utility pridiff.
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