"""
Loop through a repo of masks and blur them (for better drawing)

"""

from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
import sys

FILTER_RADIUS = 3

if __name__ == "__main__":

    mask_dir = Path(sys.argv[1])
    new_mask_dir = Path(f"{str(mask_dir)}_blurred_radius_{FILTER_RADIUS}")
    new_mask_dir.mkdir(exist_ok=True)
    for filename in mask_dir.iterdir():
        im = Image.open(filename)
        im = im.filter(ImageFilter.GaussianBlur(radius=FILTER_RADIUS))
        im.save(new_mask_dir / filename.name)