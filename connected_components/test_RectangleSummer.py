
import sys
import cv2
import numpy as np
from warp_image import warp_image
import PIL.Image
from pathlib import Path
from print_image_in_iterm2 import print_image_in_iterm2
import pprint as pp
import dataclasses



class RectangleSummer(object):
    def __init__(self, binary_mask_np_uint8):
        """
        Suppose you wanted to know the number of Trues in a rectangular region of a binary mask,
        for many different rectangular regions.  You need something like this.
        """
        height = binary_mask_np_uint8.shape[0]
        width = binary_mask_np_uint8.shape[1]
        self.y = np.zeros(shape=(height+1, width+1), dtype=np.int32)
        self.y[1:, 1:] = np.cumsum(np.cumsum(binary_mask_np_uint8, axis=0), axis=1)

    def sum(self, x_min, xMax, y_min, yMax):
        """
        returns the sum over x_min <= x < xMax and y_min <= y < yMax of the binary mask
        """
        assert x_min >= 0
        assert y_min >= 0
        assert xMax > x_min
        assert yMax > y_min
        assert xMax <= self.y.shape[1]
        assert yMax <= self.y.shape[0]
        return self.y[yMax, xMax] - self.y[yMax, x_min] - self.y[y_min, xMax] + self.y[y_min, x_min]

    def frac(self, x_min, xMax, y_min, yMax):
        return self.sum(x_min, xMax, y_min, yMax) / ((xMax - x_min) * (yMax - y_min))


# 0 1 1 1 1
# 0 0 1 1 1
# 0 0 0 1 1
# 1 1 0 0 1
# 1 1 1 0 0
def test_RectangleSummer():
    b = np.array(
        [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [1, 1, 0, 0, 1],
            [1, 1, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    summer = RectangleSummer(b)
    assert summer.sum(0, 2, 0, 2) == 2
    assert summer.sum(0, 2, 0, 5) == 6
    assert summer.sum(0, 5, 2, 4) == 5
    assert summer.sum(0, 5, 0, 5) == 16

test_RectangleSummer()
