# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
import sys
import numpy as np
from scipy.optimize import least_squares
import better_json as bj
import PIL.Image
from pathlib import Path

r = (np.round(np.random.rand(100, 100, 3) * 255)).astype(np.uint8)


# larger = np.kron(
#     r,
#     np.ones(
#         shape=(15, 15, 1),
#         dtype=np.uint8
#     )
# )

image_pil = PIL.Image.fromarray(r)

image_pil.save("photoshop_input.png")
