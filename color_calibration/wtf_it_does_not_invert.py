import numpy as np

from skimage.color import rgb2lab, lab2rgb

r = 241.0
g = 125.0
b = 37.0

print(f"{r=}, {g=}, {b=}")
lstar, astar, bstar = rgb2lab(
    np.array(
        [r, g, b]
    ).reshape(1, 1, 3) / 255.0
)[0, 0]

print(f"{lstar=}, {astar=}, {bstar=}")




recovered_r, recovered_g, recovered_b = lab2rgb(
    np.array(
        [lstar, astar, bstar]
    ).reshape(1, 1, 3)
)[0, 0] * 255.0

print(f"{recovered_r=}, {recovered_g=}, {recovered_b=}")
