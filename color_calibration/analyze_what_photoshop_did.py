# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
import sys
import numpy as np
from scipy.optimize import least_squares
import better_json as bj
import PIL.Image
from pathlib import Path
import matplotlib.pyplot as plt
from numpy import log

before_pil = PIL.Image.open("photoshop_input.png")
before_np = np.array(before_pil)
print(before_np.shape)
assert before_np.shape[2] == 3

after_pil = PIL.Image.open("photoshop_output.png")
after_np = np.array(after_pil)

assert after_np.shape == before_np.shape

xs = []
ys = []
ps = []

for i in range(0, before_np.shape[0]):
    for j in range(0, before_np.shape[0]):
        x = before_np[i,j,:]
        y = after_np[i,j,:]
        y_len = np.sqrt(np.sum(y**2))
        x_len = np.sqrt(np.sum(x**2))
        s = y[0] - x[0]
        print(f"s={s}")

        print(f"{x} --> {y}")
        print(x * s)
        xs.append(x[0])
        ys.append(y[0])
        ps.append(x[0] * 128/50)

plt.plot(xs, ys, '.')
plt.plot(xs, ps, 'r.')
plt.show()#block=False)
# plt.pause(interval=4)

