from get_a_temp_file_path import (
     get_a_temp_file_path
)
import numpy as np
from prii import (
     prii
)
from pathlib import Path
import matplotlib.pyplot as plt

txt_file_path = Path(
    "~/r/train_and_infer/progress2.txt"
).expanduser()


lines = txt_file_path.read_text().split("\n")

train_lines = [line for line in lines if line.startswith("train:")]

i = []
y0 = []
y1 = []
for index, line in enumerate(train_lines):
    # print(f"{line=}")
    a = len("train: l1_loss_for_target0: ")
    b = len("train: l1_loss_for_target0: 0.001405")
    c = len("train: l1_loss_for_target0: 0.001405, l1_loss_for_target1: ")
    d = len("train: l1_loss_for_target0: 0.001405, l1_loss_for_target1: 0.001405")
    n1 = line[a:b]
    n2 = line[c:d]
    floor_not_floor = float(n1)
    depth_map = float(n2)
    # print(floor_not_floor, depth_map)
    i.append(index)
    y0.append(floor_not_floor)
    y1.append(depth_map)

t = np.array(i) * 1.5 / 60
# save the plot to a PNG:
plt.plot(t, np.log10(y0), label="floor_not_floor")
plt.plot(t, np.log10(y1), label="depth_map")
plt.xlabel("hours training")
plt.ylabel("log_10(loss)")
plt.legend()
temp_path = get_a_temp_file_path(suffix=".png")
plt.savefig(temp_path, dpi=300)
prii(temp_path)

