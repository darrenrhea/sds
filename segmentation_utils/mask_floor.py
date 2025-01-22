import subprocess
from pathlib import Path
import sys

width = 1920
height = 1080
x_min = -47
x_max = -28.17
#x_min = 28.17,
#x_max = 47,
y_min = -7.83
y_max = 7.83


from PIL import Image
import numpy as np
import os

if __name__ == "__main__":

    suffix = "_players.png"
    for filename in os.listdir(Path("~/r/gsw1/segmentation").expanduser()):
        if filename.endswith(suffix):
            name = os.path.basename(os.path.basename(filename)[:-len(suffix)])
            path_to_frame = Path(f"~/awecom/data/clips/gsw1/frames/{name}.jpg").expanduser()
            path_to_camera = Path(f"~/awecom/data/clips/gsw1/tracking_attempts/fourth/{name}_camera_parameters.json").expanduser()
            subprocess.call(
                [
                    Path("~/r/floor-ripper/mask_floor.exe").expanduser(),
                    str(path_to_camera),
                    str(width),
                    str(height),
                    str(x_min),
                    str(x_max),
                    str(y_min),
                    str(y_max),
                    "./tmp.png"
                ]
            )

            mask = 255*np.array(Image.open("./tmp.png"))[:,:,-1]
            frame = Image.open(path_to_frame)
            new_img = Image.fromarray(np.array(frame) * mask[:,:,None])
            dir_of_output = Path("~/r/gsw1/segmentation").expanduser()
            new_img.save( dir_of_output / f"{name}_lane.png" )