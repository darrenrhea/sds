"""
We used to do gsw1, which was pure camera 1.
Now we are doing gsw1_multi, which is a program feed,
and thus has these "score bugs" overlayed.
We need to make masks that respect that the scorebug is foreground.
Also, for reasons, gsw1 and gsw1_multi have a nasty little 4 frame index translation.
We will talk in the original index convention of gsw1 since the hand annotations are
saved relative to that.
"""
import sys
import PIL
import PIL.Image
import numpy as np
from pathlib import Path
from add_alphas import add_alphas


# def add_alphas(alpha1_path, alpha2_path, rgb_path, save_path)

hand_annotations_dir = Path("~/r/final_gsw1/final_segmentation").expanduser()
for x in hand_annotations_dir.listdir():
    print(x)
