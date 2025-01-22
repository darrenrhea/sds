"""
We intend to import all of this in a jupyter notebook
by doing

::

  from all_imports_for_image_segmentation import *

"""

import imageio
from IPython.core.display import display
import numpy as np
import os
from pathlib import Path
import PIL
import pprint as pp
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms
import my_pytorch_utils
from image_loaders_and_savers import (
    get_hwc_uint8,
    get_hwc_np_uint8_from_image_path,
    get_hwc_np_uint8_from_image_path_bw,
    get_resized_hwc_uint8_from_image_path,
    save_hwc_np_uint8_to_image_path
)
from image_displayers_for_jupyter import (
    display_annotations_from_image_and_json_paths,
    display_numpy_hwc_rgb_image,
    display_numpy_chw_rgb_image,
    display_numpy_hw_grayscale_image,
    numpy_chw_rgb_to_uint8_np,
)
import matplotlib.pyplot as plt
from annotated_data import get_list_of_annotated_images
from get_numpy_arrays_of_croppings_and_their_masks import get_numpy_arrays_of_croppings_and_their_masks, get_numpy_arrays_of_croppings_and_their_masks_bw
from cut_this_many_interesting_subrectangles_from_annotated_image import cut_this_many_interesting_subrectangles_from_annotated_image, cut_this_many_interesting_subrectangles_from_annotated_image_bw
from pred_for_chunk import pred_for_chunk