from blackpad_preprocessor import blackpad_preprocessor
from get_frames_and_masks_of_dataset import get_frames_and_masks_of_dataset
import cv2
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

from joblib import Parallel, delayed
import multiprocessing
from typing import Callable
from print_image_in_iterm2 import print_image_in_iterm2


from load_images_and_masks_in_parallel import load_images_and_masks_in_parallel



if __name__ == "__main__":

    list_of_dataset_folders = [
        Path("~/alpha_mattes_temp").expanduser().resolve()
    ]

    images, masks = load_images_and_masks_in_parallel(
        list_of_dataset_folders=list_of_dataset_folders,
        dataset_kind = "nonfloor",
        preprocessor=blackpad_preprocessor,
        preprocessor_params=dict(
            desired_height=1088,
            desired_width=1920
        )
    )

    for image, mask in zip(images, masks):
        assert isinstance(image, np.ndarray)
        assert isinstance(mask, np.ndarray)
        assert image.shape[0] == mask.shape[0]
        assert image.shape[1] == mask.shape[1]
        assert image.shape[2] == 3
        assert mask.ndim == 2
        print(f"{image.shape=}, {mask.shape=}")
        print_image_in_iterm2(rgb_np_uint8=image)
        print_image_in_iterm2(grayscale_np_uint8=mask)
        




