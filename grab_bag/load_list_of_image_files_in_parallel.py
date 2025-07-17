"""
It may be better to have this work on just one GPU
and then have a separate script that
determines which GPUs aren't too busy then does parallelization
by calling this as a subprocess once per available GPU
with an assignment of work
appropriate to that GPU.
"""
import sys
import pprint as pp
import cv2
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
from typing import List, Optional




def load_list_of_image_files_in_parallel(
    list_of_image_file_paths: List[Path],
) -> List[Optional[np.ndarray]]:
    """
    Read in the specified image file paths into a list of Optional numpy frames.
    (Some frames may be None if there was an error reading them.)
    """

   
    def load_frame(list_of_image_file_paths, idx):
        try:
            fn = list_of_image_file_paths[idx]
            print(f'reading {fn}')
            frame = cv2.imread(str(fn))  # we think this removes any alpha channel
            if frame is None:
                raise Exception('error loading!')
            
            if frame.shape[2] == 4:
                raise Exception('alpha channel detected?! This should not happen without flag cv2.IMREAD_UNCHANGED')
            else:
                return idx, frame
        except Exception as e:
            print(f'ERROR processing {list_of_image_file_paths[idx]}:\n{e}')
            return idx, None

    results = Parallel(n_jobs=32)(delayed(load_frame)(list_of_image_file_paths, i)
                                    for i in tqdm(range(len(list_of_image_file_paths)), total = len(list_of_image_file_paths)))

    wondering_if_this_is_sorted = [x[0] for x in results]
    sorted_version = sorted(wondering_if_this_is_sorted)
    assert wondering_if_this_is_sorted == sorted_version, "ERROR: the results from the parallel loading of frames are not sorted by index"
    # If it is not inherently sorted, do so:
    # results = sorted(results, key = lambda x: x[0])

    frames = [res[1] for res in results]
    return frames



