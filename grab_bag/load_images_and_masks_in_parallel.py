from get_frames_and_masks_of_dataset import get_frames_and_masks_of_dataset
import cv2
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

from joblib import Parallel, delayed
import multiprocessing
from typing import Callable


def load_images_and_masks_in_parallel(
    list_of_dataset_folders: List[Path],
    dataset_kind: str,
    preprocessor: Optional[Callable] = None,
    preprocessor_params: dict = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    See also load_datapoints_in_parallel.py
    to load all 3 at once.
    Given a list of dataset folders, load all the frames and masks in parallel.
    """
    if preprocessor is not None:
        assert isinstance(preprocessor, Callable)
        assert isinstance(preprocessor_params, dict)
    
    assert isinstance(list_of_dataset_folders, list)
    for d in list_of_dataset_folders:
        assert isinstance(d, Path)
        assert d.is_dir()

    input_frames_fn, target_masks_fn = get_frames_and_masks_of_dataset(
        list_of_dataset_folders=list_of_dataset_folders,
        dataset_kind=dataset_kind
    )

    # Because we are doing things in parallel, we define this function:
    def load_training_frame(i):
        try:
            raw_frame = cv2.imread(input_frames_fn[i])

            if raw_frame is None:
                raise Exception('error loading frame')
            
            assert raw_frame.dtype == np.uint8, "ERROR: frame must be uint8 for now"

            if raw_frame.shape[2] == 4:
                print(f'WARNING: removing alpha channel from {input_frames_fn[i]}')
                raw_frame = raw_frame[:, :, :3]

            raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)

         

            # some people want to use masks that have rgb info, others just the grayscale alpha channel
            mask_maybe_bgra = cv2.imread(target_masks_fn[i], -1)

            if mask_maybe_bgra is None:
                raise Exception('error loading mask')
            
            assert mask_maybe_bgra.dtype == np.uint8, "ERROR: mask files must be uint8 for now"

            if  mask_maybe_bgra.ndim == 3 and mask_maybe_bgra.shape[2] == 4:
                raw_mask = mask_maybe_bgra[:, :, 3].copy()
            elif mask_maybe_bgra.ndim == 2:
                raw_mask = mask_maybe_bgra
            else:
                raise Exception("The mask must be either grayscale or rgba (4 channels)")

            if preprocessor is None:
                frame, mask = raw_frame, raw_mask
            else:
                frame, mask = preprocessor(
                    raw_frame,
                    raw_mask,
                    params=preprocessor_params
                )

            thresh_mask = np.ones_like(mask).astype(float)

            return i, frame, mask, thresh_mask
        
        except Exception as e:
            print(f'ERROR processing {input_frames_fn[i]}:\n{e}')
            #return i, None, None, None
            raise e
 
    results = Parallel(
        n_jobs=min(multiprocessing.cpu_count() // 2, 32),
        backend = 'threading'
    )(delayed(load_training_frame)(i) for i in tqdm(range(len(input_frames_fn)), total = len(input_frames_fn)))
    
    results = sorted(results, key = lambda x: x[0])

    good_idx = [i for i in range(len(results)) if not results[i][1] is None]
    results = [results[i] for i in good_idx]
    input_frames_fn = [input_frames_fn[i] for i in good_idx]
    target_masks_fn = [target_masks_fn[i] for i in good_idx]

    input_frames = [res[1] for res in results]
    target_masks = [res[2] for res in results]

    return input_frames, target_masks
    
