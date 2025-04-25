from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

from joblib import Parallel, delayed
import multiprocessing
from typing import Callable
from open_mask_image import open_mask_image
from open_just_the_rgb_part_of_image import open_just_the_rgb_part_of_image


def new_load_datapoints_in_parallel(
    datapoint_path_tuples: List[Tuple[Path, Path, Optional[Path]]],
    do_in_parallel: bool,
    preprocessor: Optional[Callable] = None,
    preprocessor_params: dict = None,
) -> List[np.ndarray]:
    """
    Given datapoint_path_tuples
    load all the frames and target_masks and weight_masks in parallel
    and return them as a Python list of DataPoint objects.
    TODO: generalize this to multiple target_masks, multiple weight_masks,
    no weight_mask, etc.
    """
    if preprocessor is not None:
        assert isinstance(preprocessor, Callable)
        assert isinstance(preprocessor_params, dict)
    

    for path_tuple in datapoint_path_tuples:
        assert len(path_tuple) == 3, f"ERROR: {path_tuple} does not have 3 elements"
        for path in path_tuple[:2]:
            assert isinstance(path, Path), f"ERROR: {path} is not a Path"
            assert path.is_file()
        weight_mask_path = path_tuple[2]
        assert (
            weight_mask_path is None
            or
            weight_mask_path.is_file()
        ), f"ERROR: {weight_mask_path} is not an extant file, nor is is None"



    # Because we are doing things in parallel, we define this function:
    def load_training_frame(i):
        try:
            original_frame_path, target_mask_path, weight_mask_path = datapoint_path_tuples[i]
            assert original_frame_path.is_file(), f"ERROR: {original_frame_path} does not exist"
            assert target_mask_path.is_file(), f"ERROR: {target_mask_path} does not exist"
            assert (
                weight_mask_path is None
                or
                weight_mask_path.is_file()
            ), f"ERROR: {weight_mask_path} does not exist nor is it None"

            raw_frame = open_just_the_rgb_part_of_image(image_path=original_frame_path)
            raw_target_mask = open_mask_image(target_mask_path)

            if weight_mask_path is not None:
                raw_weight_mask = open_mask_image(weight_mask_path)
            else:
                # if there is no weight mask, we just use a mask of all 255s,
                # i.e. we treat the whole image as equally relevant
                raw_weight_mask = 255 * np.ones_like(raw_target_mask)

            assert np.any(raw_weight_mask > 0), f"ERROR: raw_weight_mask is all zeros for {original_frame_path}"
            raw_stack_of_channels = np.concatenate(
                [
                    raw_frame,
                    raw_target_mask[:, :, None],
                    raw_weight_mask[:, :, None]
                ],
                axis=2
            )

            if preprocessor is None:
                channel_stack = raw_stack_of_channels
            else:
                channel_stack = preprocessor(
                    channel_stack=raw_stack_of_channels,
                    params=preprocessor_params
                )

            # the index i is to sort them later
            return i, channel_stack
        
        except Exception as e:
            print(f'YOYO ERROR processing {datapoint_path_tuples[i]}:\n{e}')
            raise e
 
    if do_in_parallel:
        results = Parallel(
            n_jobs=min(multiprocessing.cpu_count() // 2, 32),
            backend = 'threading'
        )(delayed(load_training_frame)(i) for i in tqdm(range(len(datapoint_path_tuples)), total = len(datapoint_path_tuples)))
        results = sorted(results, key = lambda x: x[0])
    else:
        results = []
        for i in range(len(datapoint_path_tuples)):
            result = load_training_frame(i)
            results.append(result)
        
   

    channel_stacks = [result[1] for result in results]

    for channel_stack in channel_stacks:
        assert isinstance(channel_stack, np.ndarray)
        assert channel_stack.shape[2] == 5, f"ERROR: {channel_stack.shape[2]=} is not 5"
        assert channel_stack.dtype == np.uint8
    
    return channel_stacks
    
