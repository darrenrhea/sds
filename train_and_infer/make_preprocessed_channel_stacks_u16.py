from make_a_channel_stack_u16_from_local_file_pathed_annotation import (
     make_a_channel_stack_u16_from_local_file_pathed_annotation
)
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Optional, List

from joblib import Parallel, delayed
import multiprocessing
from typing import Callable


def make_preprocessed_channel_stacks_u16(
    local_file_pathed_annotations: List[dict],
    channel_names: List[str],
    preprocessor: Optional[Callable] = None,
    preprocessor_params: dict = None,
) -> List[np.ndarray]:
    """
    Define a "channel_stack" as a numpy array of shape (frame_height, frame_width, how_many_channels).
    It is a stack of channels, some of which are:
    * inputs to the neural network
    * labels/targets, i.e. the desired outputs of the neural network
    * Other things that affect the loss function, like relevance weights, i.e. how much we care about each pixel
    * similar things, like visibility_masks that are kind-of an input
      because we could know them ahead-of-time, but also kind-of something that affects the loss function.

    Given local_file_pathed_annotations, which is a list of dictionaries,
    load all the frames and target_masks and weight_masks in parallel
    and return them as 3 Python lists of numpy arrays.
    TODO: generalize this to multiple target_masks, multiple weight_masks,
    no weight_mask, etc.
    """
    assert isinstance(local_file_pathed_annotations, list)
    for dct in local_file_pathed_annotations:
        assert isinstance(dct, dict)
    
    if preprocessor is not None:
        assert isinstance(preprocessor, Callable)
        assert isinstance(preprocessor_params, dict)
    

    for dct in local_file_pathed_annotations:
        assert "original" in dct["local_file_paths"]
        assert "floor_not_floor" in dct["local_file_paths"]
        assert "depth_map" in dct["local_file_paths"]

        for label_name in ["original", "floor_not_floor", "depth_map"]:
            path = dct["local_file_paths"][label_name]
            assert isinstance(path, Path), f"ERROR: {path} is not a Path"
            assert path.is_file()
        

    # Because we are doing things in parallel, we define this function:
    def load_training_frame(i):
        try:
            local_file_pathed_annotation = local_file_pathed_annotations[i]
            unpreprocessed_channel_stack = make_a_channel_stack_u16_from_local_file_pathed_annotation(
                local_file_pathed_annotation=local_file_pathed_annotation,
                channel_names=channel_names,
            )
            assert isinstance(unpreprocessed_channel_stack, np.ndarray)
            assert unpreprocessed_channel_stack.dtype == np.uint16
            assert (
                unpreprocessed_channel_stack.shape[2] == len(channel_names)
            ), f"ERROR: {unpreprocessed_channel_stack.shape[2]=} is not {len(channel_names)=}"

            if preprocessor is None:
                channel_stack = unpreprocessed_channel_stack
            else:
                channel_stack = preprocessor(
                    channel_stack=unpreprocessed_channel_stack,
                    params=preprocessor_params
                )
                assert channel_stack.dtype == np.uint16

            # the index i is to sort them later
            return i, channel_stack
        
        except Exception as e:
            print(f'YOYO ERROR processing {local_file_pathed_annotations[i]}:\n{e}')
            raise e
 
    results = Parallel(
        n_jobs=min(multiprocessing.cpu_count() // 2, 32),
        backend = 'threading'
    )(delayed(load_training_frame)(i) for i in tqdm(range(len(local_file_pathed_annotations)), total = len(local_file_pathed_annotations)))
    
    results = sorted(results, key = lambda x: x[0])

    channel_stacks = [result[1] for result in results]

    for channel_stack in channel_stacks:
        assert isinstance(channel_stack, np.ndarray)
        assert channel_stack.shape[2] == len(channel_names), f"ERROR: {channel_stack.shape[2]=} but {len(channel_names)=} since {channel_names=}"
        assert channel_stack.dtype == np.uint16
    
    return channel_stacks
    
