import os
import sys
from pathlib import Path
from typing import List


def get_frames_and_masks_of_dataset(
    list_of_dataset_folders: List[Path],
    dataset_kind: str,
):
    """
    Who knows what we might use as a dataset in the future, but
    right now a dataset if a finite list of directories that contain frames and masks of the same name.
    This function is meant to be a single point of entry for all datasets.
    Todo: add exclude functionality?
    """
    assert dataset_kind in ['nonfloor', 'nonwood']

    input_frames_fn = []
    target_masks_fn = []

    print("Gathering all training points from these directories:")
    for folder in list_of_dataset_folders:
        print(f"    {folder}")

    for folder in list_of_dataset_folders:
        print(f'loading files from {folder}')
        fn_masks = [str(p) for p in Path(folder).rglob(f'*_{dataset_kind}.png')]
        fn_frames = [fn.replace(f'_{dataset_kind}.png', '.jpg') for fn in fn_masks]

        include = []
        for i, (fn_frame, fn_mask) in enumerate(zip(fn_frames, fn_masks)):
            if not os.path.exists(fn_frame) or not os.path.exists(fn_mask): #  or exclude(fn_frame, fn_mask):
                print(f"IGNORING: {fn_frame} because:")
                print(f"{os.path.exists(fn_frame)=}")
                print(f"{os.path.exists(fn_mask)=}")
                # print(f"{exclude(fn_frame, fn_mask)=}")
                continue
            include.append(i)

        if len(include) == 0:
            print(f'ERROR: no valid (frame, mask) pairs found in directory {folder}, usually this is an error.')
            sys.exit(1)
        

        fn_frames = [fn_frames[j] for j in include]
        fn_masks = [fn_masks[j] for j in include]

        print('\n'.join(fn_masks))

        input_frames_fn.extend(fn_frames)
        target_masks_fn.extend(fn_masks)

        print(f'found {len(input_frames_fn)} frames, {len(target_masks_fn)} masks')
        assert len(input_frames_fn) == len(target_masks_fn)

    
    return input_frames_fn, target_masks_fn
