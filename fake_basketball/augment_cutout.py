import pprint as pp
from typing import Tuple
import numpy as np
import albumentations as albu


def augment_cutout(
    rgba_np_u8: np.ndarray,  # the cutout
    top_xy: Tuple[int, int],  # the top left corner of the cutout
    transform: albu.core.composition.Compose
) -> np.ndarray:
    """
    Use albumentations to vary the cutout,
    changing color, rotation, noise, blur, etc.
    """
    assert isinstance(rgba_np_u8, np.ndarray)
    assert isinstance(transform, albu.core.composition.Compose)
    image = rgba_np_u8[:, :, :3]
    mask = rgba_np_u8[:, :, 3]
    
    keypoints = [
        (top_xy[0], top_xy[1]),
    ]

    transformed = transform(
        image=image,
        mask=mask,
        keypoints=keypoints
    )

    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    transformed_keypoints = transformed["keypoints"]
    assert len(transformed_keypoints) == 1, f"{transformed_keypoints=}"
    
    new_rgba_np_u8 = np.concatenate(
       (transformed_image, transformed_mask[:,:,np.newaxis]),
       axis=2
    )
    
    new_top_xy = (
        int(transformed_keypoints[0][0]),
        int(transformed_keypoints[0][1]),
    )
    # print(f"{top_xy=}\nbecomes\n{new_top_xy=}\n\n")
    return new_rgba_np_u8, new_top_xy
