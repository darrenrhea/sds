from connected_component_masks import (
     connected_component_masks
)
import numpy as np


def gbccmaeem_get_biggest_connected_component_mask_and_everything_else_mask(
    binary_hw_np_u8: np.ndarray,  # takes in a hw np u8 array with values only in {0, 1}
):

    connected_components = connected_component_masks(
        binary_hw_np_u8=binary_hw_np_u8
    )
    connected_components = sorted(
        connected_components,
        key=lambda c: (c["measure"], c["xmin"], c["ymin"],),
        reverse=True
    )

    biggest_connected_component = connected_components[0]
    biggest_connected_component_mask = biggest_connected_component["mask"]
    
    everything_else_mask = np.logical_and(
        binary_hw_np_u8,
        np.logical_not(biggest_connected_component_mask)
    ).astype(np.uint8)

    return biggest_connected_component_mask, everything_else_mask
