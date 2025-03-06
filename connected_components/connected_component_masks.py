from typing import List
import cv2
import numpy as np


def connected_component_masks(
    binary_hw_np_u8: np.ndarray,  # takes in a hw np u8 array with values only in {0, 1}
) -> List[dict]:
    """
    This finds each connected component of a binary mask as a binary mask.

    It is naive to think that two connected components can
    always be separated by a bounding box.  Although
    a polygon with hole polygons can isolate a connected component,
    the programming complexity is way too high for now.
   
    Yet the bounding box turns out to be quite useful for
    eliminating connected components that are half off the image,
    like a person who is half off the left side of the stage.

    given a binary mask binary_hw_np_u8,
    return a list of dicts,
    one for each connected component,
    with the following attributes of the connected component:
    
    - xmin
    - xmax
    - ymin
    - ymax
    - measure: how many pixels are in the connected component
    - mask: the mask of the connected component, a hw np u8 array with values only in {0, 1}, 1 indicating the component
    - label  # not sure this is useful, a positive integer naming the component, a unique identifier for the component

    """

    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(
        binary_hw_np_u8,
        connectivity=8,
    )
    
    connected_components = []
    for i in range(numLabels):
        if i == 0:
            continue  # the opposite is label 0?
        
        mask = (labels == i).astype(np.uint8)

        x0, y0, bbox_width, bbox_height, measure = stats[i]
       
        component = dict(
            xmin=x0,
            xmax=x0 + bbox_width,
            ymin=y0,
            ymax=y0 + bbox_height,
            label=i,
            measure=measure,
            mask=mask
        )
        connected_components.append(component)

    return connected_components
