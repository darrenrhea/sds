from prii import (
     prii
)

import numpy as np



def show_color_map_for_color_correction(
    color_map: np.ndarray,
    square_size: int = 100,
):
    """
    given a uint8 np.ndarray of shape (N, 2, 3),
    which is interpreted as a list of N pairs of map this color to this color
    show the color map as a big color splotch image.
    """
    assert isinstance(color_map, np.ndarray), f"color_map should be an np.ndarray, not {type(color_map)}"
    assert color_map.ndim == 3, f"color_map.ndim should be 3, not {color_map.ndim}"
    assert color_map.shape[1] == 2, f"color_map.shape[1] should be 2, not {color_map.shape[1]=}"
    assert color_map.shape[2] == 3, f"color_map.shape[2] should be 3, not {color_map.shape[2]=}"
    # assert color_map.dtype == np.uint8, f"color_map.dtype should be np.uint8, not {color_map.dtype=}"
    # np.printoptions(threshold=np.inf)
    # print(color_map)
    bigger = np.round(color_map).clip(0, 255).astype(np.uint8).repeat(square_size, axis=0).repeat(square_size, axis=1)
    print(bigger.shape)
    prii(bigger)

  

    

    


   

