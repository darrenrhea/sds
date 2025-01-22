from typing import Tuple
import numpy as np

def draw_marker_on_np_u8(
    xy: Tuple[int, int],
    victim: np.array,
    r: int = 50
):
    """
    Takes in a 3 or 4 channel hwx uint8 numpy array
    and draws a cross on it
    """
    assert isinstance(victim, np.ndarray)
    height = victim.shape[0]
    width = victim.shape[1]
    num_channels = victim.shape[2]
    assert num_channels in [3, 4]
    x, y = xy
    if num_channels == 3:
        color = [0, 255, 0]
    else:
        color = [0, 255, 0, 255]
            
    victim[max(0, y-r):min(height, y+r), x] = color  # vertical line
    victim[y, max(0, x-r):min(width, x+r)] = color  # horizontal line
