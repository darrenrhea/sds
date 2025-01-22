from prii import (
     prii
)

import numpy as np

from blur import blur

def test_blur_1():

    # make a white rectangle
    alpha = np.zeros(
        shape=(1080, 1920),
        dtype=np.uint8
    )
    alpha[500:700, 500:1500] = 255
    
    prii(alpha)
    
    blurred = blur(alpha, r=7)
    
    prii(blurred)


def test_blur_2():

    # make a white rectangle
    alpha = np.zeros(
        shape=(1080, 1920, 3),
        dtype=np.uint8
    )
    alpha[500:700, 500:1500, :] = 255
    
    prii(alpha)
    
    blurred = blur(alpha, r=7)
    
    prii(blurred)


if __name__ == "__main__":
    test_blur_1()
    test_blur_2()