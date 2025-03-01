import numpy as np
import cv2


class ColorCorrector(object):
    def __init__(self, gamma):
        """
        We store the LUT
        """
        assert gamma > 0, f"{gamma=} must be greater than 0"
        self.gamma = gamma
        
        self.inverse_gamma = 1.0 / gamma
        
       
        # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values:
        self.table = np.array(
            [
                np.round(
                     ((i / 255.0) ** self.inverse_gamma) * 255
                )
		        for i in np.arange(0, 256)
            ]
        ).clip(0, 255).astype("uint8")
        assert self.table.shape == (256,), f"{self.table.shape=} must be (256,)"

        # the inverse table is the same as the table, but with the gamma value instead of the inverse_gamma value:
        self.inverse_table = np.array(
            [
                np.round(
                     ((i / 255.0) ** self.gamma) * 255
                )
		        for i in np.arange(0, 256)
            ]
        ).clip(0, 255).astype("uint8")

    def map(
        self,
        image
    ):
        assert image.ndim == 3, f"{image.ndim=} must be 3 dimensional for HWC"
        assert image.shape[2] == 3, f"{image.shape=} must have 3 channels for RGB or BGR"
        answer = cv2.LUT(image, self.table)
        assert answer.shape == image.shape, f"{answer.shape=} must be the same as {image.shape=}"
        assert answer.dtype == np.uint8, f"{answer.dtype=} must be np.uint8"
        return answer

    def inverse_map(
        self,
        image
    ) -> np.ndarray:
        assert image.ndim == 3, f"{image.ndim=} must be 3 dimensional for HWC"
        assert image.shape[2] == 3, f"{image.shape=} must have 3 channels for RGB or BGR"
        answer = cv2.LUT(image, self.inverse_table)
        assert answer.shape == image.shape, f"{answer.shape=} must be the same as {image.shape=}"
        assert answer.dtype == np.uint8, f"{answer.dtype=} must be np.uint8"
        return answer
        

