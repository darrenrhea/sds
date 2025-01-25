import cv2
from typing import Optional
import numpy as np


def load_frame(frame_path, inference_width, inference_height) -> Optional[np.ndarray]:
    try:
        frame_path = str(frame_path)
        full_sized_frame = cv2.imread(str(frame_path))  # we think this removes any alpha channel
        if full_sized_frame is None:
            raise Exception('error loading!')
        
        # TODO: this seems dangerous.
        frame = cv2.resize(
            full_sized_frame,
            (inference_width, inference_height)
        )
        
        if frame.shape[2] == 4:
            raise Exception('alpha channel detected?! This should not happen without flag cv2.IMREAD_UNCHANGED')
        else:
            return frame
    except Exception as e:
        print(f'ERROR processing {frame_path}:\n{e}')
        return None
