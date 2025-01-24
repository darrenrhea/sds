import torch
import numpy as np


def convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device(img):
    """
    This function converts a numpy image from
    cpu numpy HWC to gpu torch CHW,
    from uint8 to float16, and
    from [0, 255] to [0.0, 1.0].
    """
    assert isinstance(img, np.ndarray), "ERROR: img must be a numpy array"
    assert img.dtype == np.uint8, "ERROR: img must be of type np.uint8"
    img = torch.from_numpy(img).cuda().half().permute(2, 0, 1)
    img = img / 255.0
    return img
