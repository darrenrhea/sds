import torch
import numpy as np


def convert_hwc_np_f32_image_to_chw_torch_f16_on_device(hwc_np_f32):
    """
    This function converts a H x W X C np.float32 image
    (usually values are in the range [0.0, 1.0])
    to 
    cpu numpy HWC to gpu torch CHW,
    from uint8 to float16, and
    from [0, 255] to [0.0, 1.0].
    """
    assert isinstance(hwc_np_f32, np.ndarray), "ERROR: hwc_np_f32 must be a numpy array"
    assert hwc_np_f32.dtype == np.float32, "ERROR: hwc_np_f32 must be of type np.float32"
    hwc_np_f32 = torch.from_numpy(hwc_np_f32).cuda().half().permute(2, 0, 1)
    return hwc_np_f32
