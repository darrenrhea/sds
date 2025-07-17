import cv2
import torch
import torch.onnx
import time
from Patcher import Patcher
import numpy as np
from infer_all_the_patches import infer_all_the_patches
from typing import List
from convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device import convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device



def process_frame(
    patcher,
    with_amp,
    model,
    model_architecture_id,
    frame_idx,  # why
    frame_bgr,
    transform,
    device
):
    """
    currently only used for the non-parallel version of inference.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    frame_tens = transform(convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device(frame_rgb))
    patches = patcher.patch(frame = frame_tens, device = device)
    # print(f"{Fore.YELLOW}{patches.shape=}{Style.RESET_ALL}")
    

    # infer
    with torch.no_grad():
        with with_amp():
            t0 = time.time()
            mask_patches = infer_all_the_patches(
                model_architecture_id=model_architecture_id,
                model=model,
                patches=patches
            )

    stitched = patcher.stitch(mask_patches)
    stitched = torch.clip(stitched * 255.0, 0, 255).type(torch.uint8)
    stitched = stitched.cpu().numpy()
    t1 = time.time()
    dt = (t1 - t0) * 1000.0
    return frame_idx, stitched, dt