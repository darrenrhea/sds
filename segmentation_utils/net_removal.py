from pathlib import Path
from all_imports_for_image_segmentation import *
import albumentations as A
import cv2
from image_displayers_for_jupyter import display_numpy_hw_grayscale_image
import PIL
import PIL.Image
from fastai.vision.all import *
import matplotlib.pyplot as plt
import numpy as np
import torch
import pprint as pp
from TernausNet import UNet16
from print_image_in_iterm2 import print_image_in_iterm2


def infer_cookie(cookie_hwc_rgb_np_uint8, model, torch_device):
    chw_np_uint8 = np.transpose(cookie_hwc_rgb_np_uint8, axes=(2, 0, 1))

    # convert the image to float32s ranging over [0,1]:
    chw_np_float32 = chw_np_uint8[:, :, :].astype(np.float32) / 255.0

    # normalize it like AlexNet:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (chw_np_float32 - mean[..., None, None]) / std[..., None, None]
    model.eval()  

    batch_size = 1
    batch_of_tiles = normalized[np.newaxis, :, :, :]

    xb_cpu = torch.tensor(batch_of_tiles)
    xb = xb_cpu.to(torch_device)
    out = model(xb)

    out_cpu_torch = out.detach().cpu()
    probs_np = out_cpu_torch.numpy()


    for k in range(probs_np.shape[0]):     
        chw = np.array(xb_cpu[0, :3, :, :])

        feathered_prediction_float = probs_np[k, 0, :, :]
        
        feathered_prediction_uint8 = np.clip(
            feathered_prediction_float * 255.0, 0, 255
        ).astype(np.uint8)
            
        color_rgb_uint8 = cookie_hwc_rgb_np_uint8
        color_float32 = color_rgb_uint8.astype(np.float32)
        color_pil_image = PIL.Image.fromarray(color_rgb_uint8)
        feathered_pil_image = PIL.Image.fromarray(feathered_prediction_uint8)
        
        uncorrupt = (color_float32 - 0 * feathered_prediction_float[:, :, np.newaxis]) / (1.000001 - feathered_prediction_float[:, :, np.newaxis])
        
        uncorrupt_pil_image = PIL.Image.fromarray(np.round(uncorrupt).clip(0, 255).astype(np.uint8))
        
        print_image_in_iterm2(rgb_np_uint8=cookie_hwc_rgb_np_uint8)
        print("original:")
        print_image_in_iterm2(image_pil=color_pil_image)
        print("restored:")
        print_image_in_iterm2(image_pil=uncorrupt_pil_image)
        print("net:")
        print_image_in_iterm2(grayscale_np_uint8=feathered_prediction_uint8)
            


if __name__ == "__main__":


    nn_input_height = 384
    nn_input_width = 384

    full_size_image_path = Path("STLvPIT_2020-07-25_329000.jpg").resolve()
    # full_size_image_path = Path("~/synthetic_nets/synthetic_000000.jpg").expanduser().resolve()
    full_size_image_pil = PIL.Image.open(full_size_image_path)
    full_size_image_np = np.array(full_size_image_pil)
    i0 = 320
    j0 = 880
    cookie_hwc_rgb_np_uint8 = full_size_image_np[i0:i0+nn_input_height, j0:j0+nn_input_width, :]
    


    torch_device = my_pytorch_utils.get_the_correct_gpu(
        substring_of_the_name="8000",
        which_copy=1
    )

    model = UNet16(
        num_classes=1,
        num_filters=32,
        pretrained=False,
        is_deconv=False
    )

    model_path = Path(f"~/r/trained_models/net_384x384_2023-07-30.tar").expanduser()
    dct = torch.load(model_path)
    model.load_state_dict(dct['model_state_dict'])

    model.to(torch_device) 

    infer_cookie(
        cookie_hwc_rgb_np_uint8=cookie_hwc_rgb_np_uint8,
        model=model,
        torch_device=torch_device
    )