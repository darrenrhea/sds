import numpy as np
from prii import (
     prii
)
from collections import defaultdict
from calculate_loss_function import calculate_loss_function
from model_architecture_info import valid_model_architecture_ids
import torch
import torch.nn as nn
from colorama import Fore, Style
from calculate_model_outputs import calculate_model_outputs


def turn_output_tensors_into_alpha_matte(
    model_architecture_id: str,
    dict_of_output_tensors: dict
):
    """
    WARNING: this does sigmoid or softmax, so its output is already in [0.0, 1.0]
    Though a neural network outputs one or more tensors,
    it is not always clear how to transform that
    into an alpha matte, i.e. opacity values between 0 and 1.
    """
    if model_architecture_id == "duat":
        res1, res2 = dict_of_output_tensors["P1"], dict_of_output_tensors["P2"]
        pred = nn.functional.interpolate(res1 + res2, size=patches.shape[2:], mode='bilinear', align_corners=False)
        pred = pred.sigmoid().data.cpu().squeeze()
        mask_patches = pred
    else:
        pred = dict_of_output_tensors["outputs"]

    ghetto_sigmoid = False
    do_regression = True
    if do_regression:
        pred = pred.detach()
        # print(f"{torch.min(pred)=}")
        # print(f"{torch.max(pred)=}")
        # print(f"{torch.mean(pred)=}")
        # between_zero_and_one = pred.sigmoid()  # Should be this
        between_zero_and_one = torch.clip(input=pred, min=0.0, max=1.0)  # new training code wants this, and it isn't between 0 and 1 exactly
        mask_patches = between_zero_and_one[:, 0, :, :]
        # for k in range(pred.shape[0]):
        #     if pred[k, :, :].isnan().any():
        #         print(f"{Fore.RED}output patch {k} has nans inside!!!{Style.RESET_ALL}")
    elif ghetto_sigmoid:
        pred = torch.sigmoid(pred).detach()
        mask_patches = pred[:, 1, :, :]
    else:
        pred = torch.softmax(pred, dim=1).detach()
        mask_patches = pred[:, 1, :, :]
    
    return mask_patches
    

def infer_all_the_patches(
    model_architecture_id: str,  # something like effs, effm, effl, duat, ege to indicate which model architecture the weights are for
    model: nn.Module,  # the weights themselves stuffed into that neural network architecture
    patches: torch.Tensor  # The patches/cookies you want to infer on: a tensor of shape (4, 3, 224, 224) or (4, 3, 384, 384) or whatever
):
    """
    WARNING: this does sigmoid or softmax, so its output is already in [0.0, 1.0]
    TODO: model_architecture_info should be used to determine
    how to throw a patch of rgb through to get a mask prediction out.
    1. how to normalize (Alexnet)
    2. how to get a mask out
    3. use softmax?
    Some model architectures output multiple things.
    Depending on which architecture,
    the first or second thing might be the right one.
    """
    assert isinstance(model_architecture_id, str)
    assert model_architecture_id in valid_model_architecture_ids
    assert isinstance(model, nn.Module)
    assert isinstance(patches, torch.Tensor)
    assert patches.is_cuda, "patches must be on the GPU"
    
    dict_of_output_tensors = calculate_model_outputs(
        model=model,
        model_architecture_id=model_architecture_id,
        inputs=patches,
        train=True
    )

    ########## BEGIN determine wtf is going on:
    pred_gpu = dict_of_output_tensors["outputs"]
   
    inputs_gpu = patches
    for k in range(pred_gpu.shape[0]):
        normalized_input_cpu_np_float = inputs_gpu[k, :, :, :].permute(1, 2, 0).cpu().detach().numpy()
        input_cpu_np_float = normalized_input_cpu_np_float * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        input_np_u8 = np.round(
            (input_cpu_np_float * 255.0).clip(0, 255)
        ).astype("uint8")
        # prii(input_np_u8, caption="input in infer all the patches")
    
        pred_cpu_np = pred_gpu[k, 0, :, :].cpu().detach().numpy()
        pred_np_u8 = np.round(
            (pred_cpu_np * 255.0).clip(0, 255)
        ).astype("uint8")
        # prii(pred_np_u8, caption="pred in infer all the patches")

        # labels_cpu_np = labels_gpu[k, 0, :, :].cpu().detach().numpy()
        # labels_np_u8 = np.round(
        #     (labels_cpu_np * 255.0).clip(0, 255)
        # ).astype("uint8")
        # prii(labels_np_u8)
    #############
    mask_patches = turn_output_tensors_into_alpha_matte(
        model_architecture_id=model_architecture_id,
        dict_of_output_tensors=dict_of_output_tensors
    )
   
    # if model_architecture_id == "duat":
    #     res1, res2 = model(patches, train = True)
    #     pred = nn.functional.interpolate(res1 + res2, size=patches.shape[2:], mode='bilinear', align_corners=False)
    #     pred = pred.sigmoid().data.cpu().squeeze()
    #     mask_patches = pred
    # else:
    #     if model_architecture_id == "ege":
    #         gt_pre, pred = model(patches) 
    #     else:
    #         maybe_tuple = model(patches)
    #         if isinstance(maybe_tuple, tuple):
    #             pred = maybe_tuple[0]
    #         else:
    #             pred = maybe_tuple
    #     ghetto_sigmoid = False
    #     do_regression = True
    #     if do_regression:
    #         pred = pred.detach()
    #         between_zero_and_one = pred.sigmoid()
    #         mask_patches = between_zero_and_one[:, 0, :, :]
    #         # for k in range(pred.shape[0]):
    #         #     if pred[k, :, :].isnan().any():
    #         #         print(f"{Fore.RED}output patch {k} has nans inside!!!{Style.RESET_ALL}")
    #     elif ghetto_sigmoid:
    #         pred = torch.sigmoid(pred).detach()
    #         mask_patches = pred[:, 1, :, :]
    #     else:
    #         pred = torch.softmax(pred, dim=1).detach()
    #         mask_patches = pred[:, 1, :, :]

    return mask_patches
