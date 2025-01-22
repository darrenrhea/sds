from UNet1 import UNet1
from UNet2 import UNet2

from pathlib import Path
from stride_score_image import stride_score_image
import time
import PIL
import numpy as np

from pathlib import Path
import pprint as pp
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import my_pytorch_utils


def get_the_torch_device_and_model(model_id: str, model_architecture: str):
    """
    This returns the torch_device / GPU to use and the model
    where the model has been put into eval mode and is on that GPU already.
    """
    # Choose the right gpu:
    torch_device = my_pytorch_utils.get_the_correct_gpu("Quadro", which_copy=0)

    nn_input_width = 224  # after possible data-augmentation, the 256x256 is cropped to 224x224
    nn_input_height = nn_input_width

    # make an instance of the neural network
    if model_architecture == "UNet1":
        model = UNet1(
            nn_input_height=nn_input_height,
            nn_input_width=nn_input_width,
            in_channels=3,
            out_channels=2  # for binary you want 2 out_channels
        )
    elif model_architecture == "UNet2":
        model = UNet2(
            nn_input_height=nn_input_height,
            nn_input_width=nn_input_width,
            in_channels=3,
            out_channels=2  # for binary you want 2 out_channels
        )

    path_to_load_model_from = Path(f"~/r/trained_models/{model_id}.tar").expanduser()
    print(f"Going to load the PyTorch model from {path_to_load_model_from}")
    assert path_to_load_model_from.is_file(), f"{path_to_load_model_from} does not exist"
    dct = torch.load(path_to_load_model_from)
    model.load_state_dict(dct['model_state_dict'])

    # possibly after loading it full of weights, you got to move the model onto the gpu or else bad things
    # about Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
    # will happen:
    model = model.to(torch_device)  # stuff it onto the gpu
    model.eval()  # this is important!  batch_norm makes train mode not deterministic.
    return torch_device, model 
