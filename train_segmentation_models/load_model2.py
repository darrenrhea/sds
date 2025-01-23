from SResUnet import (
     SResUnet
)
from get_model_state_dict import (
     get_model_state_dict
)


import torch

from torchvision import models


def load_model2(path=None, version = 3, multigpu = False):
    num_class = 2
    # resnet34 loaded from pytorch
    encoder = models.resnet34(pretrained=True)
    model = SResUnet(encoder, out_channels = num_class)

    if path is None:
        return model

    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    #print(list(checkpoint.keys()))

    if multigpu:
        model_state_dict = get_model_state_dict(checkpoint['model'])
    else:
        model_state_dict = checkpoint['model']

    model.load_state_dict(model_state_dict)
    model.eval()
    return model
