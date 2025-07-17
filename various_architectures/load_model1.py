#silly
from get_model_state_dict import (
     get_model_state_dict
)

from ResNetUNet import (
     ResNetUNet
)

import torch


def load_model1(path = None, version = 1, multigpu = False):
    num_class = 2
    model = ResNetUNet(n_class=num_class)

    if path is None:
        return model

    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    print('checkpoint keys', list(checkpoint.keys()))

    if multigpu:
        model_state_dict = get_model_state_dict(checkpoint['model'])
    else:
        model_state_dict = checkpoint['model']

    model.load_state_dict(model_state_dict)

    model.eval()
    return model
