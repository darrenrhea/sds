from construct_model_ege import (
     construct_model_ege
)
from get_model_state_dict import (
     get_model_state_dict
)


import torch


def load_model_ege(path=None, version = 1, multigpu = False, num_class = 2, in_channels = 3):
    assert in_channels == 3, "ege only supports 3 channels"
    model = construct_model_ege(num_class=num_class)

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
