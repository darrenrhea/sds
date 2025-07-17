from construct_model_duat import (
     construct_model_duat
)
from get_model_state_dict import (
     get_model_state_dict
)
import torch


def load_model_duat(path=None, version = 1, multigpu = True, in_channels = 3, num_class = 2):
    assert num_class == 2, "duat only supports binary classification"

    model = construct_model_duat()

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

