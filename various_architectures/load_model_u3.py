from construct_model_u3 import (
     construct_model_u3
)
from get_model_state_dict import (
     get_model_state_dict
)

import torch


def load_model_u3(
        backbone,
        path=None,
        num_class = 2,
        version = 1,
        multigpu = False,
        in_channels=3,
        transpose_final=False
):

    model, _ = construct_model_u3(backbone, in_channels=in_channels, num_class = num_class, include_classification = False, return_features = False)

    if path is None:
        model.eval()
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
