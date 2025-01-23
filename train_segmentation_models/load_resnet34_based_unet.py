from get_model_state_dict import (
     get_model_state_dict
)
from construct_model_resnet34_based_unet import (
     construct_model_resnet34_based_unet
)

import torch

def load_resnet34_based_unet(
        path=None,
        num_class=2,
        multigpu = False,
        in_channels=3
):
    assert num_class == 1
    if path and '_w_' in path:
        num_class = 1
    
    model = construct_model_resnet34_based_unet(
        num_class = num_class,
        in_channels=in_channels,
    )
    if not path:
        return model

    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    # print('checkpoint keys', list(checkpoint.keys()))

    if multigpu:
        model_state_dict = get_model_state_dict(checkpoint['model'])
    else:
        model_state_dict = checkpoint['model']

    model.load_state_dict(model_state_dict)
    model.eval()
    return model

