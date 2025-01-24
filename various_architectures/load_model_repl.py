from construct_model_repl import (
     construct_model_repl
)
from get_model_state_dict import (
     get_model_state_dict
)

import torch



def load_model_repl(model_size, path=None, num_class = 2, version = 1, multigpu = False, in_channels=3):
    if path and '_w_' in path:
        num_class = 1

    model = construct_model_repl(
        model_size,
        num_class = num_class,
        return_features = True,
        in_channels=in_channels)[0]

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
