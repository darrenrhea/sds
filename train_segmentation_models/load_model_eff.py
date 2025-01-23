from construct_model_eff import (
     construct_model_eff
)
from get_model_state_dict import (
     get_model_state_dict
)
import torch



def load_model_eff(model_size, path = None, num_class = 2, version = 1, multigpu = False, in_channels=3, arch='u'):
    assert model_size in ["s", "m", "l"], "model size must be s, m, l"
    assert arch in ["u", "u++", "ma"], "arch must be u, u++, or ma"

    assert num_class == 1
    
    model = construct_model_eff(
        model_size,
        num_class = num_class,
        return_features = True,
        in_channels=in_channels,
        arch=arch)[0]
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
