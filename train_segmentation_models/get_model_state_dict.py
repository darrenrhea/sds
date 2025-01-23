from collections import OrderedDict


def get_model_state_dict(state_dict):
    model_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]
        model_state_dict[k] = v
    return model_state_dict

