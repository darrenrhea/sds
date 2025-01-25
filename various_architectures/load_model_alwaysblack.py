from construct_model_alwaysblack import (
     construct_model_alwaysblack
)


def load_model_alwaysblack(path = None, num_class = 2, version = 1, multigpu = False, in_channels=3):
    assert num_class == 1
    
    model = construct_model_alwaysblack(
        num_class = num_class,
        in_channels=in_channels,
    ) 
    model.eval()
    return model
