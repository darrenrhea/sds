
import numpy as np

def reverse_transform(inp):
    """
    takes a CHW torch tensor
    which is AlexNet normalized
    and returns a HWC numpy array
    reverse RGB color normalization.
    Few users.
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    return inp
