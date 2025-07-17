from torch.utils.data import Dataset
from unettools import reverse_transform
from print_image_in_iterm2 import print_image_in_iterm2

def visualize_dataset(
    data_set: Dataset,
    num_samples: int,
):
    """
    This can be hard because a PyTorch Dataset has already
    been normalized, so you have to reverse the normalization.
    Also, the labels y are variously shaped:

    """
    assert isinstance(data_set, Dataset)
    cntr = 0
    for x, y in data_set:
        if cntr >= num_samples:
            break
        print(f"{x.shape=}")
        print(f"{y.shape=}")
        
        cntr += 1
        x_cpu = x.detach().cpu()
        hwc_np = reverse_transform(x_cpu)
        print_image_in_iterm2(rgb_np_uint8=hwc_np)
        if y.ndim == 3 and y.shape[0] == 2:
            y = y[0, :, :]
            thing = (y.detach().cpu().numpy() * 255).round().astype('uint8')
            print_image_in_iterm2(grayscale_np_uint8=thing[1, :, :])
   
        elif y.ndim == 3 and y.shape[0] == 1:
            thing = (y.detach().cpu().numpy() * 255).round().astype('uint8')
            print_image_in_iterm2(grayscale_np_uint8=thing[0, :, :])

        elif y.ndim == 2:
            thing = (y.detach().cpu().numpy() * 255).round().astype('uint8')
            print_image_in_iterm2(grayscale_np_uint8=thing)
