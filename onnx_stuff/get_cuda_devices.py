import torch

# get available cuda devices
def get_cuda_devices(device_index = None):
    num_devices = torch.cuda.device_count()
    if num_devices > 1:
        devices = tuple([torch.device("cuda:{:d}".format(i)) for i in range(num_devices)])
    else:
        num_devices = 1
        devices = (torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), )
    # print('{} device(s): {}'.format(len(devices), devices))
    return devices


