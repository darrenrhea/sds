import torch
from ssim_loss import SSIMLoss, MS_SSIMLoss

def test_ssim_loss():
    # dim: batch x class x height x width
    pred = torch.rand((1, 1, 256, 256))
    target = torch.rand((1, 1, 256, 256))
    
    criterion = SSIMLoss()
    rst = criterion(pred, target)
    print('ssim loss', rst)

    criterion = MS_SSIMLoss()
    rst = criterion(pred, target)
    print('ms-ssim loss', rst)


if __name__ == '__main__':
    test_ssim_loss()