import torch
import torch.nn as nn

class AlwaysBlackModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        out = x[:, 0:1, :, :] * 0.0
        return out
     