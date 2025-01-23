import numpy as np
import torch
import torch.nn.functional as F
from colorama import Fore, Style

def weighted_mse_loss(
    input,  # the prediction the neural network made
    target, # the label / actual truth
    importance_weights,  # spacial importance weights to value being correct in some places more than others
):
    """
    Given three (gpu) torch f32 tensors input and target and importance weights
    all of the same 4 dimensional shape,
    calculate the weighted mean squared error loss between input and target.

    Note: when all importance_weights are 1.0,
    the result should be the same as
    torch.nn.MSELoss(input, target, reduction='mean').
    
    This will allow us to have no behavior change so long as you send in
    all ones for the importance_weights.

    But if all the importance_weights are not 0.0, then the result should be zero.
    This is because the importance_weights are input mask that indicates which pixels
    should be ignored in the loss calculation.

    Batch x Chan x Height x Width
    replicate 
    mse_loss(input, target, reduction='mean')
    where mse_loss = torch.nn.MSELoss()
    
    or equivalently:
    F.mse_loss(input, target, reduction='mean')

    manually
    calculate the mean squared error loss between them.

    https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html

    """
    # assert requirements:
    assert input.ndim == 4
    assert target.ndim == 4
    assert input.size() == target.size()
    assert input.size() == importance_weights.size()
    # this assert depends on AMP or not:
    assert (
        input.dtype == torch.float32
        or
        input.dtype == torch.float16
    )
    assert (
        target.dtype == torch.float32
        or
        target.dtype == torch.float16
    )

    diff = input - target
    # do L^2:
    # loss_prior_to_spacial_weighting = diff * diff
    
    # or do L1:
    loss_prior_to_spacial_weighting = torch.abs(diff)


    # or "weight the two types of errors differently"
    # when the prediction "input" is 0
    # but the target/label is 1 but the prediction is 0, the loss/punishment is 1
    # when the prediction "input" is 1
    # but the target is 0, the loss/punishment is 0.2

    # loss_prior_to_spacial_weighting = (
    #     torch.relu(target - input)
    #     +
    #     1.0 * torch.relu(input - target)
    # )

    # when

    weighted = loss_prior_to_spacial_weighting * importance_weights

    # we have to divide by this constant to match the previous behavior when all weights are 1.0
    denominator = input.shape[0] * input.shape[1] * input.shape[2] * input.shape[3]
    summed = weighted.sum()
    answer = summed / denominator
    return answer

