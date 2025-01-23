import torch


def unweighted_l1_loss_per_channel(
    input,  # the prediction the neural network made
    target, # the label / actual truth
):
    """
    See also weighted_l1_loss.py

    Given two (gpu) torch f32 tensors input and target
    all of the same 4 dimensional shape B x C x H x W
    calculate the average (per pixel) l1 loss between input and target.
    Like predicting constantly 0.5 would have a loss of 0.5 for binary targets.
    """
    assert input.ndim == 4
    assert target.ndim == 4
    assert input.size() == target.size()
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
    absolute_values = torch.abs(diff)
    summed = absolute_values.sum(axis=(0, 2, 3))
    assert summed.ndim == 1
    denominator = input.shape[0] * input.shape[2] * input.shape[3]
    answer = summed / denominator
    return answer


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

    # we have to divide by this constant to match the previous behavior when all weights are 1.0