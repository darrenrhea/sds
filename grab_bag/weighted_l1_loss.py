import torch


def weighted_l1_loss(
    input,  # the prediction the neural network made
    target, # the label / actual truth
    importance_weights,  # spacial importance weights to value being correct in some places more than others
):
    """
    Given three (gpu) torch f32 tensors input and target and importance weights
    all of the same 4 dimensional shape,
    calculate the weighted mean squared error loss between input and target.
    If all the importance_weights are 0.0, then the result should be zero.
  

    """
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



    weighted = loss_prior_to_spacial_weighting * importance_weights

    # we have to divide by this constant to match the previous behavior when all weights are 1.0
    denominator = input.shape[0] * input.shape[1] * input.shape[2] * input.shape[3]
    summed = weighted.sum()
    answer = summed / denominator
    return answer

