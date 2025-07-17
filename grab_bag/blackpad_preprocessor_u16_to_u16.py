import numpy as np


def blackpad_preprocessor_u16_to_u16(
    channel_stack: np.ndarray,
    params: dict
):
    """
    Preprocessors like this one, blackpad_preprocessor_u16_to_u16, are used
    on stacks_of_channels, at a minimum the 3 rgb channels and one mask channel,
    but sometimes many more channels for the aspect of multiple prediction tasks (depth_map is an example u16 data)
    or importance weighting the loss function.

    You may want to predict multiple target_masks to reuse the lower layers' computation
    to predict several segmentation conventions at once, or you may want
    weight_masks / importance_masks / relevance_masks
    to indicate whether a region of the image is important to get correct or not.

    Pad the image and masks by extra black pixels at the bottom and right.
    Sometimes it is nice to be able to act like the original image and
    the associated target_masks and weight_masks/relevance_masks are 1920x1088
    even when they are really 1920x1080 for number-theoretic constraints,
    like effs/effm want height and width to be divisible by 32.

    Pads the image and its various masks by black pixels.

    One can imagine that we want to train a model to predict a variety of target_masks,
    each with its own relevance_mask / weight_mask / importance_mask to have dont-care's
    in the loss function.

    See also reflect_preprocessor.py
    """
    assert isinstance(channel_stack, np.ndarray)
    assert channel_stack.ndim == 3, "ERROR: channel_stack should have 3 dimensions"
    
    num_channels = channel_stack.shape[2]

    assert isinstance(channel_stack, np.ndarray)
    assert (
        channel_stack.shape[2] >= 3
    ), f"ERROR: The channel_stack should probably have at least 3 channels but it has only {channel_stack.shape[2]=} channels"
    assert isinstance(params, dict)
    assert "desired_height" in params, "ERROR: desired_height must be in params for reflect_preprocessor"
    assert "desired_width" in params, "ERROR: desired_width must be in params for reflect_preprocessor"

    desired_height = params['desired_height']
    desired_width = params['desired_width']

    raw_height = channel_stack.shape[0]
    raw_width = channel_stack.shape[1]
    delta_height = desired_height - raw_height
    delta_width = desired_width - raw_width
    assert delta_height >= 0, f"ERROR: {desired_height=} but {raw_height=}"
    assert delta_width >= 0, f"ERROR: {desired_width=} but {raw_width=}"

    # pad the frame with black pixels:
    padded_frame = np.zeros((desired_height, desired_width, num_channels), dtype = np.uint16)
    padded_frame[:raw_height, :raw_width, :] = channel_stack

    assert padded_frame.shape[0] == desired_height
    assert padded_frame.shape[1] == desired_width
    assert padded_frame.shape[2] == num_channels, "ERROR: somehow we changed the number of channels"
    assert padded_frame.ndim == 3
    assert padded_frame.dtype == np.uint16

    return padded_frame

