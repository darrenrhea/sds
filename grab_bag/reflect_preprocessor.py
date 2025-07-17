import numpy as np


def reflect_preprocessor(
    channel_stack: np.ndarray,
    params: dict
):
    """
    Pads the stack of image and maybe severl types of target_masks and weight_masks,
    by reflection at the bottom and right.
    Why?

    Sometimes it is nice to be able to act like the original image and
    the associated target_masks and weight_masks/relevance_masks are 1920x1088
    even when they are really 1920x1080 for number-theoretic constraints,
    like effs/effm want height and width to be divisible by 32.


    Preprocessors like this one, reflect_preprocessor, are used
    on stacks of channels, usually the 3 rgb channels and 1 mask channels.
    You may want to predict multiple target_masks to reuse the lower layers' computation
    to predict several segmentation conventions at once, or you may want
    weight_masks / importance_masks / relevance_masks
    to indicate whether a region of the image is important to get correct or not.

    
    One can imagine that we want to train a model to predict a variety of target_masks,
    each with its own relevance_mask / weight_mask / importance_mask to have dont-care's
    in the loss function.

    TODO: don't duplicate a row of pixels when you reflect.
    """
    assert isinstance(raw_frame, np.ndarray)
    assert isinstance(raw_target_mask, np.ndarray)
    assert isinstance(raw_weight_mask, np.ndarray)
    assert raw_frame.shape[0] == raw_target_mask.shape[0]
    assert raw_frame.shape[1] == raw_target_mask.shape[1]
    assert raw_frame.shape[0] == raw_weight_mask.shape[0]
    assert raw_frame.shape[1] == raw_weight_mask.shape[1]
    assert raw_frame.shape[2] == 3
    assert raw_target_mask.ndim == 2
    assert raw_weight_mask.ndim == 2
    assert isinstance(params, dict)
    assert "desired_height" in params, "ERROR: desired_height must be in params for reflect_preprocessor"
    assert "desired_width" in params, "ERROR: desired_width must be in params for reflect_preprocessor"

    desired_height = params["desired_height"]
    desired_width = params["desired_width"]
    raw_height = raw_frame.shape[0]
    raw_width = raw_frame.shape[1]
    delta_height = desired_height - raw_height
    delta_width = desired_width - raw_width

    padded_frame = np.zeros((desired_height, desired_width, 3), dtype = np.uint8)
    padded_target_mask = np.zeros((desired_height, desired_width), dtype = np.uint8)
    padded_frame[:raw_height, :raw_width, :] = raw_frame
    
    padded_frame[raw_height:desired_height, raw_width:desired_width, :] = \
        raw_frame[
            (raw_height - delta_height):raw_height,
            (raw_width - delta_width):raw_width,
            :
        ][::-1, ::-1, :]
    
    padded_frame[raw_height:desired_height, :raw_width, :] = \
        raw_frame[
            (raw_height - delta_height):raw_height,
            :raw_width,
            :
        ][::-1, :, :]

    padded_frame[:raw_height, raw_width:desired_width, :] = \
        raw_frame[
            :raw_height,
            (raw_width - delta_width):raw_width,
            :
        ][:, ::-1, :]

    padded_target_mask[raw_height:desired_height, raw_width:desired_width] = \
        raw_target_mask[
            (raw_height - delta_height):raw_height,
            (raw_width - delta_width):raw_width
        ][::-1, ::-1]
    
    padded_target_mask[raw_height:desired_height, :raw_width] = \
        raw_target_mask[
            (raw_height - delta_height):raw_height,
            :raw_width
        ][::-1, :]
    
    padded_target_mask[:raw_height, raw_width:desired_width] = \
        raw_target_mask[
            :raw_height,
            (raw_width - delta_width):raw_width
        ][:, ::-1]
    

    padded_target_mask[:raw_height, :raw_width] = raw_target_mask
    
    return padded_frame, padded_target_mask



