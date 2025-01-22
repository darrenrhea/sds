import textwrap
import skimage.draw
import numpy as np


def acptv_assign_closed_polygon_to_value(
    list_of_xys,
    value,
    victim_image_hw_and_maybe_c_np,  # victim is modified in-place
) -> None:
    """
    Given a list of (x, y) tuples in pixel coordinates,
    with y growing down as is common in computer science,
    this will set the area inside the polygon to white.

    This does not do anti-aliasing: a pixel is either set to the value or it is not.
    TODO: maybe make an anti-aliasing version.

    The image can be either hw (shape = (height, width)
    or hwc: shape = (height, width, num_channels)
    for any number of channels,
    but the value must be a scalar if hw,
    or value must be a list of length num_channels if hwc.
    but the polygon
    """
    image = victim_image_hw_and_maybe_c_np
    assert isinstance(image, np.ndarray)
    assert (
        image.ndim in {2, 3}
    ), textwrap.dedent(
        f"""\
        ERROR: the image must be either hw or hwc, but {image.ndim=}
        """
    ) 

    if image.ndim == 3:
        assert (
            isinstance(value, list)
            or
            isinstance(value, tuple)
            or
            (isinstance(value, np.ndarray) and value.ndim == 1)
        ), textwrap.dedent(
            """\
            ERROR: if the image is hwc, then the value must be a list or tuple or 1d numpy array
            """
        )

        length_of_value = (
            len(value) if isinstance(value, list) or isinstance(value, tuple) else value.shape[0]
        )
        
        assert (
            length_of_value == victim_image_hw_and_maybe_c_np.shape[2]
        ), textwrap.dedent(
            f"""\
            ERROR: the number of channels in the image
            must match the number of coordinates in the value,
            but {image.shape[2]=} whereas {len(value)=}
            """
        )
    
    height, width = image.shape[:2]
    xys = np.array(list_of_xys)
    r = xys[:, 1]
    c = xys[:, 0]

    rr, cc = skimage.draw.polygon(
        r=r,
        c=c,
        shape=(height, width)
    )

    image[rr, cc, ...] = value
    
