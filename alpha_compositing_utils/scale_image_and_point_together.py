import PIL.Image
import numpy as np

def scale_image_and_point_together(
    rgba_np_u8,
    top_xy,
    scale_factor
):
    """
    This seems like something albumentations should be able to do
    with the keypoint type.
    Scale the cutout_image and the point together so that
    it stays on there foot bottom.
    """
    assert scale_factor > 0, f"{scale_factor=} but cannot be negative"
    
    image_pil = PIL.Image.fromarray(rgba_np_u8)
    
    downsampled_image_pil = image_pil.resize(
        (
            round(scale_factor * image_pil.width),
            round(scale_factor * image_pil.height)
        )
    )
    scaled_rgba_np_u8 = np.array(downsampled_image_pil)
    x = np.array(
        [
            top_xy[0],
            top_xy[1],
        ],
        dtype=np.float64
    )
    v = x * scale_factor
    
    new_top_xy = (
        int(round(v[0])),
        int(round(v[1])),
    )

    return scaled_rgba_np_u8, new_top_xy