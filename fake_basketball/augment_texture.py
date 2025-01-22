import numpy as np
import albumentations as albu


def augment_texture(
    rgb_np_u8: np.ndarray,  # the cutout
    transform: albu.core.composition.Compose
) -> np.ndarray:
    """
    Use albumentations to vary the ad_texture by blurring,
    changing brightness, contrast, color, hue, noise, etc.
    """
    assert isinstance(rgb_np_u8, np.ndarray)
    assert isinstance(transform, albu.core.composition.Compose)
    image = rgb_np_u8
    
    transformed = transform(
        image=image
    )

    transformed_image = transformed["image"]
    assert transformed_image.dtype == np.uint8
    assert transformed_image.shape  == image.shape
    return transformed_image
  