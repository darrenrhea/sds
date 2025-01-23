def offscreen_is_labelled_as_jetblack_wrapper(
    augmentation
):
    """
    For flattened LEDs, there is a visiblity issue in that
    often part of the LED is offscreen.
    
    We want to make sure these offscreen parts are jet black.

    The way we are currently using albumentations,
    the duck type is a function that take in numpy arrays
    image, a mask, and an importance_mask, and returns a dictionary with those keys.
    """
    def blackening_augmentation(image, mask, importance_mask):
        dct = augmentation(image=image, mask=mask, importance_mask=importance_mask)
        importance_mask = dct["importance_mask"]
        mask = dct["mask"]
        image = dct["image"]
        image[importance_mask == 0, :] = 0
        ans = dict(
            image=image,
            mask=mask,
            importance_mask=importance_mask
        )
        return ans
    return blackening_augmentation

