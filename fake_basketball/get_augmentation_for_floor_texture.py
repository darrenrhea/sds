import albumentations as albu


def get_augmentation_for_floor_texture():
    """
    """

    albu_transform = [
        albu.AdvancedBlur(
            blur_limit=(3, 17),
            sigmaX_limit=(0.2, 10.0),
            sigmaY_limit=(0.2, 10.0),
            rotate_limit=90,
            beta_limit=(0.5, 8.0),
            noise_limit=(0.9, 4.1),
            always_apply=False,
            p=0.99
        ),
        # albu.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
        # albu.augmentations.transforms.ColorJitter(
        #     brightness=0,
        #     contrast=0.01,
        #     saturation=0.02,
        #     hue=0.0,
        #     p=1.0
        # ),
        albu.RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.1),
            contrast_limit=(-0.01, 0.01),
            p=1.0
        ),
        # albu.ISONoise(color_shift=(0.01, 0.03), intensity=(0.0, 0.2), p=1.0),
    ]

    composed = albu.Compose(
        albu_transform,
        p=1.0,
        additional_targets=None
    )

    return composed