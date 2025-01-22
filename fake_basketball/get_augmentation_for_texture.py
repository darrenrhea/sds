import albumentations as albu


def get_augmentation_for_texture(
    ad_insertion_method: str
):
    if ad_insertion_method == "ads_they_sent_to_us":
        # albu_transform = []
        albu_transform = [
            albu.AdvancedBlur(
                blur_limit=(3, 21),
                sigmaX_limit=(0.2, 10.0),
                sigmaY_limit=(0.2, 10.0),
                rotate_limit=90,
                beta_limit=(0.5, 8.0),
                noise_limit=(0.9, 4.1),
                always_apply=False,
                p=0.5
            ),
            # albu.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
            # albu.augmentations.transforms.ColorJitter(
            #     brightness=0.0,
            #     contrast=0.0,
            #     saturation=0.2,
            #     hue=0.0,
            #     p=1.0
            # ),
            # albu.RandomBrightnessContrast(
            #     brightness_limit=(-0.01, 0.2),
            #     contrast_limit=(-0.001, 0.001),
            #     p=0.5
            # ),
            # albu.ISONoise(color_shift=(0.01, 0.03), intensity=(0.0, 0.5), p=0.5),
        ]
    elif ad_insertion_method == "ad_rips":
        albu_transform = []
    else:
        raise Exception(f"ERROR: {ad_insertion_method=} is not recognized.")

    composed = albu.Compose(
        albu_transform,
        p=1.0,
        additional_targets=None
    )

    return composed