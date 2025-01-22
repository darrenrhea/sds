import albumentations as albu
import cv2  # cv2.BORDER_CONSTANT below


def get_balls_augmentation():
    albu_transform = [
        albu.HorizontalFlip(p = 0.5),
    ]
    return albu_transform

    albu_transform = [
        # This is a good idea but we need the scale to be preserved
        # albu.augmentations.geometric.resize.LongestMaxSize(
        #     max_size=100,
        #     interpolation=1,
        #     always_apply=True,
        #     p=1
        # ),
        albu.augmentations.crops.transforms.CropAndPad(  # this is padding so that ElasticTransform and PiecewiseAffine don't "sample from off the texture"
            px=100,
            percent=None,
            pad_mode=0,
            pad_cval=0,
            pad_cval_mask=0,
            keep_size=False,  # we want it to grow
            sample_independently=True,
            interpolation=1,
            always_apply=True,
            p=1.0
        ),
        albu.augmentations.geometric.rotate.Rotate(
            limit=90,
            interpolation=1,
            border_mode=cv2.BORDER_CONSTANT,
            value=None,
            mask_value=None,
            rotate_method='largest_box',
            crop_border=False,
            always_apply=True,
            p=1.0
        ),
        albu.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1),
            contrast_limit=0.1,
            always_apply=True,
            p=1.0
        ),
        albu.OneOf(
            [
                albu.Blur(blur_limit = (3, 31), p = 1.0),
                # albu.MotionBlur(blur_limit = (3, 41), p = 1.0),
                # albu.ZoomBlur(max_factor = 1.3, p = 1.0),
            ],
            p=1.0
        )
    ]
    return albu_transform
    