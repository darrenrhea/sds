import albumentations as albu
import cv2  # cv2.BORDER_CONSTANT below


def get_coaches_augmentation():
    albu_transform = [
        albu.HorizontalFlip(p = 0.5),
    ]
    return albu_transform

    albu_transform = [
        albu.HorizontalFlip(p = 0.5),
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
        # albu.augmentations.geometric.transforms.ElasticTransform(
        #     alpha=1,
        #     sigma=100,
        #     alpha_affine=50,
        #     interpolation=1,
        #     border_mode=cv2.BORDER_CONSTANT,
        #     value=[255, 255, 0],  # yellow, because it want to know if alpha is 0 everywhere
        #     mask_value=0,  # set alpha to 0 of the map
        #     always_apply=False,
        #     approximate=False,
        #     same_dxdy=False, p=1.0
        # ),
        albu.augmentations.geometric.transforms.Perspective(
            scale=(0.05, 0.1),
            keep_size=True,
            pad_mode=0,
            pad_val=0,
            mask_pad_val=0,
            fit_output=False,
            interpolation=1,
            always_apply=True,
            p=1.0
        ),

        # albu.augmentations.geometric.transforms.PiecewiseAffine(
        #     scale=(0.03, 0.05),
        #     nb_rows=4,
        #     nb_cols=4,
        #     interpolation=1,
        #     mask_interpolation=0,
        #     cval=0,
        #     cval_mask=0,
        #     mode='constant',
        #     absolute_scale=False,
        #     always_apply=True,
        #     keypoints_threshold=0.01,
        #     p=1.0
        # ),

        # albu.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),

        # albu.augmentations.transforms.ColorJitter(
        #     brightness=0.05,
        #     contrast=0.01,
        #     saturation=0.01,
        #     hue=0.01,
        #     always_apply=True,
        #     p=1.0
        # ),
        
        # albu.RandomBrightnessContrast(
        #     brightness_limit=(-0.05, 0.05),
        #     contrast_limit=(-0.01, 0.01),  # this is dangerous for players
        #     p=1.0
        # ),
        
        albu.augmentations.geometric.resize.RandomScale(
            scale_limit=(-0.01, 0.01),
            interpolation=1,
            always_apply=True,
            p=1.0
        ),
        # albu.ISONoise(color_shift=(0.01, 0.03), intensity=(0.0, 0.2), p=1.0),
    ]
    return albu_transform