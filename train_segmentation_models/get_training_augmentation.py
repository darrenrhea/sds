import sys
from offscreen_is_labelled_as_jetblack_wrapper import (
     offscreen_is_labelled_as_jetblack_wrapper
)
import albumentations as A
import cv2
from imagemasktransform import ImageMaskMotionBlur
from augmentation_strategies_info import valid_augmentation_strategy_ids

def get_training_augmentation(
    augmentation_id: str,
    frame_width: int,
    frame_height: int,
):
    """
    What data augmentation to use during training the model.
    """
    assert (
        augmentation_id in valid_augmentation_strategy_ids
    ), f"ERROR: {augmentation_id=} is not valid.  Possible values are {valid_augmentation_strategy_ids}"
    label_offscreen_via_jet_black = augmentation_id in ["forflat"]

    if augmentation_id == "identity":
        train_transform = []
    elif augmentation_id == "basic":
        train_transform = [
            A.HorizontalFlip(p = 0.5),
        ]
    elif augmentation_id == "small":
        train_transform = [
            A.HorizontalFlip(p = 0.2),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.8), contrast_limit=(-0.5, 0.7), p = 1.0),
            ], p=0.1),
        ]
    elif augmentation_id == "medium":
        train_transform = [
            A.HorizontalFlip(p = 0.2),
            A.OneOf([
                # TODO: blur mask
                A.MotionBlur(blur_limit = (3, 31), p = 1.0),
                A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.8), contrast_limit=(-0.5, 0.7), p = 1.0),
            ], p=0.1),
            #A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.1),
            #A.RandomBrightnessContrast(p = 0.1),
            #A.RandomScale(scale_limit=(-0.2, 0.2), interpolation=cv2.INTER_CUBIC, p = 0.3),
            #A.PadIfNeeded(min_height=frame_height, min_width=frame_width),
            #A.CenterCrop(height=frame_height, width=frame_width),
            #A.Rotate(limit=10, p=0.2),
            #A.RandomBrightness(limit = 0.5, p=0.1),
            #A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.8), contrast_limit=(-0.5, 0.7), p = 0.5),
            #A.RandomBrightness(limit=(-0.2, 0.8), p = 0.1),
            # A.OneOf([
            #     #A.Blur(blur_limit = (3, 11), p = 1.0),
            #     A.MotionBlur(blur_limit = (3, 31), p = 1.0),
            #     A.ZoomBlur(max_factor = 1.3, p = 1.0),
            # ], p=0.05),
            #A.RGBShift(r_shift_limit = 30, g_shift_limit = 30, b_shift_limit = 30,  p=0.1),
            #A.ImageCompression(quality_lower = 20, quality_upper = 50, p=0.1),
        ]
    elif augmentation_id == "felix1":
        train_transform = [
            A.HorizontalFlip(p = 0.2),
            A.OneOf([
                # TODO: blur mask
                A.MotionBlur(blur_limit = (3, 31), p = 1.0),
                A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.8), contrast_limit=(-0.2, 0.5), p = 1.0),
            ], p=0.1),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.5), p=0.3),
            A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=0.3)
        ]
    elif augmentation_id == "felix2":
        train_transform = [
            A.ShiftScaleRotate(
                shift_limit=0.30,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.9
            ),
            A.HorizontalFlip(p = 0.2),
            A.OneOf([
                # TODO: blur mask
                A.MotionBlur(blur_limit = (3, 31), p = 1.0),
                A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.8), contrast_limit=(-0.2, 0.5), p = 1.0),
            ], p=0.1),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.5), p=0.3),
            A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=0.3)
        ]
    elif augmentation_id == "felix3":
        train_transform = [
            A.HorizontalFlip(p = 0.2),
            A.Compose([
                A.RandomScale(scale_limit=(0, 0.2), interpolation=cv2.INTER_LINEAR, p = 1.0, always_apply=True),
                A.PadIfNeeded(min_height=frame_height, min_width=frame_width, p = 1.0, always_apply=True),
                A.RandomCrop(height=frame_height, width=frame_width, p = 1.0, always_apply=True),
            ], p = 0.2),
            A.OneOf([
                ImageMaskMotionBlur(blur_limit = (9, 31), p_class=0.5, allow_shifted = False, p = 1.0),
                A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.8), contrast_limit=(-0.2, 0.5), p = 1.0),
            ], p=0.1),
            A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=0.5),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.2), p=0.1),
        ]
        
    elif augmentation_id == "someblur":
        train_transform = [
            # A.HorizontalFlip(p = 0.2),
            A.ShiftScaleRotate(
                shift_limit=0.30,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.1
            ),
            A.Compose([
                A.RandomScale(scale_limit=(0, 0.2), interpolation=cv2.INTER_LINEAR, p = 1.0),
                A.PadIfNeeded(min_height=frame_height, min_width=frame_width, p = 1.0),
                A.RandomCrop(height=frame_height, width=frame_width, p = 1.0),
            ], p = 0.2),
            ImageMaskMotionBlur(blur_limit = (9, 31), p_class=0.2, allow_shifted = False, p = 1.0),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p = 0.3),
            A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=0.1),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.2), p=0.1),
        ]
       
    elif augmentation_id == "aug_2025-07-12":
        train_transform = [
            A.RandomBrightnessContrast(
                brightness_limit=(-0.02, 0.10),
                contrast_limit=(-0.00001, 0.00001),  # turn it off
                p=0.95,
            ),
            A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=0.5),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.2), p=0.1),
        ]

    elif augmentation_id == "wednesday":
        train_transform = [
            # A.HorizontalFlip(p = 0.2),
            # A.Compose([
            #     A.RandomScale(scale_limit=(0, 0.2), interpolation=cv2.INTER_LINEAR, p = 1.0),
            #     A.PadIfNeeded(min_height=frame_height, min_width=frame_width, p = 1.0),
            #     A.RandomCrop(height=frame_height, width=frame_width, p = 1.0),
            # ], p = 0.2),
            # ImageMaskMotionBlur(blur_limit = (9, 31), p_class=0.5, allow_shifted = False, p = 1.0),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p = 0.3),
            A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=0.1),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.2), p=0.1),
        ]
                
    elif augmentation_id == "forflat":
        train_transform = [
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2), contrast_limit=(-0.2, 0.2), p = 0.3),
            A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=0.1),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.2), p=0.1),
        ]
        
    elif augmentation_id == "gamma":
        train_transform = [
            # A.HorizontalFlip(p = 0.2),
            A.Compose([
                A.RandomScale(scale_limit=(0, 0.2), interpolation=cv2.INTER_LINEAR, p = 1.0),
                A.PadIfNeeded(min_height=frame_height, min_width=frame_width, p = 1.0),
                A.RandomCrop(height=frame_height, width=frame_width, p = 1.0),
            ], p = 0.2),
            # ImageMaskMotionBlur(blur_limit = (9, 31), p_class=0.5, allow_shifted = False, p = 1.0),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p = 0.3),
            A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=0.1),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.2), p=0.1),
            A.RandomGamma(gamma_limit=(60, 90), p=0.4),
            A.ColorJitter(hue=(-0.01, 0.01), p=0.4)
        ]
         
    elif augmentation_id == "evenless":
        train_transform = [
            A.ShiftScaleRotate(
                shift_limit=0.10,
                scale_limit=0.2,
                rotate_limit=5,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.9
            ),
            A.HorizontalFlip(p = 0.5),
        ]
       
    
    elif augmentation_id == "justflip":
        train_transform = [
            A.HorizontalFlip(p = 0.5),
        ]
    
    elif augmentation_id == "sunday":
        train_transform = [
            A.ShiftScaleRotate(
                shift_limit=0.30,
                scale_limit=0.1,
                rotate_limit=5,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.9
            ),
            A.HorizontalFlip(p = 0.5),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.0, 0.2), p=0.3),
            A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=0.3),
        ]
    elif augmentation_id == "imagemaskmotionblur":

        train_transform = [
            A.ShiftScaleRotate(
                shift_limit=0.30,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.9
            ),
            A.HorizontalFlip(p = 0.2),
            A.OneOf(
                [
                    ImageMaskMotionBlur(
                        blur_limit = (9, 31),  # used to say 61
                        allow_shifted = False,
                        p_class=0.5,
                        p = 1.0
                    ),
                    # A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.8), contrast_limit=(-0.2, 0.5), p = 1.0),
                ],
                p=1.0
            ),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.5), p=0.3),
            A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=0.3),
        ]
    else:
        raise Exception(f"ERROR: unknown augmentation_id: {augmentation_id}")


    augmentation = A.Compose(
        train_transform,
        additional_targets={'importance_mask': 'mask'},  # https://albumentations.ai/docs/examples/example_multi_target/
        p=1.0
    )

    if label_offscreen_via_jet_black:
        ans = offscreen_is_labelled_as_jetblack_wrapper(augmentation)
    else:
        ans = augmentation
    
    return ans