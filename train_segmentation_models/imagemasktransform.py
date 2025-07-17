from colorama import Fore, Style
import random
import warnings
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from albumentations.augmentations import functional as FMain
from albumentations.augmentations.blur import functional as F
from albumentations.core.transforms_interface import (
    DualTransform,
    ScaleIntType,
    to_tuple,
)


from create_psf import create_trajectory, create_psfs


class ImageMaskBlur(DualTransform):
    """Blur the input image using a random-sized kernel.

    Args:
        blur_limit (int, (int, int)): maximum kernel size for blurring the input image.
            Should be in range [3, inf). Default: (3, 7).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, blur_limit: ScaleIntType = 7, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)

        self.blur_limit = to_tuple(blur_limit, 3)

    def apply_with_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:  # skipcq: PYL-W0613
        if params is None:
            return kwargs
        params = self.update_params(params, **kwargs)
        res = {}

        key = 'image'
        target_dependencies = {k: kwargs[k] for k in self.target_dependence.get(key, [])}

        img, mask = self.apply_combined(kwargs['image'], kwargs['mask'], **dict(params, **target_dependencies))
        res['image'] = img
        res['mask'] = mask
        res["importance_mask"] = kwargs['importance_mask']
        return res
    
    def apply(self, img: np.ndarray, ksize: int = 3, **params) -> np.ndarray:
        #print('apply 1')
        return F.blur(img, ksize)

    def apply_to_mask(self, mask: np.ndarray, ksize: int = 3, **params) -> np.ndarray:
        #print('apply 2')
        return F.blur(mask, ksize)
    
    def apply_combined(self, img: np.ndarray, mask: np.ndarray, ksize: int = 3, **params) -> Sequence[np.ndarray]:
        #print('apply 3')
        return self.apply(img, ksize = ksize, **params), self.apply_to_mask(mask, ksize = ksize, **params)

    def get_params(self) -> Dict[str, Any]:
        return {"ksize": int(random.choice(list(range(self.blur_limit[0], self.blur_limit[1] + 1, 2))))}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("blur_limit",)


class ImageMaskMotionBlur(ImageMaskBlur):
    """Apply motion blur to the input image using a random-sized kernel.

    Args:
        blur_limit (int): maximum kernel size for blurring the input image.
            Should be in range [3, inf). Default: (3, 7).
        allow_shifted (bool): if set to true creates non shifted kernels only,
            otherwise creates randomly shifted kernels. Default: True.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        blur_limit: ScaleIntType = 7,
        allow_shifted: bool = True,
        always_apply: bool = False,
        p_class: float = 0.5,
        p: float = 0.5,
    ):
        super().__init__(blur_limit=blur_limit, always_apply=always_apply, p=p)
        self.allow_shifted = allow_shifted
        self.p_class = p_class

        if not allow_shifted and self.blur_limit[0] % 2 != 1 or self.blur_limit[1] % 2 != 1:
            raise ValueError(f"Blur limit must be odd when centered=True. Got: {self.blur_limit}")

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return super().get_transform_init_args_names() + ("allow_shifted", 'p_class', )

    def apply(self, img: np.ndarray, kernel: np.ndarray = None, **params) -> np.ndarray:  # type: ignore
        #print('apply')
        return FMain.convolve(img, kernel=kernel)

    def apply_to_mask(self, mask: np.ndarray, kernel: np.ndarray = None, **params) -> np.ndarray:  # type: ignore
        #print('apply mask')
        return FMain.convolve(mask, kernel=kernel)

    def apply_to_masks(self, masks: Sequence[np.ndarray], **params) -> List[np.ndarray]:
        #print('apply masks')
        return [self.apply_to_mask(mask, **params) for mask in masks]
    
    def apply_combined(self, img: np.ndarray, mask: np.ndarray, kernel: np.ndarray = None, **params) -> Sequence[np.ndarray]:
        #print('apply_combined')
        if random.random() > self.p_class:
            # blur whole image
            return self.apply(img, kernel = kernel, **params), self.apply_to_mask(mask, kernel = kernel, **params)
        else:
            # blur foreground or background only
            if mask.dtype == np.uint8:
                mask_was_uint8 = True
                blurred_mask_u8 = self.apply_to_mask(mask, kernel = kernel, **params)
                assert blurred_mask_u8.dtype == np.uint8
                blurred_mask_f32 = blurred_mask_u8.astype(float) / 255.
                weight_f32 = mask.astype(np.float32) / 255.
                mask_f32 = mask.astype(np.float32) / 255.
            else:
                mask_was_uint8 = False
                assert mask.dtype == np.float32 or mask.dtype == np.float64
                blurred_mask_f32 = self.apply_to_mask(mask, kernel = kernel, **params) 
                assert blurred_mask_f32.dtype == np.float32 or blurred_mask_f32.dtype == np.float64, f"{blurred_mask_f32.dtype=}"
                mask_f32 = mask.copy()
                weight_f32 = mask.copy()

            if img.dtype == np.uint8:
                img_was_uint8 = True
                blurred_img_u8 = self.apply(img, kernel = kernel, **params)
                assert blurred_img_u8.dtype == np.uint8
                blurred_img_f32 = blurred_img_u8.astype(float) / 255.
                img_f32 = img.astype(float) / 255.
            else:
                img_was_uint8 = False
                blurred_img_f32 = self.apply(img, kernel = kernel, **params)
                assert blurred_img_f32.dtype == np.float32 or blurred_img_f32.dtype == np.float64, f"{blurred_img_f32.dtype=}"
            
            cidx = random.random() < 0.5
            if cidx:
                #print(f"{Fore.YELLOW}blurring background i.e. LED board{Style.RESET_ALL}")
                weight_f32 = 1.0 - weight_f32
            else:
                pass
                # print(f"{Fore.YELLOW}blurring foreground i.e. all non LED{Style.RESET_ALL}")

            
            result_img_f32 = np.zeros_like(blurred_img_f32)   
            for chan in range(blurred_img_f32.shape[2]):
                result_img_f32[..., chan] = weight_f32 * blurred_img_f32[..., chan] + (1.0 - weight_f32) * img_f32[..., chan]

           
            result_mask_f32 = weight_f32 * blurred_mask_f32 + (1.0 - weight_f32) * mask_f32
          
            if mask_was_uint8:
                result_mask_u8 = np.round(np.clip(result_mask_f32 * 255, 0, 255)).astype(np.uint8)
                result_mask = result_mask_u8
            else:
                result_mask = result_mask_f32

            if img_was_uint8:
                result_img_u8 = np.round(np.clip(result_img_f32 * 255, 0, 255)).astype(np.uint8)
                result_img = result_img_u8
            else:
                result_img = result_img_f32

            assert result_img.dtype == img.dtype
            assert result_mask.dtype == mask.dtype
            return result_img, result_mask


    def get_params(self) -> Dict[str, Any]:

        if random.random() < 0.25:
            #print('curve')
            traj_curve = create_trajectory(do_show=False)
            psfs = create_psfs(traj_curve, do_show=False, T=[0.5], do_center=True)
            kernel = psfs[0]            
        else:
            ksize = random.choice(list(range(self.blur_limit[0], self.blur_limit[1] + 1, 2)))
            if ksize <= 2:
                raise ValueError("ksize must be > 2. Got: {}".format(ksize))
            kernel = np.zeros((ksize, ksize), dtype=np.uint8)
            x1, x2 = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
            if x1 == x2:
                y1, y2 = random.sample(range(ksize), 2)
            else:
                y1, y2 = random.randint(0, ksize - 1), random.randint(0, ksize - 1)

            def make_odd_val(v1, v2):
                len_v = abs(v1 - v2) + 1
                if len_v % 2 != 1:
                    if v2 > v1:
                        v2 -= 1
                    else:
                        v1 -= 1
                return v1, v2

            if not self.allow_shifted:
                x1, x2 = make_odd_val(x1, x2)
                y1, y2 = make_odd_val(y1, y2)

                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2

                center = ksize / 2 - 0.5
                dx = xc - center
                dy = yc - center
                x1, x2 = [int(i - dx) for i in [x1, x2]]
                y1, y2 = [int(i - dy) for i in [y1, y2]]

            cv2.line(kernel, (x1, y1), (x2, y2), 1, thickness=1)

        # Normalize kernel
        return {"kernel": kernel.astype(np.float32) / np.sum(kernel)}

