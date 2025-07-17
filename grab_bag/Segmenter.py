from get_normalization_and_chw_transform import (
     get_normalization_and_chw_transform
)
import torch
import torch.onnx
from torchvision import transforms
from unettools import MODEL_LOADERS
from pathlib import Path
from Patcher import Patcher
from infer_all_the_patches import infer_all_the_patches
from typing import List, Tuple
from convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device import convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device


class Segmenter(object):

    def __init__(
        self,
        device,  # which gpu to use
        fn_checkpoint,
        model_architecture_id: str,
        inference_height: int,
        inference_width: int,
        original_height: int,
        original_width: int,
        pad_height: int,
        pad_width: int,
        patch_height: int,
        patch_width: int,
        patch_stride_height: int,
        patch_stride_width: int,
    ):
        """
        This allows you to segment file in file out.
        This is segment_thread without all the multithreading issues.
        """
        print(f"{patch_width=}")
        self.device = device
        self.model_architecture_id = model_architecture_id
        self.original_height = original_height
        self.original_width = original_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_stride_height = patch_stride_height
        self.patch_stride_width = patch_stride_width
        self.pad_height = pad_height
        self.pad_width = pad_width
        
        self.transform = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )

        self.patcher = Patcher(
            frame_width=inference_width,
            frame_height=inference_height,
            patch_width=patch_width,
            patch_height=patch_height,
            stride_width=patch_stride_width,
            stride_height=patch_stride_height,
            pad_width=0,
            pad_height=pad_height
        )

        in_channels = 3
        num_class = 1  # TODO: for regression this might need to be 1
        self.model = MODEL_LOADERS[model_architecture_id](
            fn_checkpoint,
            multigpu = True,
            in_channels = in_channels,
            num_class = num_class
        )
        
        # WOW.  This is the footgun that I was looking for.
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        self.model.to(self.device).eval()

    def infer_rgb_hwc_np_u8_to_hw_np_u8(
        self,
        rgb_hwc_np_u8
    ):
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # TODO: shouldnt this be WITH_AMP?
                between_0_and_1 = convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device(rgb_hwc_np_u8)
                print(f"{torch.min(between_0_and_1)=}")
                print(f"{torch.max(between_0_and_1)=}")
                transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                # alexnet /imagenet normalization:
                frame_tens = transform(between_0_and_1)

                patches = self.patcher.patch(frame = frame_tens, device = self.device)

                
                mask_patches = infer_all_the_patches(
                    model_architecture_id=self.model_architecture_id,
                    model=self.model,
                    patches=patches
                )
            
                stitched = self.patcher.stitch(mask_patches)
               
                stitched_torch_u8 = torch.clip(stitched * 255.0, 0, 255).type(torch.uint8)
                stitched_np_u8 = stitched_torch_u8.cpu().numpy()

                return stitched_np_u8
            
    
    def infer_rgb_hwc_np_u8_to_hw_np_f32(
        self,
        rgb_hwc_np_u8
    ):
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # TODO: shouldnt this be WITH_AMP?
                frame_tens = self.transform(convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device(rgb_hwc_np_u8))
                patches = self.patcher.patch(frame = frame_tens, device = self.device)

                
                mask_patches = infer_all_the_patches(
                    model_architecture_id=self.model_architecture_id,
                    model=self.model,
                    patches=patches
                )
            
                stitched_torch_f16 = self.patcher.stitch(mask_patches)
                
                clipped_torch_f16 = torch.clip(
                    stitched_torch_f16,
                    0,
                    1
                )

                mask_np_f32 = clipped_torch_f16.cpu().numpy()
                assert mask_np_f32.shape[0] == self.original_height
                assert mask_np_f32.shape[1] == self.original_width
                return mask_np_f32
                    