import numpy as np
import torch
import torch.onnx
from torchvision import transforms
from unettools import MODEL_LOADERS
from Patcher import Patcher
from infer_all_the_patches import infer_all_the_patches
from convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device import (
     convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device
)


class RamInRamOutSegmenter(object):
        
    def __init__(
        self,
        device,  # which gpu to use
        fn_checkpoint,
        model_architecture_id: str,
        inference_height: int,
        inference_width: int,
        pad_height: int,
        pad_width: int,
        patch_height: int,
        patch_width: int,
        patch_stride_height: int,
        patch_stride_width: int,
    ):
        """
        Once instantiated
        which involves moving the weights into a particular gpu,
        RamInRamOutSegmenter can be used to segment a frame into a mask.
        allows you to segment a rgb_hwc_np_u8 frame into a mask_hw_np_u8.
        """
        self.device = device
        self.model_architecture_id = model_architecture_id
        self.transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
        self.model = MODEL_LOADERS[model_architecture_id](fn_checkpoint, multigpu = True, in_channels = in_channels, num_class = num_class)
        
        self.model.to(device).eval()

    def infer(
        self,
        frame_rgb: np.array,
    ) -> np.array:
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # TODO: shouldnt this be WITH_AMP?
                # patch frame

                frame_tens = self.transform(
                        convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device(
                            frame_rgb
                    )
                )
                #frame_tens = transform(frame_rgb)
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
                    