from prii_hw_np_nonlinear_u16 import (
     prii_hw_np_nonlinear_u16
)
from calculate_model_outputs import (
     calculate_model_outputs
)
from typing import List
from convert_hwc_np_f32_image_to_chw_torch_f16_on_device import (
     convert_hwc_np_f32_image_to_chw_torch_f16_on_device
)
import numpy as np
import torch
import torch.onnx
from torchvision import transforms
from unettools import MODEL_LOADERS
from Patcher import Patcher
from infer_all_the_patches import infer_all_the_patches



class RamInRamOutSegmenterForMultipleOutputs(object):
        
    def __init__(
        self,
        device,  # which gpu to use
        num_outputs,
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
        
        self.num_outputs = num_outputs

        in_channels = 3
        num_class = num_outputs
        self.model = MODEL_LOADERS[model_architecture_id](fn_checkpoint, multigpu = True, in_channels = in_channels, num_class = num_class)
        
        self.model.to(device).eval()

    def infer(
        self,
        frame_rgb_hwc_np_f32: np.array,
    ) -> List[np.array]:
        
        assert frame_rgb_hwc_np_f32.dtype == np.float32
        assert frame_rgb_hwc_np_f32.ndim == 3
        assert frame_rgb_hwc_np_f32.shape[2] == 3

        with torch.no_grad():
            with torch.cuda.amp.autocast():  # TODO: shouldnt this be WITH_AMP?
                # We will patch the frame when that works for multiple outputs

                chw_torch_f16_gpu = convert_hwc_np_f32_image_to_chw_torch_f16_on_device(
                    frame_rgb_hwc_np_f32
                )
                assert chw_torch_f16_gpu.dtype == torch.float16
                assert chw_torch_f16_gpu.size() == (3, 1088, 1920)

                bchw_torch_f16_gpu = chw_torch_f16_gpu.unsqueeze(0)
                assert bchw_torch_f16_gpu.size() == (1, 3, 1088, 1920)

                dict_of_output_tensors = calculate_model_outputs(
                    model=self.model,
                    model_architecture_id=self.model_architecture_id,
                    inputs=bchw_torch_f16_gpu,
                    train=False
                )
                outputs_gpu = dict_of_output_tensors['outputs']

                assert outputs_gpu.size() == (1, self.num_outputs, 1088, 1920)
                
                between_0_and_1_gpu = torch.sigmoid(outputs_gpu)
                
                
                # Only one datapoint went through the model:
                outputs_cpu_np_f16 = between_0_and_1_gpu[0, :, :, :].cpu().detach().numpy()
                
                outputs_cpu_np_f32 = outputs_cpu_np_f16.astype(np.float32)
                
                outputs_cpu_np_u16 = np.round(outputs_cpu_np_f32 * 65535).clip(0, 65535).astype(np.uint16)
                assert outputs_cpu_np_u16.ndim == 3
                assert outputs_cpu_np_u16.shape[0] == self.num_outputs
                assert outputs_cpu_np_u16.dtype == np.uint16

                return outputs_cpu_np_u16
            
                



                # #frame_tens = self.transform(frame_rgb)
                # patches = self.patcher.patch(frame = chw_torch_f16_gpu, device = self.device)

                
                # mask_patches = infer_all_the_patches(
                #     model_architecture_id=self.model_architecture_id,
                #     model=self.model,
                #     patches=patches
                # )
            
                # stitched = self.patcher.stitch(mask_patches)
                # stitched_torch_u8 = torch.clip(stitched * 255.0, 0, 255).type(torch.uint8)
                # stitched_np_u8 = stitched_torch_u8.cpu().numpy()

                # return stitched_np_u8
                    