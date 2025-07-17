import numpy as np
from pathlib import Path

import torch

from minimal_u3net import build_unet3plus

class SegmentationInferrerFor1920x1088u3fasternetsModels:
    def __init__(
        self,
        pt_weights_file_path: Path,
        device: torch.device,
    ):
        self.input_size = (1920, 1088)
        self.model_type = "u3fasternets"
        self.device = device
        
        self.model = build_unet3plus(
            num_classes=1,
            encoder="fasternets",
            pretrained=False,
        )

        checkpoint = torch.load(
            f=pt_weights_file_path,
            weights_only=False
        )#, map_location=torch.device('cpu'))
        print('checkpoint keys', list(checkpoint.keys()))

   
        model_state_dict = checkpoint['model']

        self.model.load_state_dict(model_state_dict)
       
        self.model.to(device)
        self.model.half()
        self.model.eval()

    def infer(self, rgb_hwc_np_nonlinear_f32):
        """
        Perform inference on a numpy height x width x channel rgb image whose values
        range from 0.0 to 1.0 in the naive / nonlinear way.
        """
        assert rgb_hwc_np_nonlinear_f32.dtype == np.float32
        assert rgb_hwc_np_nonlinear_f32.shape == (1088, 1920, 3), f"Expected shape (1088, 1920, 3), got {rgb_hwc_np_nonlinear_f32.shape}"
        trivial_batch_gpu_f16 = torch.from_numpy(
            rgb_hwc_np_nonlinear_f32
        ).cuda().half().permute(2, 0, 1).unsqueeze(0)

       

        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float16,
            device=self.device,
        ).view(1, 3, 1, 1)

        std  = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float16,
            device=self.device
        ).view(1, 3, 1, 1)

        trivial_batch_chw_alexnet_gpu_f16 = (trivial_batch_gpu_f16 - mean) / std

        trivial_batch_of_masks_gpu_f16 = self.model(trivial_batch_chw_alexnet_gpu_f16)

        trivial_batch_of_masks_gpu_in_01_f16 = torch.sigmoid(trivial_batch_of_masks_gpu_f16)

        mask_gpu_f16 = trivial_batch_of_masks_gpu_in_01_f16.squeeze(0).squeeze(0)
        mask_hw_np_f16 = mask_gpu_f16.detach().cpu().numpy()
        print(f"{mask_hw_np_f16.shape=}")
        mask_hw_np_f32 = mask_hw_np_f16.astype(np.float32)
        return mask_hw_np_f32