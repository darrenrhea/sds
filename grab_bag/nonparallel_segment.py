import cv2
import torch
import torch.onnx
from torchvision import transforms
from unettools import MODEL_LOADERS
from pathlib import Path
from Patcher import Patcher
from infer_all_the_patches import infer_all_the_patches
from typing import List, Tuple
from convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device import convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device
from load_frame import load_frame
from write_frame import write_frame


def nonparallel_segment(
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
    list_of_input_and_output_file_paths: List[Tuple[Path, Path]],
):
    """
    This allows you to segment file in file out.
    This is segment_thread without all the multithreading issues.
    """
    
    for input_file_path, output_file_path in list_of_input_and_output_file_paths:
        assert isinstance(input_file_path, Path)
        assert input_file_path.is_file()
        assert isinstance(output_file_path, Path)

    transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    patcher = Patcher(
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
    model = MODEL_LOADERS[model_architecture_id](fn_checkpoint, multigpu = True, in_channels = in_channels, num_class = num_class)
    
    model.to(device).eval()

    with torch.no_grad():
        with torch.cuda.amp.autocast():  # TODO: shouldnt this be WITH_AMP?
            for input_file_path, output_file_path in list_of_input_and_output_file_paths:
                frame_bgr = load_frame(
                    frame_path=input_file_path,
                    inference_width=inference_width,
                    inference_height=inference_height
                )

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # patch frame

                frame_tens = transform(convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device(frame_rgb))
                #frame_tens = transform(frame_rgb)
                patches = patcher.patch(frame = frame_tens, device = device)

                
                mask_patches = infer_all_the_patches(
                    model_architecture_id=model_architecture_id,
                    model=model,
                    patches=patches
                )
               
                stitched = patcher.stitch(mask_patches)
                stitched_torch_u8 = torch.clip(stitched * 255.0, 0, 255).type(torch.uint8)
                stitched_np_u8 = stitched_torch_u8.cpu().numpy()

                write_frame(
                    frame=stitched_np_u8,
                    output_file_path=output_file_path,
                    original_height=original_height,
                    original_width=original_width,
                )
                