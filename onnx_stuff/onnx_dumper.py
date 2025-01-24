import cv2
import torch
import torch.onnx
from torchvision import transforms
from unettools import MODEL_LOADERS
from pathlib import Path
from colorama import Fore, Style
from Patcher import Patcher
from typing import List, Tuple
from convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device import convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device
from load_frame import load_frame
from write_frame import write_frame


def onnx_dumper(
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
    onnx_out_file_path: Path,
    onnx_opset_version: int
):
    """
    This is to dump a Pytorch .pt checkpoint/ to an ONNX model.
    """

    for input_file_path, output_file_path in list_of_input_and_output_file_paths:
        assert isinstance(input_file_path, Path)
        assert input_file_path.is_file()
        assert isinstance(output_file_path, Path)
        
    print(f'onnx_dumper using device {device})')

    print(f'loading model {model_architecture_id} from {fn_checkpoint}..')

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
    
    # We want the sigmoiding baked in during export.
    model.include_sigmoid = True

    model.to(device).eval()

    with torch.no_grad():

        # BEGIN do something to come up with a tensor of patches to throw through the model:
        frame_bgr = load_frame(
            frame_path=input_file_path,
            inference_width=inference_width,
            inference_height=inference_height
        )

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # patch frame

        frame_tens = transform(convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device(frame_rgb))
        patches = patcher.patch(frame = frame_tens, device = device)

        # does not seem like people have been dumping under AMP
        # export the model
        print(f'exporting onnx to {onnx_out_file_path}..')
        print(f'patches {patches.shape}')
        output_names = ['segmentation']
        
        if getattr(model, 'classification_head', False):
            output_names += ['classification']
        if getattr(model, 'return_features', False):
            output_names += ['features']

        assert patches.dtype == torch.float32, f"patches.dtype is {patches.dtype} but should be torch.float16"
        assert patches.ndim == 4, f"patches.ndim is {patches.ndim} but should be 4"
        assert patches.shape[1] == 3, f"patches.shape[1] is {patches.shape[1]} but should be 3"
        # assert patches.shape[2] == 1088
        # assert patches.shape[3] == 1920
        patches = patches[:1, :, :, :]
        assert patches.shape[0] == 1, f"patches.shape[0] is {patches.shape[0]} but should be 1"

        patches_fp16_gpu = patches.half()
        print(f'{patches_fp16_gpu.device=}')
        print(f'{patches_fp16_gpu.dtype=}')

        assert patches_fp16_gpu.dtype == torch.float16, f"patches.dtype is {patches.dtype} but should be torch.float16"
        assert patches_fp16_gpu.ndim == 4, f"patches.ndim is {patches.ndim} but should be 4"
        assert patches_fp16_gpu.shape[0] == 1, f"patches.shape[0] is {patches.shape[0]} but should be 1"
        assert patches_fp16_gpu.shape[1] == 3, f"patches.shape[1] is {patches.shape[1]} but should be 3"
        
        # assert patches_fp16_gpu.shape[2] == 1088
        # assert patches_fp16_gpu.shape[3] == 1920


        torch.onnx.export(
            model.half(), # model being run, with .half() if you want fp16
            patches_fp16_gpu, # tensor input with .half() if you want fp16
            onnx_out_file_path, # where to save the model (can be a file or file-like object)
            verbose=True,
            export_params=True, # store the trained parameter weights inside the model file
            do_constant_folding=True, # whether to execute constant folding for optimization
            input_names = ["input"], # the model's input names.  Our standard is "input"
            # Can we just do "segmentation" and leave out "features"? You get x.444 in the output_names if you do that
            # for these two-tensor-emitting models.
            output_names = output_names, # Our standard output name is "segmentation". Most models don't output features, but for instance plain effs you will need to add "features" here to avoid the second output being called x.444 as seen in Netron
            opset_version=onnx_opset_version, # the ONNX version to export the model to
        )

        print(f"{Fore.YELLOW}wrote {onnx_out_file_path}{Style.RESET_ALL}")   