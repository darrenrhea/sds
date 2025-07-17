"""
It may be better to have this work on just one GPU
and then have a separate script that
determines which GPUs aren't too busy then does parallelization
by calling this as a subprocess once per available GPU
with an assignment of work
appropriate to that GPU.
"""
from get_cuda_devices import (
     get_cuda_devices
)
from make_input_output_file_path_pairs import make_input_output_file_path_pairs
import glob
import cv2
import argparse
import torch
import torch.onnx
from unettools import transforms, MODEL_LOADERS
from tqdm import tqdm
from pathlib import Path
import PIL.Image
from colorama import Fore, Style
from Patcher import Patcher
import numpy as np
from infer_all_the_patches import infer_all_the_patches
from typing import List, Tuple
from load_list_of_image_files_in_parallel import load_list_of_image_files_in_parallel
from convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device import convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device

# In python 3.12 this will be possible:
# from itertools import batched
from itertools import islice

def batched(iterable, n):
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch

WITH_AMP = torch.cuda.amp.autocast


def infer_one_frame(model_architecture_id, model, patcher, frame_bgr, transform, device):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # patch frame
    frame_tens = transform(convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device(frame_rgb))
    patches = patcher.patch(frame = frame_tens, device = device)

    # infer
    with torch.no_grad():
        with WITH_AMP():
            mask_patches = infer_all_the_patches(
                model_architecture_id=model_architecture_id,
                model=model,
                patches=patches
            )

    stitched = patcher.stitch(mask_patches)
    stitched = torch.clip(stitched * 255.0, 0, 255).type(torch.uint8)
    stitched = stitched.cpu().numpy()
    return stitched


def write_inferred_mask_to_disk(
    mask_np_u8: np.ndarray,  # may actually be grayscale and thus ndim = 2 not 3
    output_file_path: Path,  # where to write the frame to
)-> None:
    # tqdm.write(f"writing {output_file_path}")
    # TODO: maybe use cv2.imwrite instead of PIL.Image.save
    PIL.Image.fromarray(mask_np_u8).save(output_file_path, format="PNG")




def infer_a_chunk_of_frames(
    device: torch.device,
    model_architecture_id: str,
    model: torch.nn.Module,
    transform: transforms.Compose,
    patcher: Patcher,
    chunk_of_input_output_file_path_pairs: List[Tuple[Path, Path]],
):
    """
    You can run out of memory if you try
    load too many input frames into RAM,
    so for longer videos we need to process in chunks.
    """
    input_output_file_path_pairs = chunk_of_input_output_file_path_pairs

    assert isinstance(input_output_file_path_pairs, list), f"input_output_file_path_pairs should be a list, not a {type(input_output_file_path_pairs)}"
    assert len(input_output_file_path_pairs) <= 1000, "Not too many frames at once please"

    list_of_input_image_file_paths = [x[0] for x in input_output_file_path_pairs]
    list_of_output_image_file_paths = [x[1] for x in input_output_file_path_pairs]

    frames = load_list_of_image_files_in_parallel(
        list_of_image_file_paths=list_of_input_image_file_paths
    )

    frame_output_path_pairs  = list(zip(frames, list_of_output_image_file_paths))
   
    

    pbar = tqdm(
        iterable=range(len(frame_output_path_pairs)),
        total=len(frame_output_path_pairs)
    )

    for frame, output_file_path in frame_output_path_pairs:
        

        mask_np_u8 = infer_one_frame(
            device=device,
            model_architecture_id=model_architecture_id,
            model=model,
            patcher=patcher,
            frame_bgr=frame,
            transform=transform,
        )
        
        assert mask_np_u8.dtype == np.uint8
        assert mask_np_u8.ndim == 2, f"mask_np_uint8.ndim is {mask_np_u8.ndim}"
        
        write_inferred_mask_to_disk(
            mask_np_u8=mask_np_u8,
            output_file_path=output_file_path,
        )
        pbar.update(1)

    tqdm.write('finishing..')
    pbar.close()


def main():

    argp = argparse.ArgumentParser()
    argp.add_argument('model', type=str, help="model")
    argp.add_argument('checkpoint', type=str, help="checkpoint")
    argp.add_argument('input', type=str, help="description of which input frames to infer on")
    argp.add_argument('--output', type=str, default=None, help="output base name")
    argp.add_argument('--patch-width', type=int, required=True, help="patch width (224, 384, ..)")
    argp.add_argument('--patch-height', type=int, required=True, help="patch height (224, 384, ..)")
    argp.add_argument('--patch-stride-width', type=int, default=0, help="patch stride (default patch width)")
    argp.add_argument('--patch-stride-height', type=int, default=0, help="patch stride (default patch height)")
    argp.add_argument('--first-frame', type=int, default=0, help="first frame (default 0)")
    argp.add_argument('--last-frame', type=int, default=-1, help="last frame (inclusive, default -1)")
    argp.add_argument('--frame-step', type=int, default=1, help="frame hop size")
    argp.add_argument('--original-size', type=str, default=None, help="All the frames coming in, whether from video or from jpes, should have this size, width x height a comma separated list of ints)")
    argp.add_argument('--inference-size', type=str, default=None, help="scale frame to this width x height before inference, a comma separated list of ints for absolute)")
    argp.add_argument('--out-dir', type=str, default=None, help="directory to put inference results into")
    argp.add_argument('--model-id-suffix', type=str, default=None, help="usually we put a suffix on the output file name that identifies the segmentation method, like effs20231008halfstride")
    opt = argp.parse_args()

    if opt.model_id_suffix is None:
        model_id_suffix = ''
    else:
        model_id_suffix = f"_{opt.model_id_suffix}"

    assert opt.out_dir is not None or opt.video_out, "You must specify an output directory for frame output"

    if opt.out_dir is not None:
        out_dir = str(Path(opt.out_dir).resolve())
        assert Path(out_dir).is_dir(), f"{out_dir} is not an extant directory"
    else:
        out_dir = None

    if opt.inference_size is None:
        opt.inference_size = opt.original_size


    assert opt.original_size is not None, "You must specify the original size of the frames coming in"
    assert ',' in opt.original_size, "There should be a comma in the --original_size argument"
    assert ',' in opt.inference_size, "There should be a comma in the --original_size argument"


    w, h = opt.original_size.split(',')
    original_width = int(w)
    original_height = int(h)

    if opt.inference_size is None:
        opt.inference_size = opt.original_size

    w, h = opt.inference_size.split(',')
    inference_width = int(w)
    inference_height = int(h)

    print(f"{Fore.YELLOW}{original_height=}, {original_width=}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{inference_height=}, {inference_width=}{Style.RESET_ALL}")



    transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    model_name = opt.model
    model_architecture_id = opt.model
    fn_checkpoint = opt.checkpoint
    fn_input = opt.input
    # patch_size = opt.patch_size
    patch_width = opt.patch_width
    patch_height = opt.patch_height
    # patch_stride = opt.patch_stride
    patch_stride_width = opt.patch_stride_width
    patch_stride_height = opt.patch_stride_height

    # check for cuda devices
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    devices = get_cuda_devices()
    num_devices = len(devices)
    assert (
        num_devices == 1
    ), "This script is not intended to run on multiple GPUs because it is parallelized externally.  Please set CUDA_VISIBLE_DEVICES to one GPU."
    device = devices[0]

    print(f'loading model {model_name} from {fn_checkpoint}..')
    in_channels = 3
    num_class = 1  # TODO: for regression this might need to be 1
    model = MODEL_LOADERS[model_name](fn_checkpoint, multigpu = True, in_channels = in_channels, num_class = num_class)
    model.to(device).eval()

    fn_frames = glob.glob(fn_input)
    fn_frames = [fn.replace('_nonwood.png', '.jpg').replace('_nonfloor.png', '.jpg') for fn in fn_frames]

    input_output_file_path_pairs = make_input_output_file_path_pairs(
        model_id_suffix=model_id_suffix,
    )

    patcher = Patcher(
        frame_width=inference_width,
        frame_height=inference_height,
        patch_width=patch_width,
        patch_height=patch_height,
        stride_width=patch_stride_width,
        stride_height=patch_stride_height,
        pad_width=0,
        pad_height=0
    )

    for chunk_of_input_output_file_path_pairs in batched(input_output_file_path_pairs, 1000):
        infer_a_chunk_of_frames(
            device=device,
            model_architecture_id=model_architecture_id,
            model=model,
            transform=transform,
            patcher=patcher,
            chunk_of_input_output_file_path_pairs=chunk_of_input_output_file_path_pairs,
        ) 

    


if __name__ == '__main__':
    main()