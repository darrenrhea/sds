"""
we abandoned infer3.py this to write this:
nonparallel_infer3.py (and also parallel_infer3.py)
because we don't like if elses around parallel or not.
"""
import torch
from pathlib import Path
from colorama import Fore, Style
from get_cuda_devices import get_cuda_devices
from parse_cli_args_for_inferers import (
     parse_cli_args_for_inferers
)
from get_list_of_input_and_output_file_paths_from_json_file_path import (
     get_list_of_input_and_output_file_paths_from_json_file_path
)
from get_list_of_input_and_output_file_paths_from_old_style import (
     get_list_of_input_and_output_file_paths_from_old_style
)
from nonparallel_segment import (
     nonparallel_segment
)

print(f"{Fore.MAGENTA}HELLO from nonparallel_infer3.py{Style.RESET_ALL}")
WITH_AMP = torch.cuda.amp.autocast


opt = parse_cli_args_for_inferers()

pad_height = opt.pad_height
pad_width = opt.pad_width

if opt.model_id_suffix is None:
    model_id_suffix = ""
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
assert "," in opt.original_size, "There should be a comma in the --original_size argument"
assert "," in opt.inference_size, "There should be a comma in the --original_size argument"


w, h = opt.original_size.split(",")
original_width = int(w)
original_height = int(h)

if opt.inference_size is None:
    opt.inference_size = opt.original_size

w, h = opt.inference_size.split(",")
inference_width = int(w)
inference_height = int(h)

print(f"{Fore.YELLOW}{original_height=}, {original_width=}{Style.RESET_ALL}")
print(f"{Fore.YELLOW}{inference_height=}, {inference_width=}{Style.RESET_ALL}")

model_name = opt.model
model_architecture_id = opt.model
fn_checkpoint = opt.checkpoint
fn_input = opt.input
fn_output = opt.output
patch_width = opt.patch_width
patch_height = opt.patch_height
patch_stride_width = opt.patch_stride_width
patch_stride_height = opt.patch_stride_height

MODEL_NAME = fn_checkpoint.split("/")[-1].split(".")[0]

# check for cuda devices
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

devices = get_cuda_devices()
num_devices = len(devices)
jobs = opt.jobs

if fn_input.endswith(".json") or fn_input.endswith(".json5"):
    # this is a json file containing a list of input and output file paths
    # this is the new style
    json_path = Path(fn_input).resolve()
    print(f"{Fore.MAGENTA}reading which frames to infer from {fn_input}{Style.RESET_ALL}")
    assert json_path.is_file(), f"{json_path} is not a file"

    list_of_input_and_output_file_paths = get_list_of_input_and_output_file_paths_from_json_file_path(
        json_path=fn_input,
        out_dir = Path(out_dir).resolve(),
        model_id_suffix = model_id_suffix
    )
else:
    list_of_input_and_output_file_paths = get_list_of_input_and_output_file_paths_from_old_style(
        fn_input = fn_input,
        limit = opt.limit,
        out_dir = Path(out_dir).resolve(),
        model_id_suffix = model_id_suffix
    )

number_of_frames = len(list_of_input_and_output_file_paths)

# not video, but particular blown-out frames:
fn_output = f"{out_dir}/{{:s}}.png"
print(f"{Fore.MAGENTA}writing output frames to {fn_output}..{Style.RESET_ALL}")
print(f"saving output frames to {fn_output}..")

devices = get_cuda_devices()
device = devices[0]

nonparallel_segment(
    list_of_input_and_output_file_paths=list_of_input_and_output_file_paths,
    device=device,
    model_name=model_name,
    fn_checkpoint=fn_checkpoint,
    model_architecture_id=model_architecture_id,
    original_height=original_height,
    original_width=original_width,
    inference_height=inference_height,
    inference_width=inference_width,
    pad_height=pad_height,
    pad_width=pad_width,
    patch_height=patch_height,
    patch_width=patch_width,
    patch_stride_height=patch_stride_height,
    patch_stride_width=patch_stride_width,
)


