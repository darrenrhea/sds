"""
we abandoned infer3.py this to write this:
parallel_infer3.py (and also nonparallel_infer3.py)
because we don't like if elses.
"""
from ColorCorrector import (
     ColorCorrector
)
import pprint as pp
import torch
from tqdm import tqdm
import time
import threading
from queue import Queue
import signal
from pathlib import Path
from colorama import Fore, Style

from get_cuda_devices import get_cuda_devices
from segment_thread import segment_thread
from parse_cli_args_for_inferers import parse_cli_args_for_inferers
from check_todo_task import check_todo_task
from get_list_of_input_and_output_file_paths_from_json_file_path import get_list_of_input_and_output_file_paths_from_json_file_path
from get_list_of_input_and_output_file_paths_from_old_style import get_list_of_input_and_output_file_paths_from_old_style

WITH_AMP = torch.cuda.amp.autocast


class ServiceExit(Exception):
    """
    This is a custom exception
    which the signal handler can throw / raise
    to trigger the clean exit
    of all running threads and the main program.
    https://www.g-loaded.eu/2016/11/24/how-to-terminate-running-python-threads-using-signals/
    """
    pass
 
 
def service_shutdown(signum, frame):
    print("Caught signal %d" % signum)
    raise ServiceExit
 

signal.signal(signal.SIGTERM, service_shutdown) # kill
signal.signal(signal.SIGINT, service_shutdown)  # ctrl-c


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
# patch_size = opt.patch_size
patch_width = opt.patch_width
patch_height = opt.patch_height
# patch_stride = opt.patch_stride
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
        json_file_path=json_path,
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

print(f"{Fore.YELLOW}number of frames to infer: {number_of_frames}{Style.RESET_ALL}")

# loop over frames
first_frame_idx = opt.first_frame
last_frame_idx = min(opt.last_frame if opt.last_frame > -1 else number_of_frames, number_of_frames)
frame_step = opt.frame_step
num_segmenter_threads = num_devices * jobs

print(f"{Fore.YELLOW}WARNING: Parallel codepath is happening{Style.RESET_ALL}")
print(f"starting {num_segmenter_threads} segmenter processes...")

# make the todo queue, fill it up:
seg_queue_todo = Queue()
pending = 0
for idx, (input_file_path, output_file_path) in enumerate(list_of_input_and_output_file_paths):
    todo_task = (input_file_path, output_file_path)
    check_todo_task(todo_task)
    seg_queue_todo.put(todo_task)
    pending += 1

total_num_frames_to_infer = len(list_of_input_and_output_file_paths)


pbar = tqdm(
    range(total_num_frames_to_infer),
    total=total_num_frames_to_infer
)


seg_queue_done = Queue()

segmenters = []
termination_events = []

color_corrector = ColorCorrector(
    gamma=opt.gamma_correction
)

for id in range(num_segmenter_threads):
    termination_event = threading.Event()
    termination_events.append(termination_event)
    device = devices[id % num_devices]
    print(f"segmenter process {id} will use device {device}")
    segmenters.append(
        threading.Thread(
            target=segment_thread,
            kwargs=dict(
                color_corrector=color_corrector,
                id=id,
                seg_queue_todo=seg_queue_todo,
                seg_queue_done=seg_queue_done,
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
                termination_event=termination_event,
                total_num_frames_to_infer=total_num_frames_to_infer,
            )
        )
    )


for segmenter in segmenters:
    segmenter.start()


try:
    while True:   # the main thread has to be alive to receive the signal
        time.sleep(0.5)
        num_completed = seg_queue_done.qsize()
        if pbar.n != num_completed:
            pbar.n = num_completed
            pbar.refresh()
        if all([termination_event.is_set() for termination_event in termination_events]) or num_completed >= total_num_frames_to_infer:
            break
except ServiceExit:
    print("Caught ServiceExit, so setting all termination_events to True to request graceful termination of all worker threads.")
    for termination_event in termination_events:
        termination_event.set()
    
    


for segmenter in segmenters:
    segmenter.join()


tqdm.write("finishing..")
pbar.close()

