"""
Seems like this, i.e. infer2.py is still used by the tests in
run_tests.py
"""
from get_cuda_devices import (
     get_cuda_devices
)
import sys
import cv2
import argparse
import torch
import torch.onnx
from unettools import MODEL_LOADERS
from tqdm import tqdm
import time
from threading import Thread
from queue import Queue
import signal
import glob
from pathlib import Path
from joblib import Parallel, delayed
import PIL.Image
from colorama import Fore, Style
from Patcher import Patcher
import numpy as np
from infer_all_the_patches import infer_all_the_patches
from convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device import convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device
from torchvision import transforms
WITH_AMP = torch.cuda.amp.autocast

# https://pytorch.org/docs/stable/onnx.html
# default: 14, latest: 16
ONNX_OPSET_VERSION = 14

class FrameReader(object):

    def __init__(
        self,
        fn_input,
        inference_width: int,
        inference_height: int,
        limit = 0
    ):
        self.inference_height = inference_height
        self.inference_width = inference_width

        self.fn_frames = []
        if '**' in fn_input:
            tokens = fn_input.split('/')
            path = '/'.join(tokens[:-1])
            self.fn_frames = [str(x) for x in Path(path).rglob(tokens[-1].replace('**', '*'))]
        elif '*' in fn_input:
            self.fn_frames = glob.glob(fn_input)
        else:
            self.fn_frames = [fn_input]

        self.fn_frames = [fn.replace('_nonwood.png', '.jpg').replace('_nonfloor.png', '.jpg') for fn in self.fn_frames]

        if limit > 0:
            np.random.seed(1337)
            perm = np.random.permutation(len(self.fn_frames))[:limit]
            self.fn_frames = [self.fn_frames[j] for j in perm]

        def load_frame(fn_frames, idx):
            try:
                fn = fn_frames[idx]
                print(f'reading {fn}')
                full_sized_frame = cv2.imread(fn)  # we think this removes any alpha channel
                if full_sized_frame is None:
                    raise Exception('error loading!')
                
                frame = cv2.resize(
                    full_sized_frame,
                    (self.inference_width, self.inference_height)
                )
                
                if frame.shape[2] == 4:
                    raise Exception('alpha channel detected?! This should not happen without flag cv2.IMREAD_UNCHANGED')
                    # return idx, frame[:, :, :3]
                else:
                    return idx, frame
            except Exception as e:
                print(f'ERROR processing {fn_frames[idx]}:\n{e}')
                return idx, None

        results = Parallel(n_jobs=32)(delayed(load_frame)(self.fn_frames, i)
                                        for i in tqdm(range(len(self.fn_frames)), total = len(self.fn_frames)))

        results = sorted(results, key = lambda x: x[0])

        good_idx = [i for i in range(len(results)) if not results[i][1] is None]
        results = [results[i] for i in good_idx]
        self.fn_frames = [self.fn_frames[i] for i in good_idx]

        self.frames = [res[1] for res in results]
        # self.frame_height = self.frames[0].shape[0]
        # self.frame_width = self.frames[0].shape[1]
        self.number_of_frames = len(self.frames)
        self.frame_rate = 60


    def __getitem__(self, key):
        return self.frames[key]

    def __len__(self):
        return self.number_of_frames


onnx_written = False
def process_frame(frame_idx, frame_bgr, transform, device = None):
    global MODEL_NAMEl, ONNX_OPSET_VERSION, opt, onnx_written

    if device is None:
        device = devices[0]

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

   
    # patch frame
    frame_tens = transform(convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device(frame_rgb, device))
    patches = patcher.patch(frame = frame_tens, device = device)
    # print(f"{Fore.YELLOW}{patches.shape=}{Style.RESET_ALL}")
    

    # infer
    with torch.no_grad():

        if opt.onnx and not onnx_written:
            # export the model
            print(f'exporting onnx to {opt.onnx}..')
            print(f'patches {patches.shape}')
            onnx_written = True
            output_names = ['segmentation']
            if hasattr(model, 'classification_head') and model.classification_head:
                output_names += ['classification']
            torch.onnx.export(model, # model being run
                            patches, # model input (or a tuple for multiple inputs)
                            opt.onnx, # where to save the model (can be a file or file-like object)
                            verbose=True,
                            export_params=True, # store the trained parameter weights inside the model file
                            opset_version=ONNX_OPSET_VERSION, # the ONNX version to export the model to
                            do_constant_folding=True, # whether to execute constant folding for optimization
                            input_names = ['input'], # the model's input names
                            output_names = output_names # the model's output names
                            #,
                            #dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                            #                'output' : {0 : 'batch_size'}}
                            )

        with WITH_AMP():
            print(f"{Fore.YELLOW}{type(model)=}{Style.RESET_ALL}")
            t0 = time.time()
            print(f"{Fore.YELLOW}input patches shape is {patches.shape}{Style.RESET_ALL}")
            mask_patches = infer_all_the_patches(
                model_architecture_id=model_architecture_id,
                model=model,
                patches=patches
            )

    stitched = patcher.stitch(mask_patches)
    stitched = torch.clip(stitched * 255.0, 0, 255).type(torch.uint8)
    stitched = stitched.cpu().numpy()
    t1 = time.time()
    dt = (t1 - t0) * 1000.0

    return frame_idx, stitched, dt


def write_frame(
    frame_idx,
    frame_bgr,  # may actually be grayscale and thus ndim = 2
    original_height,
    original_width
):
    fn_frame = vin.fn_frames[frame_idx].split('/')[-1].split('.')[0] + model_id_suffix
    fn_out = fn_output.format(fn_frame)

    if '*' in fn_out:
        fn_out = fn_out.replace('*', '+')

    tqdm.write(f'writing {fn_out}')
    full_size = cv2.resize(frame_bgr, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
    PIL.Image.fromarray(full_size).save(fn_out, format="PNG")


def segment_thread(id, seg_queue_todo, seg_queue_done, device, model_name, fn_checkpoint, CHANNEL_MASK, transform):
    print(f'segmenter process {id} starting (device {device})')

    print(f'segmenter{id}: loading model {model_name} from {fn_checkpoint}..')
    in_channels = 3
    num_class = 1  # TODO: for regression this might need to be 1
    model = MODEL_LOADERS[model_name](fn_checkpoint, multigpu = True, in_channels = in_channels, num_class = num_class)
    
    model.to(device).eval()

    #transform = get_preprocessing(preproc)

    with torch.no_grad():
        with torch.cuda.amp.autocast():

            print(f'segmenter thread {id} ready')

            while True:
                item = seg_queue_todo.get(block=True)

                if item is None:
                    print(f'segmenter process {id} exiting.')
                    del model
                    return 0

                # got work to do
                frame_idx, frame_bgr = item

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # patch frame

                frame_tens = transform(convert_hwc_u8_np_image_to_chw_torch_f16_in_0_1_range_on_device(frame_rgb, device))
                #frame_tens = transform(frame_rgb)
                patches = patcher.patch(frame = frame_tens, device = device)

                # infer
                t0 = time.time()
                mask_patches = infer_all_the_patches(
                    model_architecture_id=model_architecture_id,
                    model=model,
                    patches=patches
                )
                t1 = time.time()
                dt = (t1 - t0) * 1000.0

                stitched = patcher.stitch(mask_patches)
                stitched_torch_u8 = torch.clip(stitched * 255.0, 0, 255).type(torch.uint8)
                stitched_np_u8 = stitched_torch_u8.cpu().numpy()

                # frame_mask_bgr = frame_bgr.copy()
                # # combine with frame - bgr
                # frame_mask_bgr = frame_bgr.copy()
                # frame_mask_bgr[:, :, CHANNEL_MASK] |= stitched

                # done!
                # the done queue gets a tuple of (id, frame_idx, frame_mask_bgr, dt)
                seg_queue_done.put([id, frame_idx, stitched_np_u8, dt])


def signal_handler(sig, frame):
    global frames_per_batch, seg_queue_todo, segmenters
  
    if PARALLEL:
        print('requesting segmenter threads shutdown..')
        for id in range(frames_per_batch):
            seg_queue_todo.put(None)
            time.sleep(0.1)

        for segmenter in segmenters:
            segmenter.join()

    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

argp = argparse.ArgumentParser()

argp.add_argument('model', type=str, help="model")
argp.add_argument('checkpoint', type=str, help="checkpoint")
argp.add_argument('input', type=str, help="description of which input frames to infer on")
argp.add_argument('--patch-width', type=int, required=True, help="patch width (224, 384, ..)")
argp.add_argument('--patch-height', type=int, required=True, help="patch height (224, 384, ..)")
argp.add_argument('--patch-stride-width', type=int, default=0, help="patch stride (default patch width)")
argp.add_argument('--patch-stride-height', type=int, default=0, help="patch stride (default patch height)")
argp.add_argument('--first-frame', type=int, default=0, help="first frame (default 0)")
argp.add_argument('--last-frame', type=int, default=-1, help="last frame (inclusive, default -1)")
argp.add_argument('--frame-step', type=int, default=1, help="frame hop size")
argp.add_argument('--original-size', type=str, default=None, help="All the frames coming in, whether from video or from jpes, should have this size, width x height a comma separated list of ints)")
argp.add_argument('--inference-size', type=str, default=None, help="scale frame to this width x height before inference, a comma separated list of ints for absolute)")
argp.add_argument('--jobs', type=int, default=1, help="number of inference jobs per cuda device")
argp.add_argument('--report-interval', type=int, default=1000, help="frame interval for stats report")
argp.add_argument('--onnx', type=str, default=None, help="filename of onnx export")
argp.add_argument('--limit', type=int, default=0, help="limit frame number")
argp.add_argument('--out-dir', type=str, default=None, help="directory to put inference results into")
argp.add_argument('--model-id-suffix', type=str, default=None, help="usually we put a suffix on the output file name that identifies the segmentation method, like effs20231008halfstride")
argp.add_argument('--pad-height', type=int, default=0, help="how much to pad on the bottom of the frame")
opt = argp.parse_args()
pad_height = opt.pad_height

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


MIN_WORKER_JOBS = 8

CHANNEL_MASK = 1
transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

model_name = opt.model
model_architecture_id = opt.model
fn_checkpoint = opt.checkpoint
fn_input = opt.input
patch_width = opt.patch_width
patch_height = opt.patch_height
patch_stride_width = opt.patch_stride_width
patch_stride_height = opt.patch_stride_height

MODEL_NAME = fn_checkpoint.split('/')[-1].split('.')[0]


# check for cuda devices
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

devices = get_cuda_devices()
num_devices = len(devices)
jobs = opt.jobs
if num_devices > 1 or jobs > 1:
    PARALLEL = True
else:
    PARALLEL = False

if not PARALLEL:
    print(f'loading model {model_name} from {fn_checkpoint}..')
    in_channels = 3
    num_class = 1  # TODO: for regression this might need to be 1
    model = MODEL_LOADERS[model_name](fn_checkpoint, multigpu = True, in_channels = in_channels, num_class = num_class)
    model.to(devices[0]).eval()

print(f"{Fore.YELLOW}Yo, reading frames not video!{Style.RESET_ALL}")
vin = FrameReader(
    fn_input=fn_input,
    inference_width=inference_width,
    inference_height=inference_height,
    limit = opt.limit
)
video_in = False


print(f"""
patcher = Patcher(
    frame_width={inference_width},
    frame_height={inference_height},
    patch_width={patch_width},
    patch_height={patch_height},
    stride_width={patch_stride_width},
    stride_height={patch_stride_height},
    pad_width={0},
    pad_height={pad_height}
)
""")
      
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


# not video, but particular blown-out frames:
fn_output = f"{out_dir}/{{:s}}.png"
print(f"{Fore.MAGENTA}writing output frames to {fn_output}..{Style.RESET_ALL}")
print(f'saving output frames to {fn_output}..')


# loop over frames
first_frame_idx = opt.first_frame
last_frame_idx = min(opt.last_frame if opt.last_frame > -1 else vin.number_of_frames, vin.number_of_frames)
frame_step = opt.frame_step
frames_per_batch = num_devices * jobs

if PARALLEL:  # without CUDA_VISIBLE_DEVICES, this will attempt to use all available GPUs
    print(f"{Fore.YELLOW}WARNING: Parallel codepath is happening{Style.RESET_ALL}")
    print(f'starting {frames_per_batch} segmenter processess..')
    seg_queue_todo = Queue()
    seg_queue_done = Queue()
    
    segmenters = [
        Thread(
            target=segment_thread,
            args=(id, seg_queue_todo, seg_queue_done, devices[id % num_devices], model_name, fn_checkpoint, CHANNEL_MASK, transform)
            )
        for id in range(frames_per_batch)]

    for segmenter in segmenters:
        segmenter.start()

    time.sleep(0.1)

pbar = tqdm(
    range(first_frame_idx, last_frame_idx - first_frame_idx + 1, frame_step),
    total=(last_frame_idx - first_frame_idx + 1) // frame_step)

# main loop
last_report = -1
cur_frame_idx = first_frame_idx
pending = 0
while ((cur_frame_idx <= last_frame_idx) or pending > 0) and not (cur_frame_idx == last_frame_idx and pending == 0):
    #print(f'cur_frame_idx {cur_frame_idx} last_frame_idx {last_frame_idx} pending {pending}')

    if not PARALLEL:
        frame = vin[cur_frame_idx]
        _, frame_proc, dt = process_frame(cur_frame_idx, frame, transform)
        if last_report < 0 or cur_frame_idx - last_report > opt.report_interval:
            tqdm.write(f'frame {cur_frame_idx}: inference in {dt:.2f}ms')
            last_report = cur_frame_idx
        assert frame_proc.ndim == 2, f"frame_proc.ndim is {frame_proc.ndim}"
        write_frame(
            frame_idx=cur_frame_idx,
            frame_bgr=frame_proc,
            original_height=original_height,
            original_width=original_width
        )
        cur_frame_idx += frame_step
        pbar.update(1)
        continue

    # parallel mode
    idxs = list(range(cur_frame_idx, min(cur_frame_idx + frames_per_batch * frame_step, last_frame_idx), frame_step))
    for idx in idxs:
        seg_queue_todo.put([idx, vin[idx]])
        pending += 1

    if pending > MIN_WORKER_JOBS * num_devices * jobs or (cur_frame_idx >= last_frame_idx):
        results = []
        while pending > MIN_WORKER_JOBS * num_devices * jobs or (cur_frame_idx >= last_frame_idx):
            try:
                results.append(seg_queue_done.get(block=pending > MIN_WORKER_JOBS * num_devices * jobs))
                pending -= 1
                if pending == 0:
                    break
            except:
                time.sleep(0.1)
                break

        # sort by frame number
        # id, frame_idx, frame_mask_bgr, dt
        results = sorted(results, key = lambda x: x[1])

        if last_report < 0 or (cur_frame_idx - last_report > opt.report_interval):
            for result in results:
                tqdm.write(f'frame {result[1]}: inference in {result[3]:.2f}ms (cuda{result[0] % num_devices})')
                tqdm.write(f'{pending} jobs pending')
                last_report = cur_frame_idx

        for result in results:
            frame_idx = result[1]
            frame_bgr = result[2]  # may actually be grayscale and thus ndim = 2
            assert frame_bgr.ndim == 2, f"frame_proc.ndim is {frame_proc.ndim}"
            write_frame(
                frame_idx=frame_idx,
                frame_bgr=frame_bgr,
                original_height=original_height,
                original_width=original_width
            )

        pbar.update(len(results))

    if len(idxs) > 0:
        # advance if not waiting in mp mode
        cur_frame_idx = idxs[-1] + frame_step

tqdm.write('finishing..')
pbar.close()

if PARALLEL:
    print('requesting segmenter threads shutdown..')
    for id in range(frames_per_batch):
        seg_queue_todo.put(None)
        time.sleep(0.1)

    for segmenter in segmenters:
       segmenter.join()

