from feathered_paste_for_images_of_the_same_size import (
     feathered_paste_for_images_of_the_same_size
)
from get_normalization_and_chw_transform import (
     get_normalization_and_chw_transform
)
import sys
import cv2
import argparse
import torch
import torch.onnx
from videoreader import VideoReader
from unettools import *
from tqdm import tqdm
import time
from threading import Thread
from queue import Queue
import signal
import sys
import subprocess as sp
import glob
from pathlib import Path
from joblib import Parallel, delayed
import shutil
import os
import multiprocessing

from torchvision import transforms
from DummyWith import DummyWith
from get_cuda_devices import get_cuda_devices
from Patcher import Patcher
import numpy as np

# https://pytorch.org/docs/stable/onnx.html
# default: 14, latest: 16
ONNX_OPSET_VERSION = 14

def load_frame(fn, idx):
    global ts_model
    try:
        print(f'reading {fn}')
        frame_bgr = cv2.imread(fn)

        if frame_bgr is None:
            raise Exception('error loading!')

        if frame_bgr.shape[2] == 4:
            print(f'WARNING removing alpha channel from {fn}')
            frame_bgr = frame_bgr[:, :, :3]

        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if ts_model:
            ts_mask = ts_inference_frame(ts_model, ts_width, ts_height, frame, devices[0], WITH_AMP)
            ts_mask *= 255.
            ts_mask_uint8 = np.clip(ts_mask.cpu().numpy(), 0, 255)
            frame_tmp = np.zeros(shape = (frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
            assert frame.dtype == np.uint8 and frame.shape[2] == 3
            frame_tmp[:, :, :3] = frame
            frame_tmp[:, :, 3] = ts_mask_uint8
            frame = frame_tmp

        return idx, frame
    except Exception as e:
        print(f'ERROR processing {fn}:\n{e}')
        return idx, None

class FrameReader(object):

    def __init__(self, fn_input, limit = 0, cache_frames=None, randomize=True):
        global opt, ts_model, ts_width, ts_height, devices, WITH_AMP

        self.fn_frames = []
        if '**' in fn_input:
            tokens = fn_input.split('/')
            path = '/'.join(tokens[:-1])
            self.fn_frames = [str(x) for x in Path(path).rglob(tokens[-1].replace('**', '*'))]
        elif '*' in fn_input:
            self.fn_frames = glob.glob(fn_input)
        else:
            self.fn_frames = [fn_input]

        #self.fn_frames = [fn.replace('_nonwood.png', '.jpg').replace('_nonfloor.png', '.jpg') for fn in self.fn_frames]

        if len(self.fn_frames) == 0 or len(self.fn_frames) < opt.skipfirst:
            raise Exception('no frames found!')

        self.fn_frames = list(sorted(self.fn_frames))

        if opt.skipfirst > 0:
            self.fn_frames = self.fn_frames[opt.skipfirst:]

        if limit > 0 and len(self.fn_frames) > limit:
            if randomize:
                np.random.seed(opt.seed)
                perm = np.random.permutation(len(self.fn_frames))[:limit]
            else:
                perm = range(0, limit)
            self.fn_frames = [self.fn_frames[j] for j in perm]

        if cache_frames is None:
            if len(self.fn_frames) < 200:
                cache_frames = True
            else:
                cache_frames = False

        self.cache_frames = cache_frames

        if cache_frames:
            print(f'caching {len(self.fn_frames)} frames..')

            if not ts_model:
                results = Parallel(n_jobs=min(multiprocessing.cpu_count() // 2, 32), backend = 'threading')(delayed(load_frame)(self.fn_frames[i], i)
                                                for i in tqdm(range(len(self.fn_frames)), total = len(self.fn_frames)))
            else:
                results = [load_frame(self.fn_frames[i], i) for i in tqdm(range(len(self.fn_frames)), total = len(self.fn_frames))]

            results = sorted(results, key = lambda x: x[0])

            good_idx = [i for i in range(len(results)) if not results[i][1] is None]
            results = [results[i] for i in good_idx]
            self.fn_frames = [self.fn_frames[i] for i in good_idx]

            self.frames = [res[1] for res in results]
            self.frame_height = self.frames[0].shape[0]
            self.frame_width = self.frames[0].shape[1]
        else:
            first_frame = load_frame(self.fn_frames[0], 0)[1]
            self.frame_height = first_frame.shape[0]
            self.frame_width = first_frame.shape[1]

        self.number_of_frames = len(self.fn_frames)
        self.frame_rate = opt.framerate

    def __getitem__(self, key):
        if self.cache_frames:
            return self.frames[key]
        else:
            frame = load_frame(self.fn_frames[key], key)[1]
            if frame is None:
                frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            return frame

    def __len__(self):
        return self.number_of_frames



onnx_written = False
def process_frame(model, frame_idx, frame_bgr, device = None):
    global MODEL_NAMEl, ONNX_OPSET_VERSION, opt, onnx_written, frame_width, frame_height, in_channels, vin

    if in_channels == 4:
        transform = TRANSFORM_NORM_IMAGE_TS
    else:
        transform = get_normalization_and_chw_transform()

    if device is None:
        device = devices[0]

    if frame_bgr.shape[0] != frame_height or frame_bgr.shape[1] != frame_width:
        frame_bgr = cv2.resize(frame_bgr, (frame_width, frame_height), interpolation = cv2.INTER_LINEAR)

    if type(vin) is FrameReader:
        #print('bgr<=>rgb')
        frame_rgb = frame_bgr
        if in_channels == 4:
            frame_bgr = frame_bgr[:, :, :3]
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
    else:
        #print('bgr=rgb')
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # SLOW FML
    #TRANSFORM_NORM_IMAGE(frame_rgb)

    # patch frame
    frame_tens = transform(frame_rgb).to(device)
    patches = patcher.patch(frame = frame_tens, device = device, channels=in_channels)

    # infer
    with torch.no_grad():
        outc = None

        with WITH_AMP():
            NONFLOOR_CHANNEL = 0
            t0 = time.time()
            if getattr(model, 'return_features', False) or getattr(model, 'classification_head', False):
                pred, feats = model(patches)
                #outc = classifier(feats).cpu().numpy()
                #print('classifier', outc.shape)
            else:
                pred = model(patches)

            

            if matting:
                pred = torch.sigmoid(pred).detach()
                mask_patches = pred[:, 0, :, :]
                #print('mask_patches', mask_patches.shape)
            else:
                pred = torch.sigmoid(pred).detach()
                mask_patches = pred[:, NONFLOOR_CHANNEL, :, :]
            #print('AFTER DEVICES')
            #torch.cuda.synchronize()
            t1 = time.time()
            dt = (t1 - t0) * 1000.0

    # stitch mask
    stitched = patcher.stitch(mask_patches)
    stitched = torch.clip(stitched * 255.0, 0, 255).type(torch.uint8)
    stitched = stitched.cpu().numpy()

    frame_mask_bgra = np.zeros((frame_bgr.shape[0], frame_bgr.shape[1], 4), dtype=np.uint8)
    frame_mask_bgra[:, :, :3] = frame_bgr
    frame_mask_bgra[:, :, 3] = stitched

    t1 = time.time()

    return frame_idx, frame_mask_bgra, dt, outc

def write_frame(frame_idx, frame_bgra):
    # frame_bgr is always bgr
    global ffmpeg_pipe, vin, video_in, frame_width, frame_height, opt

    if frame_bgra.shape[0] == 1088 and frame_bgra.shape[1] == 1920:
        # TODO: remove hard coded value - just here for speed reasons on S3
        # better way: read original image, resize according to dims
        frame_bgra = cv2.resize(frame_bgra, (1920, 1080), interpolation=cv2.INTER_LINEAR)

    mask = frame_bgra[:, :, 3]
    frame_bgr = frame_bgra[:, :, :3]

    if frame_bgr.shape[0] != frame_height or frame_bgr.shape[1] != frame_width:
        frame_bgr = cv2.resize(frame_bgr, (frame_width, frame_height), interpolation = cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (frame_width, frame_height), interpolation = cv2.INTER_LINEAR)

    if video_in:
        fn_out = fn_output.format(frame_idx)
        fn_frame = f'{frame_idx}'
    else:
        fn_frame = vin.fn_frames[frame_idx].split('/')[-1].split('.')[0]
        fn_out = fn_output.format(fn_frame)

    tok = fn_frame.split('/')[-1].replace('.jpg', '')

    #print('frame_bgr', frame_bgr.shape, frame_bgr.dtype, tok)
    if False:  # 5m12
        frame_bgr_masked = frame_bgr.copy()
        frame_bgr_masked[:, :, CHANNEL_MASK] |= mask
    else: # 7m36.809s
        
        green_background_np_uint8 = np.zeros((frame_bgr.shape[0], frame_bgr.shape[1], 3), dtype=np.uint8)
        green_background_np_uint8[:, :, CHANNEL_MASK] = 255

        frame_bgr_masked = feathered_paste_for_images_of_the_same_size(
            bottom_layer_color_np_uint8=green_background_np_uint8,
            top_layer_rgba_np_uint8=frame_bgra,
        )




    if not opt.no_burn:
        # print(f'burning: {tok} {fn_frame} {fn_out}')
        frame_bgr_masked = cv2.putText(
            frame_bgr_masked, tok, (8, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    if video_out or not opt.fixup:
        frame_bgr = frame_bgr_masked

    # write frame
    if video_out:
        #vout.write(frame_bgr)
        frame_rgb = cv2.cvtColor(frame_bgr_masked, cv2.COLOR_BGR2RGB)
        ffmpeg_pipe.stdin.write(frame_rgb.tobytes())
    else:
        if '*' in fn_out:
            fn_out = fn_out.replace('*', '+')

        out_path = '/'.join(fn_out.split('/')[:-1])
        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)

        tqdm.write(f'writing {fn_out}')
        if not opt.fixup:
            cv2.imwrite(fn_out, frame_bgr)
        else:
            #mask = cv2.bilateralFilter(mask, 15, 75, 75)

            #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            #mask_high = cv2.morphologyEx(mask_high, cv2.MORPH_OPEN, kernel)

            mask_high = mask > 128
            mask[mask_high] = 255

            if not opt.fixup_sourcepath:
                if not video_in:
                    # make 1:1 copy of frame jpeg
                    if os.path.exists(vin.fn_frames[frame_idx]):
                        shutil.copyfile(vin.fn_frames[frame_idx], fn_out)
                    else:
                        cv2.imwrite(fn_out, frame_bgr)
                else:
                    cv2.imwrite(fn_out, frame_bgr)

            if opt.fixup_sourcepath:
                fn_out = vin.fn_frames[frame_idx]
                suffix = opt.fixup_sourcepath_suffix
            else:
                suffix = ''

            cv2.imwrite(fn_out.replace('.jpg', '_masked.jpg'), frame_bgr_masked)
            cv2.imwrite(fn_out.replace('.jpg', '_mask.jpg'), cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

            frame_bgra = np.zeros((*frame_bgr.shape[:2], 4), dtype=np.uint8)
            frame_bgra[:, :, :3] = frame_bgr
            frame_bgra[:, :, 3] = mask



            fn_mask_out = fn_out.replace('.jpg', f'_{opt.dataset_kind}{suffix}.png')
            tqdm.write(f'yoyo writing {fn_mask_out}')
            cv2.imwrite(fn_mask_out, frame_bgra)

def segment_thread(thread_id, seg_queue_todo, seg_queue_done, device, model_name, fn_checkpoint, CHANNEL_MASK, transform, twostage, videoin):
    print(f'segmenter process {thread_id} starting (device {device})')

    print(f'segmenter{thread_id}: loading model {model_name} from {fn_checkpoint}..')
    in_channels = 4 if twostage else 3
    num_class = 1 if matting else 2
    num_class = 1  #
    model = MODEL_LOADERS[model_name](fn_checkpoint, multigpu = True, in_channels = in_channels, num_class = num_class, transpose_final = not opt.no_transpose_final)
    model.to(device).eval()

    if twostage:
        ts_name, ts_checkpoint, ts_width, ts_height = twostage.split(':')

        print(f'loading two stage model {ts_name} from {ts_checkpoint}..')
        ts_model = MODEL_LOADERS[ts_name](ts_checkpoint, multigpu = True, transpose_final = not opt.no_transpose_final)
        ts_model = ts_model.to(device).eval()
        ts_width = int(ts_width)
        ts_height = int(ts_height)
    else:
        print('single stage processing')
        ts_model = None
        ts_width = -1
        ts_height = -1




    print(f'segmenter thread {thread_id} ready')
    while True:
        item = seg_queue_todo.get(block=True)

        if item is None:
            print(f'segmenter process {thread_id} exiting.')
            del model
            return 0

        # got work to do
        frame_idx, frame_bgr = item

        if frame_bgr is None:
            print(f'warning: segmenter process {thread_id} received empty frame')
            continue

        _, frame_mask_bgr, dt, _ = process_frame(model, frame_idx, frame_bgr, device = device)

        # done!
        seg_queue_done.put([thread_id, frame_idx, frame_mask_bgr, dt])

def signal_handler(sig, frame):
    global video_out, frames_per_batch, seg_queue_todo, segmenters
    if video_out:
        print('closing video output file')
        #vout.release()
        ffmpeg_pipe.stdin.close()
        ffmpeg_pipe.stderr.close()
        ffmpeg_pipe.wait()

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

argp.add_argument('model', type=str, help="model architecture family id") 
argp.add_argument('checkpoint', type=str, help="checkpoint")
argp.add_argument('input', type=str, help="input video file")
argp.add_argument('--output', type=str, default=None, help="output base name")
argp.add_argument('--dataset-kind', type=str, default='nonfloor', help="dataset kind (nonfloor, nonwood)")
argp.add_argument('--patch-width', type=int, default=384, help="patch width (224, 384, ..)")
argp.add_argument('--patch-height', type=int, default=384, help="patch width (224, 384, ..)")
argp.add_argument('--stride-width', type=int, default=0, help="patch stride, width")
argp.add_argument('--stride-height', type=int, default=0, help="patch stride, height")
argp.add_argument('--stride-boost', type=int, default=5, help="multiplier for center region (set 0 to disable)")
argp.add_argument('--first-frame', type=int, default=0, help="first frame (default 0)")
argp.add_argument('--last-frame', type=int, default=-1, help="last frame (inclusive, default -1)")
argp.add_argument('--frame-step', type=int, default=1, help="frame hop size")
argp.add_argument('--frame-scale', type=str, default=None, help="scale frame before inference (widthxheight)")
argp.add_argument('--frame-pad', type=str, default=None, help="scale frame before inference (widthxheight)")
argp.add_argument('--video-out', action='store_true', help="write output video")
argp.add_argument('--jobs', type=int, default=1, help="number of inference jobs per cuda device")
argp.add_argument('--report-interval', type=int, default=1000, help="frame interval for stats report")
argp.add_argument('--onnx', type=str, default=None, help="filename of onnx export")
argp.add_argument('--limit', type=int, default=0, help="limit frame number")
argp.add_argument('--amp', action='store_true', help="mixed precision training")
argp.add_argument('--twostage', type=str, default=None, help="two stage network (full frame, patches)")
# two stage: model_name:checkpoint:frame_width:frame_height
argp.add_argument('--ffmpeg', type=str, default='/usr/local/bin/ffmpeg', help="ffmpeg executable")
argp.add_argument('--norand', action='store_true', help="do not randomize images when limiting (select first n)")
argp.add_argument('--fixup', action='store_true', help="fixup mode (export to fixup directory)")
argp.add_argument('--seed', type=int, default=1337, help="seed for random number generator")
argp.add_argument('--no-burn', action='store_true', help="do not burn frame name / number")
argp.add_argument('--skipfirst', type=int, default=0, help="frames to skip in the beginning")
argp.add_argument('--framerate', type=float, default=59.94, help="frame rate when creating video from files")
argp.add_argument('--frames', type=str, default=None, help="frame numbers, comma separated")
argp.add_argument('--pad-width', type=int, default=0, help="width padding")
argp.add_argument('--pad-height', type=int, default=0, help="height padding")
argp.add_argument('--halfstride', action='store_true', help="half patches overlap")
argp.add_argument('--no-transpose-final', action='store_true', help="u3: no final transpose")
argp.add_argument('--fixup-sourcepath', action='store_true', help="save fixup into sourcepath")
argp.add_argument('--fixup-sourcepath-suffix', type=str, default='_pred', help="suffix for segmentation prediction in source path")
argp.add_argument('--invert', action='store_true', help="invert mask, i.e. green floor")
opt = argp.parse_args()

MIN_WORKER_JOBS = 8

CHANNEL_MASK = 1
transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


model_name = opt.model
fn_checkpoint = opt.checkpoint
fn_video = opt.input
video_out = opt.video_out
fn_output = opt.output
patch_width = opt.patch_width
patch_height = opt.patch_height
stride_width = opt.stride_width or patch_width
stride_height = opt.stride_height or patch_height
pad_width = opt.pad_width
pad_height = opt.pad_height
opt.matting = '_a_' in opt.checkpoint or '_ar_' in opt.checkpoint or '_at_' in opt.checkpoint or '_atr_' in opt.checkpoint
matting = opt.matting
#rgba = '_rgba_' in opt.checkpoint

if opt.halfstride:
    stride_width = patch_width // 2
    stride_height = patch_height // 2

MODEL_NAME = fn_checkpoint.split('/')[-1].split('.')[0]
VIDEO_NAME = fn_video.split('/')[-1].split('.')[0]

if opt.twostage:
    in_channels = 4
else:
    in_channels = 3

if '_amp_' in model_name:
    opt.amp = True

if '_ntf_' in model_name:
    opt.no_transpose_final = True

WITH_AMP = torch.cuda.amp.autocast if opt.amp else DummyWith

# check for cuda devices
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

devices = get_cuda_devices()
num_devices = len(devices)
jobs = opt.jobs
if num_devices > 1 or jobs > 1 and opt.twostage is None:
    PARALLEL = True
else:
    # two stage only on single device
    PARALLEL = False

ts_model = None
ts_width = -1
ts_height = -1

if not PARALLEL:
    print(f'loading model {model_name} from {fn_checkpoint}..')
    if matting:
        num_class = 1
    else:
        num_class = 2
    model = MODEL_LOADERS[model_name](fn_checkpoint, multigpu = True, in_channels = in_channels, num_class = num_class, transpose_final = not opt.no_transpose_final)
    model.to(devices[0]).eval()

    if opt.twostage:
        ts_name, ts_checkpoint, ts_width, ts_height = opt.twostage.split(':')

        print(f'loading two stage model {ts_name} from {ts_checkpoint}..')
        ts_model = MODEL_LOADERS[ts_name](ts_checkpoint, multigpu = True, transpose_final = not opt.no_transpose_final)
        ts_model = ts_model.to(devices[0]).eval()
        ts_width = int(ts_width)
        ts_height = int(ts_height)
    else:
        print('single stage processing')
        ts_model = None
        ts_width = -1
        ts_height = -1


if '*' in fn_video or '.png' in fn_video.lower() or '.jpg' in fn_video.lower() or '.jpeg' in fn_video.lower():
    vin = FrameReader(fn_video, limit = opt.limit, randomize=not opt.video_out and not opt.norand)
    video_in = False
else:
    print(f'opening input video {fn_video}..')
    if opt.twostage:
        raise Exception('twostage not supported for video input yet!')
    vin = VideoReader(fn_video)
    video_in = True

# frame size known here
scale_factor = opt.frame_scale
if scale_factor:
    if 'x' in scale_factor:
        # specified size
        w, h = scale_factor.split('x')
        frame_width = int(w)
        frame_height = int(h)
    else:
        # specified factor
        scale_factor = float(scale_factor)
        frame_width = int(vin.frame_width * scale_factor)
        frame_height = int(vin.frame_height * scale_factor)
else:
    # orginial resolution
    frame_width = vin.frame_width
    frame_height = vin.frame_height

frame_rate = vin.frame_rate

print(f"""
patcher = Patcher(
    frame_width={frame_width}, frame_height={frame_height},
    patch_width={patch_width}, patch_height={patch_height},
    stride_width={stride_width}, stride_height={stride_height},
    pad_width={pad_width}, pad_height={pad_height}, boost_center={opt.stride_boost}
)
"""
)
patcher = Patcher(
    frame_width=frame_width, frame_height=frame_height,
    patch_width=patch_width, patch_height=patch_height,
    stride_width=stride_width, stride_height=stride_height,
    boost_center=opt.stride_boost,
    pad_width=pad_width, pad_height=pad_height)

if video_out:
    if fn_output is None:
        fn_output = f'out/seg_{VIDEO_NAME}_{MODEL_NAME}.ts'
    else:
        if not '.' in fn_output:
            fn_output = f'{fn_output}.ts'

    fn_output = fn_output.replace('*', '+')
    print(f'writing output video {fn_output}..')

    if os.path.exists(opt.ffmpeg):
        ffmpeg_bin = opt.ffmpeg
    else:
        ffmpeg_bin = 'ffmpeg'

    # http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/
    command = [ffmpeg_bin,
            '-hide_banner', '-loglevel', 'error', '-nostats',
            '-y', # (optional) overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-s', f'{frame_width}x{frame_height}',
            '-pix_fmt', 'rgb24',
            '-r', f'{frame_rate:.2f}', # frames per second
            '-i', '-',
            '-an', '-pix_fmt', 'yuv420p',
            '-codec:v', 'libx264', '-crf', '22',
            f'{fn_output}' ]

    print('FFMPEG command: {}'.format(' '.join(command)))

    # try opening ffmpeg pipe - bail on failure
    ffmpeg_pipe = sp.Popen( command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
    dummy_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    res = ffmpeg_pipe.communicate(input = dummy_frame.tobytes())
    if ffmpeg_pipe.returncode != 0:
        print('could not open ffmpeg:', res[1].decode("utf-8"))
        sys.exit(1)
    del ffmpeg_pipe

    # ffmpeg works, do the real thing
    ffmpeg_pipe = sp.Popen( command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
else:
    if fn_output is None:
        if opt.fixup:
            fn_output = 'out/'
        else:
            fn_output = 'out/seg_'

    if video_in:
        if opt.fixup:
            # original frame name for fixup
            fn_output += f'{VIDEO_NAME}_{{:06d}}.jpg'
        else:
            fn_output += f'{VIDEO_NAME}_{MODEL_NAME}_{{:06d}}.jpg'
    else:
        if opt.fixup:
            # original frame name for fixup
            fn_output += f'{{:s}}.jpg'
        else:
            fn_output += f'{MODEL_NAME}_{{:s}}.jpg'

    print(f'saving output frames to {fn_output}..')


# loop over frames
first_frame_idx = opt.first_frame
last_frame_idx = min(opt.last_frame if opt.last_frame > -1 else vin.number_of_frames - 1, vin.number_of_frames - 1)
frame_step = opt.frame_step
frames_per_batch = num_devices * jobs

if PARALLEL:
    print(f'starting {frames_per_batch} segmenter processess..')
    seg_queue_todo = Queue()
    seg_queue_done = Queue()
    patcher_args = dict(frame_width=frame_width, frame_height=frame_height,
        patch_width=patch_width, patch_height=patch_height,
        stride_width=stride_width, stride_height=stride_height,
        pad_width=0, pad_height=0)
    segmenters = [
        Thread(
            target=segment_thread,
            args=(id, seg_queue_todo, seg_queue_done, devices[id % num_devices], model_name, fn_checkpoint, CHANNEL_MASK, transform, opt.twostage, type(vin) is VideoReader)
            )
        for id in range(frames_per_batch)]

    for segmenter in segmenters:
        segmenter.start()

    time.sleep(0.1)

if opt.frames:
    frame_indices = [int(fi) for fi in opt.frames.split(',')]
else:
    frame_indices = list(range(first_frame_idx, last_frame_idx + 1, frame_step))

if len(frame_indices) == 0:
    print('nothing to do!')
    sys.exit(0)

first_frame_idx = frame_indices[0]
last_frame_idx = frame_indices[-1]
frame_indices.append(last_frame_idx + 1)

#pbar = tqdm(
#    range(first_frame_idx, last_frame_idx - first_frame_idx + 1, frame_step),
#    total=(last_frame_idx - first_frame_idx) // frame_step)

pbar = tqdm(frame_indices, total=len(frame_indices) - 2)

# main loop
last_report = -1
cur_frame_idx2 = 0
cur_frame_idx = frame_indices[0]
pending = 0
outcs = []
last_written = cur_frame_idx - 1
results = []
while ((cur_frame_idx <= last_frame_idx) or pending > 0) and not (cur_frame_idx == last_frame_idx + 1 and pending == 0):
    #print(f'cur_frame_idx {cur_frame_idx} last_frame_idx {last_frame_idx} pending {pending}')

    if not PARALLEL:
        frame = vin[cur_frame_idx]

        if frame is None:
            print(f'warning: segmenter received empty frame')
            break

        _, frame_proc, dt, outc = process_frame(model, cur_frame_idx, frame)

        outcs.append(outc)

        if last_report < 0 or cur_frame_idx - last_report > opt.report_interval:
            # tqdm.write(f'frame {cur_frame_idx}: inference in {dt:.2f}ms')
            last_report = cur_frame_idx

        write_frame(cur_frame_idx, frame_proc)
        #cur_frame_idx += frame_step
        cur_frame_idx2 += 1
        cur_frame_idx = frame_indices[cur_frame_idx2]
        pbar.update(1)

        continue

    # parallel mode
    #idxs = list(range(cur_frame_idx, min(cur_frame_idx + frames_per_batch * frame_step, last_frame_idx), frame_step))
    idxs = frame_indices[cur_frame_idx2:cur_frame_idx2 + frames_per_batch]
    if idxs[-1] == last_frame_idx + 1:
        idxs = idxs[:-1]

    for idx in idxs:
        seg_queue_todo.put([idx, vin[idx]])
    pending += len(idxs)

    if len(idxs) > 0:
        # advance if not waiting in mp mode
        #cur_frame_idx = idxs[-1] + frame_step
        cur_frame_idx2 = min(cur_frame_idx2 + frames_per_batch, len(frame_indices) - 1)
        cur_frame_idx = frame_indices[cur_frame_idx2]

    if pending > MIN_WORKER_JOBS * num_devices * jobs or (cur_frame_idx >= last_frame_idx):
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
            tqdm.write(f'{len(results)} jobs finished, {pending} jobs pending')
            last_report = cur_frame_idx

        for result in results:
            if result[1] == last_written + 1:
                # tqdm.write(f'frame {result[1]}: inference in {result[3]:.2f}ms (cuda{result[0] % num_devices})')
                write_frame(*result[1:3])
                last_written += 1
                pbar.update(1)
            elif result[1] > last_written + 1:
                break


tqdm.write('finishing..')
pbar.close()

if PARALLEL:
    print('requesting segmenter threads shutdown..')
    for id in range(frames_per_batch):
        seg_queue_todo.put(None)
        time.sleep(0.1)

    for segmenter in segmenters:
       segmenter.join()


# close video output
if video_out:
    print('closing output video file')
    #vout.release()
    ffmpeg_pipe.stdin.close()
    ffmpeg_pipe.stderr.close()
    ffmpeg_pipe.wait()

print('all done.')
