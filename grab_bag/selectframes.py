import os
import argparse
from tqdm import tqdm
import shutil

# TODO: add video file support

argp = argparse.ArgumentParser()

argp.add_argument('inpath', type=str, help="path or video file for input")
argp.add_argument('outpath', type=str, help="path for output")
argp.add_argument('frames', type=int, nargs='+', help="frame numbers to select")
argp.add_argument('--dry', action='store_true', help="dry mode")

opt = argp.parse_args()

frame_numbers = opt.frames
inpath = opt.inpath
outpath = opt.outpath

if not os.path.exists(outpath):
    print(f'creating path {outpath}')
    if not opt.dry:
        os.makedirs(outpath, exist_ok=True)

for frame_number in tqdm(frame_numbers):
    fn_in = inpath.format(frame_number)
    base_out = fn_in.split('/')[-1]
    fn_out = os.path.join(outpath, base_out)
    tqdm.write(f'{fn_in} -> {fn_out}')
    if not opt.dry:
       shutil.copyfile(fn_in, fn_out)
