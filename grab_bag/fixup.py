import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
import os
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
import shutil



from segmenter import my_image_segmenter
from segmenter import myindices, myws

# load a sample image
import urllib

dirty_frames = set()
color_idx = 0
MASK_COLORS = ['magenta', 'cyan', 'green', 'blue', 'red', 'yellow']

CLEANUP_KERNEL_SIZE = 3
CLEANUP_MIN_AREA = 64
CLOSE_KERNEL_SIZE = 5

cur_frame = None
cur_mask = None
cur_modified = False

def export_file(infile):
    global opt

    frame_bgr, mask, _ = load_frame(infile, bgr = True, clean = False)

    marked = (mask == 1)
    mask[marked] = 255
    mask[~marked] = 0
    mask = mask.astype(np.uint8)

    frame_bgra = np.zeros((*frame_bgr.shape[:2], 4), dtype = np.uint8)
    frame_bgra[:, :, :3] = frame_bgr[...,:3]
    frame_bgra[:, :, 3] = mask

    outfile_base = os.path.join('fixed', infile.split('/')[-1].replace(f'_{opt.dataset_kind}.png', ''))

    # original frame
    shutil.copyfile(infile.replace(f'_{opt.dataset_kind}.png', '.jpg'), f'{outfile_base}.jpg')

    # annotation
    outfile = f'{outfile_base}_{opt.dataset_kind}.png'
    tqdm.write(f'writing {outfile}')
    cv2.imwrite(outfile, frame_bgra)

    # frame with mask
    outfile = f'{outfile_base}_masked.jpg'
    frame_bgr[:, :, 1] |= mask
    tqdm.write(f'writing {outfile}')
    cv2.imwrite(outfile, frame_bgr)


def load_frame(fn_mask, bgr = False, clean = True):
    global cur_frame, cur_mask, cur_modified, opt

    mask = cv2.imread(fn_mask, -1)
    if mask is None:
        print(f'could not load mask {fn_mask}!')
        return None, None

    mask = mask[:, :, 3]

    fn_frame = fn_mask.replace(f'_{opt.dataset_kind}.png', '.jpg')
    frame = cv2.imread(fn_frame, -1)
    if frame is None:
        print(f'could not load frame {fn_frame}!')
        return None, None

    if not bgr:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if mask.shape[:2] != frame.shape[:2]:
        # reshape mask
        print(f'reshaping mask {mask.shape[:2]} to {frame.shape[:2]}')
        mask = cv2.resize(mask, frame.shape[:2], interpolation = cv2.INTER_NEAREST)

    print(f'loaded {fn_mask.replace(f"_{opt.dataset_kind}.png", "")}')

    mask[mask < 250] = 0
    mask[mask > 250] = 1

    mask_orig = mask.copy()

    if clean:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLEANUP_KERNEL_SIZE, CLEANUP_KERNEL_SIZE))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        output = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        for i in range(1, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if area < CLEANUP_MIN_AREA:
                mask[labels == i] = 0

    if np.all(mask == mask_orig):
        dirty = False
    else:
        dirty = True

    cur_frame = frame
    cur_mask = mask
    cur_modified = dirty

    return frame, mask, dirty

def move_by(offset):
    global segmenter, curfile, infiles, dirty_frames
    if offset == 0:
        return
    try:
        d = segmenter.save_mask(infiles[curfile])
        if d:
            dirty_frames.add(infiles[curfile])

        curfile = (curfile + offset) % len(infiles)
        print(f'curfile {curfile}')

        frame, mask, dirty = load_frame(infiles[curfile])
        #lims = segmenter.get_lim()
        segmenter.set_image(frame, mask = mask, fn = infiles[curfile].split('/')[-1].replace(f'_{opt.dataset_kind}.png', ''))
        #segmenter.set_lim(*lims)
        segmenter.set_dirty(dirty)
    except Exception as e:
        print(f'ERROR changing frame {curfile} {offset}: {e}')


def close_mask():
    global segmenter

    mask = segmenter.mask.copy()

    output = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < CLEANUP_MIN_AREA:
            mask[labels == i] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    segmenter.set_mask(mask)



def on_press(event):
    global segmenter, curfile, infiles, dirty_frames, color_idx
    print('press', event.key)
    sys.stdout.flush()
    update = False

    if event.key == 'q':
        print('quit')
        d = segmenter.save_mask(infiles[curfile])
        if d:
            dirty_frames.add(infiles[curfile])
        plt.close()
        return
    elif event.key == 's' or event.key == 'enter':
        d = segmenter.save_mask(infiles[curfile])
        if d:
            dirty_frames.add(infiles[curfile])
    elif event.key == 'R':
        infiles = list(sorted(glob(os.path.join(opt.path, f'*_{opt.dataset_kind}.png'))))
        old = curfile
        curfile = 0
        move_by(old)
        print(f'loaded {len(infiles)} files')
    elif event.key == 'a':
        segmenter.set_mode('add')
    elif event.key == 'd':
        segmenter.set_mode('del')
    elif event.key == 'c':
        segmenter.set_mode('copy')
    elif event.key == 'v':
        segmenter.paste()
    elif event.key == 'l':
        segmenter.set_mode('close')
    elif event.key == 'w':
        segmenter.set_mode('watershed')
    elif event.key == 't':
        if segmenter.mode == 'add':
            segmenter.set_mode('del')
        else:
            segmenter.set_mode('add')
    elif event.key == 'u':
        segmenter.undo()
    elif event.key == 'r':
        segmenter.redo()
    elif event.key == 'right':
        move_by(1)
    elif event.key == 'left':
        move_by(-1)
    elif event.key == 'up':
        move_by(10)
    elif event.key == 'down':
        move_by(-10)
    elif event.key == '0':
        segmenter.reset_zoom()
    elif event.key == '1':
        segmenter.set_mask_color(alpha = 0.0)
    elif event.key == '2':
        segmenter.set_mask_color(alpha = 0.3)
    elif event.key == '3':
        segmenter.set_mask_color(alpha = 1.0)
    elif event.key == '[':
        segmenter.set_mask_color(alpha = segmenter.mask_alpha - 0.1)
    elif event.key == ']':
        segmenter.set_mask_color(alpha = segmenter.mask_alpha + 0.1)
    elif event.key == '=':
        color_idx = (color_idx + 1) % len(MASK_COLORS)
        segmenter.set_mask_color(mask_colors= MASK_COLORS[color_idx])

argp = argparse.ArgumentParser()

argp.add_argument('path', type=str, help="path to fixup files")
argp.add_argument('--dataset-kind', type=str, default='nonfloor', help="dataset kind")
argp.add_argument('--output', type=str, default=None, help="output base name")

opt = argp.parse_args()

outpath = opt.output or 'out'

infiles = list(sorted(glob(os.path.join(opt.path, f'*_{opt.dataset_kind}.png'))))
if len(infiles) == 0:
    print(f'no fixup files found in {opt.path}!')
    sys.exit(0)

print(f'processing {len(infiles)} frames')

curfile = 0
frame, mask, dirty = load_frame(infiles[0])

plt.rcParams["keymap.quit"] = ''
plt.rcParams["keymap.fullscreen"] = ''
plt.rcParams["keymap.save"] = ''

segmenter = my_image_segmenter(frame,
                               mask_colors=MASK_COLORS[color_idx],
                               mask_alpha=0.3,
                               figsize=(16, 9),
                               mask=mask,
                               filename = infiles[0].split('/')[-1].replace(f'_{opt.dataset_kind}.png', ''))
segmenter.set_dirty(dirty)

xl = segmenter.fig.axes[0].set_xlabel('erasing')
xl.set_visible(segmenter.erasing)

segmenter.fig.canvas.mpl_connect('key_press_event', on_press)

plt.show()

print(f'exporting {len(dirty_frames)} masks..')

if len(dirty_frames) > 0:
    if not os.path.exists('fixed'):
        os.mkdir('fixed')

for infile in tqdm(dirty_frames, total = len(dirty_frames)):
    export_file(infile)
