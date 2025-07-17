import cv2
import argparse
import numpy as np
import sys
import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

argp = argparse.ArgumentParser()

argp.add_argument('inpath', type=str, help="path to process")
argp.add_argument('outpath', type=str, help="path to put output")

opt = argp.parse_args()

infiles = glob(os.path.join(opt.inpath, '*.png'))
print(f'processing {len(infiles)} files..')

for fn_mask in tqdm(infiles, total=len(infiles)):
    fn_frame = fn_mask.replace('_nonfloor.png', '.jpg').replace('_nonwood.png', '.jpg')
    if not os.path.exists(fn_frame):
        tqdm.write(f'missing frame {fn_frame} for mask {fn_mask}, skipping')
        continue

    mask = cv2.imread(fn_mask, -1)
    if mask is None:
        tqdm.write(f'could not read mask {fn_mask}, skipping')
        continue

    if mask.shape[2] != 4:
        tqdm.write(f'missing alpha channel in {fn_mask}, skipping')
        continue

    frame = cv2.imread(fn_frame)
    if frame is None:
        tqdm.write(f'could not read frame {fn_frame}, skipping')
        continue

    #ret, markers = cv2.connectedComponents(mask[:, :, 3])
    output = cv2.connectedComponentsWithStats(mask[:, :, 3], 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        tqdm.write(f'cc {i}: x {x} y {y} w {w} h {h} a {area}')

        frame_base = fn_frame.split('/')[-1].replace('.jpg', '')
        base = f'{frame_base}_object{i}'

        cc = mask[y:y+h, x:x+h, :]

        for j in range(cc.shape[1] - 1):
            allzero = True
            for k in range(j + 1, cc.shape[1]):
                if cc[:, k, 3].sum() > 0:
                    allzero = False
            if allzero:
                w = j

        for j in range(cc.shape[0] - 1):
            allzero = True
            for k in range(j + 1, cc.shape[0]):
                if cc[k, :, 3].sum() > 0:
                    allzero = False
            if allzero:
                h = j

        if w < 50 or h < 50:
            continue

        cc = cc[:h, :w, :]
        #plt.matshow(cc[:, :, 3])

        fn_mask = os.path.join(opt.outpath, f'{base}.png')
        cv2.imwrite(fn_mask, cc)
    #plt.show()
