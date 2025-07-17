import glob
import os
import cv2
import shutil

fns = glob.glob('fixup*.png')

for fn in fns:
   mask = cv2.imread(fn, -1)

   number = fn.replace('_nonfloor.png', '')[-6:]
   fn_orig = f'/mnt/nas/volume1/videos/baseball/clips/20200725PIT-STL-CFCAM-PITCHCAST_inning1/20200725PIT-STL-CFCAM-PITCHCAST_inning1_{number}.jpg'

   frame = cv2.imread(fn_orig, -1)

   shutil.copyfile(fn_orig, 'out/{}'.format(fn.replace('_nonfloor.png', '.jpg')))

   mask[:, :, :3] = frame

   cv2.imwrite(f'out/{fn}', mask)


