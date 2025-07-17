import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys


# load frame
fn_mask = 'basic/20200725PIT-STL-CFCAM-PITCHCAST_inning1_000300_nonfloor.png'
fn_mask = '/Users/felix/Dropbox/emersys2/segmenter_unet/segmentation_datasets/hou_core_2022-2023a_wood/thomas/HOUvLAC_PGM_core_att_11-14-2022_156000_nonwood.png'
img = cv2.imread(fn_mask, -1)
mask = img[:, :, 3]
img = img[..., :3]

img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# border of mask
masked = mask > 0

erosion_size = 10
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
inside = cv2.erode(mask, kernel)

border = np.logical_xor(masked.astype(bool), inside.astype(bool))


# work on uv
img_u = img_yuv[..., 1].copy()
img_v = img_yuv[..., 2].copy()

# blur uv
blur = erosion_size
img_u_blur = cv2.blur(img_u, (blur, blur))
img_v_blur = cv2.blur(img_v, (blur, blur))

# shift set
shift = erosion_size

# to left
# img_v_blur[:, :-shift] = img_v_blur[:, shift:]
# img_u_blur[:, :-shift] = img_u_blur[:, shift:]

# to right
img_u_blur[:, shift:] = img_u_blur[:, :-shift]
img_v_blur[:, shift:] = img_v_blur[:, :-shift]

# up
# img_u_blur[:-shift, :] = img_u_blur[shift:, :]
# img_v_blur[:-shift, :] = img_v_blur[shift:, :]

# down
# img_u_blur[shift:, :] = img_u_blur[:-shift, :]
# img_v_blur[shift:, :] = img_v_blur[:-shift, :]

# blur mask
blur_mask = erosion_size
mask_blurred = cv2.blur(masked.astype(float), (blur_mask, blur_mask))

#border[:, :] = True

# linear combination using blurred mask
img_comp = img_yuv.astype(float)
img_comp[border, 1] = np.clip(img_u_blur[border] * mask_blurred[border] + (1.0 - mask_blurred[border]) * img_yuv[border, 1], 0, 255)
img_comp[border, 2] = np.clip(img_v_blur[border] * mask_blurred[border] + (1.0 - mask_blurred[border]) * img_yuv[border, 2], 0, 255)
img_comp = img_comp.astype(np.uint8)

# img_comp = img_yuv.copy()
# img_comp[border, 1] = img_u_blur[border]
# img_comp[border, 2] = img_v_blur[border]

img_comp_bgr = cv2.cvtColor(img_comp, cv2.COLOR_YUV2BGR)

cv2.imwrite('out1.png', img[:, :, :3])

cv2.imwrite(f'out2_{erosion_size}.png', img_comp_bgr)


plt.matshow(border)
plt.matshow(img_yuv[:, :, 1])
plt.matshow(img_u_blur)

#img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#plt.matshow(img_hsv[:, :, 0])
#plt.matshow(img_hsv[:, :, 1])
#plt.matshow(img_hsv[:, :, 2])


#plt.show()
