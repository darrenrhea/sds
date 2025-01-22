from prii import (
     prii
)
# https://stackoverflow.com/questions/67708224/shapely-polygon-to-binary-mask


import numpy as np
import cv2
from shapely.geometry import box


get_mask_from_polygon(
    
mask = np.zeros(
    shape=[960, 1280],
    dtype=np.uint8
)
# create an example bounding box polygon
x1, y1, x2, y2 = 480, 540, 780, 840
polygon = box(x1, y1, x2, y2)
points = [[x, y] for x, y in zip(*polygon.boundary.coords.xy)]

points_int32 = np.array([points]).astype(np.int32)
print(f"{points_int32.shape=}")
mask = cv2.fillPoly(
    img=mask,
    pts=points_int32,
    color=255
)
print(f"{mask.shape=}")
prii(mask)