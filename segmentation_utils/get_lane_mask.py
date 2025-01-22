"""
Given a well-fit camera, determine the 3D location of select points on the floor.
"""

import better_json as bj
from nuke_lens_distortion import get_floor_point_world_coordinates_of_pixel_coordinates
import sys
from CameraParameters import CameraParameters
import numpy as np
from PIL import Image

if __name__ == "__main__":

    img_pil = Image.open(str(sys.argv[1])).convert("RGBA")
    hwc_np_uint8 = np.array(img_pil)

    height = 1080
    width = 1920

    camera_parameters_dict = bj.load(sys.argv[2])
    camera_parameters = CameraParameters.from_dict(camera_parameters_dict)

    # mask = np.zeros((height, width, channel)).astype(np.uint8)
    mask = hwc_np_uint8
    for i in range(height):
        print(i, end='\r')
        for j in range(width):
            x_wc, y_wc = get_floor_point_world_coordinates_of_pixel_coordinates(
                x_pixel=j, 
                y_pixel=i,
                camera_parameters=camera_parameters,
                photograph_width_in_pixels=width,
                photograph_height_in_pixels=height
            )
            MASK = 0
            if (y_wc > -7.83) and (y_wc < 7.83):
                if (( np.abs(x_wc) > 28.17) and ( np.abs(x_wc) < 47)):
                    MASK = 255
            mask[i, j, -1] = MASK

    img = Image.fromarray(mask).convert("RGBA")
    img.save(sys.argv[3])
