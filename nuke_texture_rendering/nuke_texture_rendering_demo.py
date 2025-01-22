"""
Renders a 3D image of the floor texture under a given set of camera parameters.
Not only can it render a full-sized 3d image,
but importantly it can partially render, i.e.
what would a given subrectangle of the full render would look like.
"""
import sys
from pathlib import Path
import numpy as np
import PIL
import PIL.Image
import scipy
from scipy.ndimage import map_coordinates
import requests
import better_json as bj
from stage_data_resources import stage_this_data_resource
import rodrigues_utils
import nuke_texture_rendering
from CameraParameters import CameraParameters

def main():
    color = len(sys.argv) > 1
    if color:
        texture_path = stage_this_data_resource(name="full_resolution_color.png")
        print(texture_path)
        texture_x_min_wc = -56.0
        texture_x_max_wc = 56.0
        texture_y_min_wc = -30.0
        texture_y_max_wc = 30.0
    else:
        texture_x_min_wc = -56.0
        texture_x_max_wc = 56.0
        texture_y_min_wc = -30.0
        texture_y_max_wc = 30.0
        texture_path = stage_this_data_resource(name="full_resolution_green.png")

    camera_parameters_path = stage_this_data_resource(name="stapleslahou1_000070_camera_parameters.json")

    camera_parameters = CameraParameters.from_dict(
        bj.load(camera_parameters_path)
    )
    
    dict(
        rod=[1.78356276, -0.24938955, 0.1912733],
        loc=[0.23953680563463098, -123.8297299087407, 34.47149337103112],
        f=5.377273793213683,
        k1=-0.031969675713926685,
        k2=0.001530790818926706,
    )

    original_image_path = stage_this_data_resource(name="stapleslahou1_000070.png")

    image_pil = PIL.Image.open(str(original_image_path)).convert("RGBA")
    w_image = image_pil.width
    h_image = image_pil.height
    assert (w_image, h_image) == (1920, 1080)
    original_photo_rgba_np_uint8 = np.array(image_pil)
    assert (
        original_photo_rgba_np_uint8.shape[2] == 4
    ), "original_photo_rgba_np_uint8 should be RGBA thus 4 channels"

    texture_pil = PIL.Image.open(str(texture_path)).convert(
        "RGBA"
    )  # texture has transparent bits
    texture_rgba_np_float32 = np.array(texture_pil).astype(np.float32)

    composition_rgba_np_uint8 = nuke_texture_rendering.partial_render(
        photograph_width_in_pixels=1920,
        photograph_height_in_pixels=1080,
        original_photo_rgba_np_uint8=original_photo_rgba_np_uint8,
        camera_parameters=camera_parameters,
        texture_rgba_np_float32=texture_rgba_np_float32,
        texture_x_min_wc=texture_x_min_wc,
        texture_x_max_wc=texture_x_max_wc,
        texture_y_min_wc=texture_y_min_wc,
        texture_y_max_wc=texture_y_max_wc,
        i_min=927,
        i_max=1080,
        j_min=0,
        j_max=400,
    )

    final_pil = PIL.Image.fromarray(composition_rgba_np_uint8)
    final_pil.save("partial_render.png")
    print(f"open partial_render.png")


if __name__ == "__main__":
    main()
