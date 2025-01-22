"""
One-off invokation of nuke_texture_rendering.partial_render for  debugging.
"""
from pathlib import Path
import numpy as np
import PIL
import PIL.Image
import nuke_texture_rendering
from CameraParameters import CameraParameters

def main():
   
    texture_x_min_wc = -54.0
    texture_x_max_wc = 54.0
    texture_y_min_wc = -28.5
    texture_y_max_wc = 28.5

    camera_parameters = CameraParameters(
        rod=[1.70731, -0.03875, 0.03948],
        loc=[-3.69597, -82.56610, 16.78545],
        f=2.7191,
        ppi=-0.00000,
        ppj=0.00000,
        k1=0.03485,
        k2=-0.00000,
        k3=0.0,
        p1=0.022602688612478473,
        p2=-0.04138072973110877
    )
    
    texture_path = Path("~/r/floor_textures/ncaa_kansas_floor_texture.png").expanduser()

    texture_pil = PIL.Image.open(str(texture_path)).convert(
        "RGBA"
    )  # texture has transparent bits
    texture_rgba_np_float32 = np.array(texture_pil).astype(np.float32)

    composition_rgba_np_uint8 = nuke_texture_rendering.partial_render(
        photograph_width_in_pixels=1920,
        photograph_height_in_pixels=1080,
        original_photo_rgba_np_uint8=None,
        camera_parameters=camera_parameters,
        texture_rgba_np_float32=texture_rgba_np_float32,
        texture_x_min_wc=texture_x_min_wc,
        texture_x_max_wc=texture_x_max_wc,
        texture_y_min_wc=texture_y_min_wc,
        texture_y_max_wc=texture_y_max_wc,
        i_min=988,
        i_max=1009,
        j_min=69,
        j_max=90
    )

    final_pil = PIL.Image.fromarray(composition_rgba_np_uint8)
    final_pil.save("partial_render.png")
    print(f"open partial_render.png")


if __name__ == "__main__":
    main()
