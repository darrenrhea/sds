"""
Renders a 3D image of the floor texture under a given set of camera parameters.
Not only can it render a full-sized 3d image,
but importantly it can partially render, i.e.
what would a given subrectangle of the full render would look like.
"""
from pathlib import Path
import numpy as np
import PIL
import PIL.Image
from homography_utils import (
    somehow_get_homography_from_video_frame_to_world_floor
)

from homography_texture_rendering import homography_ad_insertion


def main():
    frame_index = 4900
    homography_from_photo_to_world_floor = somehow_get_homography_from_video_frame_to_world_floor(
        frame_index=frame_index
    )

    texture_path = Path(
        "~/r/floor_textures/ncaa_kansas_floor_texture.png"
    ).expanduser()


    texture_pil = PIL.Image.open(str(texture_path)).convert(
        "RGBA"
    )  # texture has transparent bits
    texture_x_min_wc = -54.0
    texture_x_max_wc = 54.0
    texture_y_min_wc = -28.5
    texture_y_max_wc = 28.5


    original_image_path = Path(
        f"~/awecom/data/clips/swinney1/frames/swinney1_{frame_index:06d}.jpg"
    ).expanduser()

    image_pil = PIL.Image.open(str(original_image_path)).convert("RGBA")
    w_image = image_pil.width
    h_image = image_pil.height
    assert (w_image, h_image) == (1920, 1080)
    original_photo_rgba_np_uint8 = np.array(image_pil)
    assert (
        original_photo_rgba_np_uint8.shape[2] == 4
    ), "original_photo_rgba_np_uint8 should be RGBA thus 4 channels"
   
    texture_rgba_np_float32 = np.array(texture_pil).astype(np.float32)

    composition_rgba_np_uint8 = homography_ad_insertion(
        photograph_width_in_pixels=1920,
        photograph_height_in_pixels=1080,
        original_photo_rgba_np_uint8=original_photo_rgba_np_uint8,
        homography_from_photo_to_world_floor=homography_from_photo_to_world_floor,
        texture_rgba_np_float32=texture_rgba_np_float32,
        texture_x_min_wc=texture_x_min_wc,
        texture_x_max_wc=texture_x_max_wc,
        texture_y_min_wc=texture_y_min_wc,
        texture_y_max_wc=texture_y_max_wc
    )

    final_pil = PIL.Image.fromarray(composition_rgba_np_uint8)
    final_pil.save("floor_insertion.png")
    print(f"open floor_insertion.png")

    ad_path = Path(
        "~/awecom/data/ads/red_cokelogo.png"
    ).expanduser()


    ad_pil = PIL.Image.open(str(ad_path)).convert(
        "RGBA"
    )  # texture has transparent bits
    ad_rgba_np_float32 = np.array(ad_pil).astype(np.float32)
    x_center = 37.6
    y_center = 13.2
    width = 8
    height = 5

    insertion_rgba_np_uint8 = homography_ad_insertion(
        photograph_width_in_pixels=1920,
        photograph_height_in_pixels=1080,
        original_photo_rgba_np_uint8=original_photo_rgba_np_uint8,
        homography_from_photo_to_world_floor=homography_from_photo_to_world_floor,
        texture_rgba_np_float32=ad_rgba_np_float32,
        texture_x_min_wc=x_center - width/2,
        texture_x_max_wc=x_center + width/2,
        texture_y_min_wc=y_center - height/2,
        texture_y_max_wc=y_center + height/2
    )

    final_pil = PIL.Image.fromarray(insertion_rgba_np_uint8)
    final_pil.save("ad_insertion.png")
    print(f"open ad_insertion.png")



if __name__ == "__main__":
    main()
