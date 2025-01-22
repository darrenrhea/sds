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


def prepare_ads():
    ads = [
        dict(
            png_path= "~/awecom/data/ads/cokelogo.png",
            x_center=37.6,
            y_center=13.2,
            width=8,
            height=5
        ),
        dict(
            png_path= "~/awecom/data/ads/cokelogo.png",
            x_center=-37.6,
            y_center=13.2,
            width=8,
            height=5
        ),
        dict(
            png_path= "~/awecom/data/ads/black_cokelogo.png",
            x_center=-37.6,
            y_center=-11.2,
            width=8,
            height=5
        ),
        dict(
            png_path= "~/awecom/data/ads/red_cokelogo.png",
            x_center=37.6,
            y_center=13.2,
            width=8,
            height=5
        )
    ]

    prepared_ads = []
    for ad in ads:
        png_path = Path(ad["png_path"]).expanduser()
        ad_pil = PIL.Image.open(str(png_path)).convert("RGBA")
        ad_rgba_np_float32 = np.array(ad_pil).astype(np.float32)
        x_center = ad["x_center"]
        y_center = ad["y_center"]
        width = ad["width"]
        height = ad["height"]
        prepared_ads.append(
            dict(
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                ad_rgba_np_float32=ad_rgba_np_float32
            )
        )
   
    return prepared_ads

