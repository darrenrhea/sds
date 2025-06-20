from prii import (
     prii
)
from get_font_file_path import (
     get_font_file_path
)
from pathlib import Path
import numpy as np
import PIL
from PIL import ImageDraw, ImageFont

font_path = get_font_file_path()
font = ImageFont.truetype(str(font_path), 16)


class Drawable2DImage(object):
    """
    Possibly starting with a photograph or just starting out blank,
    makes an drawable 2D image 
    that you can draw 2D lines and curves onto and annotate points with text.
    Oftentimes the original photograph / rasterized image does not
    have enough resolution to draw complex text labels onto it, 
    so we can enlargen the image by an integer factor expand_by_factor.
    """

    def __init__(
        self,
        rgba_np_uint8,
        expand_by_factor=1
    ):
        assert isinstance(rgba_np_uint8, np.ndarray)
        assert rgba_np_uint8.ndim == 3
        assert rgba_np_uint8.shape[2] in [3, 4], "Must have 3 or 4 channels and be hwc"
        self.original_photograph_height_in_pixels = rgba_np_uint8.shape[0]
        self.original_photograph_width_in_pixels = rgba_np_uint8.shape[1]
        assert expand_by_factor in range(
            1, 100
        ), "expand_by_factor must be a small positive integer"
        self.expand_by_factor = expand_by_factor
      
        original_image_pil = PIL.Image.fromarray(rgba_np_uint8)
        self.image_pil = original_image_pil.resize(
            (
                expand_by_factor * self.original_photograph_width_in_pixels,
                expand_by_factor * self.original_photograph_height_in_pixels,
            )
        )
        self.photograph_width_in_pixels = self.image_pil.width
        self.photograph_height_in_pixels = self.image_pil.height
        self.drawable = ImageDraw.Draw(self.image_pil)

    def prii(self, caption=None):
        rgb_pil = self.image_pil.convert('RGB')
        prii(rgb_pil, caption=caption)


    def save(self, output_image_file_path, format=None):
        output_image_file_path = Path(output_image_file_path).expanduser()
        if format is None:
            if str(output_image_file_path).endswith(".jpg"):
                format = "JPEG"
            elif str(output_image_file_path).endswith(".png"):
                format = "PNG"
            elif str(output_image_file_path).endswith(".bmp"):
                format = "BMP"
            else:
                raise Exception(f"extension on {output_image_file_path} not understood to be JPEG, BMP, nor PNG")
            
        if format == "JPEG":
            rgb = self.image_pil.convert('RGB')
            rgb.save(
                fp=str(output_image_file_path),
                format="JPEG",
                optimize=True,
                quality=95
            )
        elif format == "BMP":
            self.image_pil.save(
                fp=str(output_image_file_path),
                format="BMP"
            )
        elif format == "PNG":
            self.image_pil.save(
                fp=str(output_image_file_path),
                format="PNG",
                optimize=False,
                compress_level=1
            )
        else:
            raise Exception(f"format {format} is unrecognized.  Only JPEG, BMP an PNG are allowed")

    def draw_line_from_point_to_point(self, xy0, xy1, rgb, width=0):
        """
        given the camera_parameters, draw a line from a to b,
        where a and b are 2D positions.
        """
        self.drawable.line(
            (
                xy0[0] * self.expand_by_factor,
                xy0[1] * self.expand_by_factor,
                xy1[0] * self.expand_by_factor,
                xy1[1] * self.expand_by_factor,
            ),
            fill=rgb,
            width=width
        )
        

    def draw_2d_curve(self, t_min, t_max, t_to_xy_function, rgb, width=0, num_steps=100):
        """
        given the camera_parameters, draw a parametric curve
        where the parameter t varies from t_min to t_max.
        """
        for k in range(num_steps):
            u = k / num_steps  # ranges over [0, 1]
            t = (1 - u) * t_min + u * t_max
            x0, y0 = t_to_xy_function(t)
           
            u = (k + 1) / num_steps
            t = (1 - u) * t_min + u * t_max
            x1, y1 = t_to_xy_function(t)
           
           
            self.draw_line_from_point_to_point(
                xy0=(x0, y0),
                xy1=(x1, y1),
                rgb=rgb,
                width=width
            )

    def circle(self, center, radius, rgb, width=0, num_steps=1000):
        def t_to_xy_function(t):
            return center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)
        t_min = 0
        t_max = 2 * np.pi
        self.draw_2d_curve(
            t_min=t_min,
            t_max=t_max,
            t_to_xy_function=t_to_xy_function,
            rgb=rgb,
            width=width,
            num_steps=num_steps
        )

    def draw_fine_point(self, x0, y0, rgb, text):
        """
        Draws a point where it would appear under the perspective projection.
        """
        apx = x0 * self.expand_by_factor
        apy = y0 * self.expand_by_factor

        self.draw_line_from_point_to_point(
            xy0=(apx, apy),
            xy1=(apx, apy),
            rgb=rgb,
            width=0
        )

        if text is not None and text != "":
            self.drawable.text(
                xy=(apx + 15, apy - 10), text=text, font=font, fill=rgb
            )

    def draw_plus(self, x0, y0, rgb, size=3, text=""):
        apx = x0 * self.expand_by_factor
        apy = y0 * self.expand_by_factor
        s = size 
        self.drawable.line((apx-s, apy, apx+s, apy), fill=rgb, width=1)
        self.drawable.line((apx, apy-s, apx, apy+s), fill=rgb, width=1)
        if text is not None and text != "":
            self.drawable.text(
                xy=(apx + 15, apy - 10), text=text, font=font, fill=rgb
            )
       
    def draw_cross_at_2d_point(self, x_pixel, y_pixel, color, size, text):
        expanded_x_pixel = x_pixel * self.expand_by_factor
        expanded_y_pixel = y_pixel * self.expand_by_factor
        s = size
        self.drawable.line(
            (
                expanded_x_pixel - s,
                expanded_y_pixel - s,
                expanded_x_pixel + s,
                expanded_y_pixel + s,
            ),
            fill=color,
            width=1,
        )
        self.drawable.line(
            (
                expanded_x_pixel - s,
                expanded_y_pixel + s,
                expanded_x_pixel + s,
                expanded_y_pixel - s,
            ),
            fill=color,
            width=1,
        )
        if text is not None and text != "":
            self.drawable.text(
                xy=(expanded_x_pixel + 15, expanded_y_pixel - 10), text=text, font=font, fill=color
            )
    
    def draw_plus_at_2d_point(self, x_pixel, y_pixel, rgb, size, text):
        expanded_x_pixel = x_pixel * self.expand_by_factor
        expanded_y_pixel = y_pixel * self.expand_by_factor
        s = size
        self.drawable.line(
            (
                expanded_x_pixel - s,
                expanded_y_pixel,
                expanded_x_pixel + s,
                expanded_y_pixel,
            ),
            fill=rgb,
            width=1,
        )
        self.drawable.line(
            (
                expanded_x_pixel,
                expanded_y_pixel - s,
                expanded_x_pixel,
                expanded_y_pixel + s,
            ),
            fill=rgb,
            width=1,
        )
        if text is not None and text != "":
            self.drawable.text(
                xy=(expanded_x_pixel + 15, expanded_y_pixel - 10), text=text, font=font, fill=rgb
            )

        

