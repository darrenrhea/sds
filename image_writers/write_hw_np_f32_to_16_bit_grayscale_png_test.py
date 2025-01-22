import textwrap
import numpy as np
import PIL.Image
from get_a_temp_file_path import (
     get_a_temp_file_path
)
from write_hw_np_f32_to_16_bit_grayscale_png import (
     write_hw_np_f32_to_16_bit_grayscale_png
)


def test_write_hw_np_f32_to_16_bit_grayscale_png_1():
    ys = np.linspace(-1.0, 1.0, 1080)
    xs = np.linspace(-1.0, 1.0, 1920)
    y, x = np.meshgrid(ys, xs, indexing="ij")
    assert x.shape == (1080, 1920)
    assert y.shape == (1080, 1920)

    hw_np_f32 = (x+y + 1).astype(np.float32)

    out_abs_file_path = get_a_temp_file_path(
        suffix=".png"
    )

    print(
        textwrap.dedent(
            """\
            You better see a grayscale image in your terminal
            which is black in the upper left quarter triangle, then
            gradients to white in the lower right triangle:
            """
        )
    )

    write_hw_np_f32_to_16_bit_grayscale_png(
        hw_np_f32=hw_np_f32,
        out_abs_file_path=out_abs_file_path,
        display_image_in_iterm2=True,
    )

    img = PIL.Image.open(out_abs_file_path)

    # Get the mode and bit depth per channel
    mode = img.mode
    bit_depth_per_channel = img.getextrema()[1].bit_length()
    # TODO: assert the image is 16-bit PNG according to exiftool.
    assert mode == "I"
    assert bit_depth_per_channel == 16


if __name__ == "__main__":
    test_write_hw_np_f32_to_16_bit_grayscale_png_1()