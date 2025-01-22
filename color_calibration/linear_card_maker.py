from write_rgb_hwc_np_u8_to_png import (
     write_rgb_hwc_np_u8_to_png
)
from pathlib import Path
import shutil
import numpy as np
from prii import prii

def linear_card_maker():
    """
    Use EasyRes to select a non-retina resolution,
    or better yet, use a monitor on linux without scaling.
    """
    # three columns, the middle one if alternating rows
    out_dir = out_path = Path("~/cards").expanduser()
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=False, exist_ok=False)
    width = 1920
    height = 1200
    
    alt_rows_rgb_hwc_np_u8 = np.zeros(
        shape=(
            height,
            width // 3,
            3
        ),
        dtype=np.uint8
    )

    alt_rows_rgb_hwc_np_u8[::2, :, :] = [255, 255, 255]

    for c in range(180, 190 + 1):
        const_color_rgb_hwc_np_u8 = np.zeros(
            shape=(
                height,
                width // 3,
                3
            ),
            dtype=np.uint8
        )
        const_color_rgb_hwc_np_u8[:, :, :] = [c, c, c]

        stack = np.concatenate(
            (alt_rows_rgb_hwc_np_u8, const_color_rgb_hwc_np_u8, alt_rows_rgb_hwc_np_u8),
            axis=1
        )
        print(f"{c=}:")
        out_path = Path(f"~/cards/{c}.png").expanduser()
        prii(stack, out=out_path)

    print("ffb ~/cards")

    x = 186 / 255
    y = x ** 2.2 
    print(f"{x=}, {y=}")


def average_two_colors_correctly():
    """
    This shows how to map colors to a "linear space",
    where operations like addition and averaging work correctly.
    Proven by physical averaging of alternating rows of two colors
    viewed from far away.
    """
    # three columns, the middle one if alternating rows
    out_dir = out_path = Path("~/cards").expanduser()
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=False, exist_ok=False)
    alt_rows_rgb_hwc_np_u8 = np.zeros((1080, 1920//3, 3), dtype=np.uint8)

    r, g, b = np.random.randint(0, 256, 3)
    R, G, B = np.random.randint(0, 256, 3)

    alt_rows_rgb_hwc_np_u8[1::2, :, :] = [r, g, b]
    alt_rows_rgb_hwc_np_u8[0::2, :, :] = [R, G, B]

    # get r, g, b into the interval [0, 1], but still nonlinear:
    r_float_nonlinear = r / 255
    g_float_nonlinear = g / 255
    b_float_nonlinear = b / 255

     # get r, g, b into the interval [0, 1], but still nonlinear:
    R_float_nonlinear = R / 255
    G_float_nonlinear = G / 255
    B_float_nonlinear = B / 255
    
    # now get to the linear space:
    r_linear = r_float_nonlinear ** 2.2
    g_linear = g_float_nonlinear ** 2.2
    b_linear = b_float_nonlinear ** 2.2

     # now get to the linear space:
    R_linear = R_float_nonlinear ** 2.2
    G_linear = G_float_nonlinear ** 2.2
    B_linear = B_float_nonlinear ** 2.2

    avg_red_linear = (r_linear + R_linear) / 2
    avg_green_linear = (g_linear + G_linear) / 2
    avg_blue_linear = (b_linear + B_linear) / 2

    # now back to the nonlinear space:
    avg_red_nonlinear = avg_red_linear ** (1/2.2)
    avg_green_nonlinear = avg_green_linear ** (1/2.2)
    avg_blue_nonlinear = avg_blue_linear ** (1/2.2)

    # now back to the [0, 255] interval, integers:
    avg_r = int(np.round(avg_red_nonlinear * 255))
    avg_g = int(np.round(avg_green_nonlinear * 255))
    avg_b = int(np.round(avg_blue_nonlinear * 255))

    const_color_rgb_hwc_np_u8 = np.zeros((1080, 1920//3, 3), dtype=np.uint8)
    const_color_rgb_hwc_np_u8[:, :, :] = [avg_r, avg_g, avg_b]

    stack = np.concatenate(
        (alt_rows_rgb_hwc_np_u8, const_color_rgb_hwc_np_u8, alt_rows_rgb_hwc_np_u8),
        axis=1
    )
    

    print(f"{r=}, {g=}, {b=}, mixed with {R=}, {G=}, {B=} is {avg_r=}, {avg_g=}, {avg_b=}")

    out_path = Path("temp.png").resolve()
    write_rgb_hwc_np_u8_to_png(
        rgb_hwc_np_u8=stack,
        out_abs_file_path=out_path,
    )

    print("ffb temp.png")


   

if __name__ == "__main__":
    # linear_card_maker()
    average_two_colors_correctly()
    # average_two_colors_correctly(0, 255)