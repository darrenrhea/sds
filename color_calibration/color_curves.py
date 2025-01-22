import cv2
import numpy as np
import sys

from pathlib import Path

from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt

from print_image_in_iterm2 import print_image_in_iterm2
import PIL.Image



def get_average_color_in_box(rgb_hwc_np_uint8, rectangle):
    x_min = rectangle["x_min"]
    x_max = rectangle["x_max"]
    y_min = rectangle["y_min"]
    y_max = rectangle["y_max"]
    crop = rgb_hwc_np_uint8[y_min:y_max, x_min:x_max, :]

    print(
        f"The average of rectangle [{x_min}, {x_max}] x [{y_min}, {y_max}]:"
    )
    print_image_in_iterm2(rgb_np_uint8=crop)
    r_avg = int(np.round(np.mean(crop[..., 0].astype(np.float32))))
    g_avg = int(np.round(np.mean(crop[..., 1].astype(np.float32))))
    b_avg = int(np.round(np.mean(crop[..., 2].astype(np.float32))))

    print(f"is r={r_avg}, g={g_avg}, b={b_avg}\n\n")
    rgb_tuple = (r_avg, g_avg, b_avg)

    return rgb_tuple




def color_correct(
    image_file_path,
    red_in,
    red_out,
    green_in,
    green_out,
    blue_in,
    blue_out
):
    red_curve = CubicSpline(x=red_in, y=red_out, bc_type="natural")
    green_curve = CubicSpline(x=green_in, y=green_out, bc_type="natural")
    blue_curve = CubicSpline(x=blue_in, y=blue_out, bc_type="natural")

    zero_to_255 = np.arange(0, 256)

    red_lut = red_curve(zero_to_255).astype(np.uint8)
    blue_lut = blue_curve(zero_to_255).astype(np.uint8)
    green_lut = green_curve(zero_to_255).astype(np.uint8)

    fig, axs = plt.subplots(nrows=3, figsize=(3, 8))

    axs[0].set_xlim(0, 255)
    axs[0].set_ylim(0, 255)
    axs[0].set_aspect('equal', 'box')
    axs[0].plot(red_in, red_out, 'ro', label='data')
    axs[0].plot(zero_to_255, red_curve(zero_to_255), 'r-', label='red')

    axs[1].set_xlim(0, 255)
    axs[1].set_ylim(0, 255)
    axs[1].set_aspect('equal', 'box')
    axs[1].plot(green_in, green_out, 'go', label='data')
    axs[1].plot(zero_to_255, green_curve(zero_to_255), 'g-', label='green')

    axs[2].set_xlim(0, 255)
    axs[2].set_ylim(0, 255)
    axs[2].set_aspect('equal', 'box')
    axs[2].plot(blue_in, blue_out, 'bo', label='data')
    axs[2].plot(zero_to_255, blue_curve(zero_to_255), 'b-', label='blue')

    fig.tight_layout()

    plt.show(block=False)
    plt.pause(interval=1)

    image = cv2.imread(str(image_file_path))

    transformed_image = np.zeros_like(image)

    print(image.shape)
    red_lut = np.round(red_curve(np.arange(0, 256))).astype(np.uint8)
    green_lut = np.round(green_curve(np.arange(0, 256))).astype(np.uint8)
    blue_lut = np.round(blue_curve(np.arange(0, 256))).astype(np.uint8)


    transformed_image[..., 0] = cv2.LUT(src=image[..., 0], lut=blue_lut)
    transformed_image[..., 1] = cv2.LUT(src=image[..., 1], lut=green_lut)
    transformed_image[..., 2] = cv2.LUT(src=image[..., 2], lut=red_lut)

    cv2.imwrite('22-23_OKC_CORE_transformed.png', transformed_image)


def main():
    pass

if __name__ == "__main__":


    # an image whose colors aren't quite right:
    bad_colors_image_file_path = Path("22-23_OKC_CORE_floortexture.png")

    # an image that has the rgb triplets we desire:
    good_colors_image_file_path = Path("OKCvTOR_11-11-2022_PGM_BAL_147000.jpg")

    good_colors_sample_rects = dict(
        lines=dict(
            x_min = 319,
            y_min = 709,
            x_max = 335,
            y_max = 719,
        ),
        paint=dict(
            x_min = 1670,
            y_min = 591,
            x_max = 1718,
            y_max = 630,
        ),
        wood=dict(
            x_min = 400,
            x_max = 600,
            y_min = 800,
            y_max = 900,
        ),
    )

    bad_colors_sample_rects = dict(
        lines=dict(
            x_min = 2663,
            y_min = 22,
            x_max = 2680,
            y_max = 128,
        ),
        paint=dict(
            x_min = 3125,
            y_min = 780,
            x_max = 3270,
            y_max = 861,
        ),
        
        wood=dict(
            x_min = 1963,
            y_min = 545,
            x_max = 2507,
            y_max = 777,
        ),
        
    )

    domain_color_names = sorted([k for k in bad_colors_sample_rects])
    range_color_names = sorted([k for k in good_colors_sample_rects])
    assert domain_color_names == range_color_names
    color_names = domain_color_names
        
    

    domain_rgb_np_uint8 = np.array(
        PIL.Image.open(str(bad_colors_image_file_path))
    )

    range_rgb_np_uint8 = np.array(
        PIL.Image.open(str(good_colors_image_file_path))
    )

    red_in_out = []
    green_in_out = []
    blue_in_out = []

    for color_name in color_names:
        print(f"As far as correcting {color_name}:")
        domain_rectangle = bad_colors_sample_rects[color_name]
        range_rectangle = good_colors_sample_rects[color_name]

        domain_rgb = get_average_color_in_box(
            domain_rgb_np_uint8,
            domain_rectangle
        )
        print("This would need to map to:")

        range_rgb = get_average_color_in_box(
            range_rgb_np_uint8,
            range_rectangle
        )

        red_in_out.append(
            (
                domain_rgb[0],
                range_rgb[0],
            )
        )

        green_in_out.append(
            (
                domain_rgb[1],
                range_rgb[1],
            )
        )

        blue_in_out.append(
            (
                domain_rgb[2],
                range_rgb[2],
            )
        )
    
    red_in_out = sorted(red_in_out, key=lambda x: x[0])
    green_in_out = sorted(green_in_out, key=lambda x: x[0])
    blue_in_out = sorted(blue_in_out, key=lambda x: x[0])

    print("red_in_out:")
    print(red_in_out)

    print("green_in_out:")
    print(green_in_out)

    print("blue_in_out:")
    print(blue_in_out)

    blue_in =  [0, 206, 232, 255]
    blue_out = [0, 145, 159, 228]

    green_in = [0, 130, 231, 255]
    green_out = [0, 93, 171, 221]

    red_in = [0, 255]
    red_out = [40, 184]

    image_file_path = Path(
        '22-23_OKC_CORE_floortexture.png'
    )
    color_correct(
        image_file_path=image_file_path,
        red_in=red_in,
        red_out=red_out,
        green_in=green_in,
        green_out=green_out,
        blue_in=blue_in,
        blue_out=blue_out,
    )


   

    