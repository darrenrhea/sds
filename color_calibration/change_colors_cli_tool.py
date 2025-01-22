from write_rgb_np_u8_to_png import (
     write_rgb_np_u8_to_png
)
import argparse

import numpy as np
from quantize_colors_via_kmeans import (
     quantize_colors_via_kmeans
)
from pathlib import Path
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from prii import (
     prii
)

def print_color_sample_in_terminal(rgb):
    assert (
        (isinstance(rgb, np.ndarray) and rgb.shape == (3,) and rgb.dtype == np.uint8)
        or
        (isinstance(rgb, list) and len(rgb) == 3 and all(isinstance(x, int) for x in rgb) and all(0 <= x <= 255 for x in rgb))
        or
        (isinstance(rgb, tuple) and len(rgb) == 3 and all(isinstance(x, int) for x in rgb) and all(0 <= x <= 255 for x in rgb))
    ), "rgb must be a list, tuple, or np.ndarray of 3 ints in the range 0-255"

    sample = np.zeros((1, 1, 3), dtype=np.uint8)
    sample[0, 0, 0] = rgb[0]
    sample[0, 0, 1] = rgb[1]
    sample[0, 0, 2] = rgb[2]
    bigger = sample.repeat(200, axis=0).repeat(200, axis=1)
    
    prii(bigger)


def change_colors_cli_tool():
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "image_path", type=str, help="path to the image"
    )
    argp.add_argument(
        "num_colors", type=int, help="how_many_colors_to_quantize_to"
    )

    opt = argp.parse_args()
    original_path = Path(opt.image_path).resolve()
    num_colors_to_quantize_to = opt.num_colors

    assert num_colors_to_quantize_to > 0, "num_colors must be greater than 0"

    original_rgb_np_u8 = open_as_rgb_hwc_np_u8(
        image_path=original_path
    )

    prii(
        original_rgb_np_u8,
        caption="this is what we started with:",  
    )

    rgb_values = original_rgb_np_u8.reshape(-1, 3)

    cluster_indices, centroids_np_u8 = quantize_colors_via_kmeans(
        rgb_values=rgb_values,
        num_colors_to_quantize_to=num_colors_to_quantize_to
    )

    quantized = centroids_np_u8[cluster_indices].reshape(original_rgb_np_u8.shape)

    prii(
        quantized,
        caption="this is quantized",
    )
    print(f"Into these {num_colors_to_quantize_to} colors:")
    for i, centroid in enumerate(centroids_np_u8):
        print(f"the {i}-ith color is:")
        print_color_sample_in_terminal(centroid)

    print(
        "\n\n\n\n\nNow let us map those colors, each in turn, to new colors that better matching the original footage."
    )
    new_centroids_np_u8 = centroids_np_u8.copy()

    for centroid_index, centroid in enumerate(centroids_np_u8):
        print(f"considering the {centroid_index}-ith color:")
        print_color_sample_in_terminal(centroid)
        ans = input("What rgb do you want to map it to?\nEnter 3 uint8s separated by commas, or two color indices separated by commas, or press enter to skip:")
        if len(ans.split(",")) == 3:
            new_rgb = np.array([int(x) for x in ans.split(",")], dtype=np.uint8)
        elif len(ans.split(",")) == 2:
            print("you entered 2 values, so I will average the specified new colors")
            index_a = int(ans.split(",")[0])
            index_b = int(ans.split(",")[1])
            color_a = new_centroids_np_u8[index_a]
            color_b = new_centroids_np_u8[index_b]
            print_color_sample_in_terminal(color_a)
            print_color_sample_in_terminal(color_b)
            print("to get the average of these two colors:")
            new_rgb = (new_centroids_np_u8[index_a] + new_centroids_np_u8[index_b]) // 2
        else:
            new_centroids_np_u8[centroid_index, :] = new_rgb
            print("skipping the rest of the colors, leaving them as is")
            break
    
        print(f"mapping {centroid} to {new_rgb}")
        print_color_sample_in_terminal(centroid)
        print("to:")
        print_color_sample_in_terminal(new_rgb)
        
        new_centroids_np_u8[centroid_index, :] = new_rgb

        new_quantized = new_centroids_np_u8[cluster_indices].reshape(original_rgb_np_u8.shape)

        prii(
            new_quantized,
            caption="So far we have:",
        )
    
    write_rgb_np_u8_to_png(
        rgb_hwc_np_u8=new_quantized,
        out_abs_file_path=Path("temp.png").resolve()
    )

    prii(
        "temp.png",
        caption="this is the final result",
    )


    

   
    
