usage_message = """
Usage:

    cast_judgement <attempt.png> <true_foreground_mask.png> \\
                   <relevance_mask.png> <original_path.png> \\
                   <judgment_png_path> <compare_gif_path> \\
                   <compare_png_path>
                   

If someone makes an attempt to foreground-mask an image,
and we know the actual truth, we should be able to judgement on how well they did,
How many pixels are True Positives? (white)
How many pixels are True Negativess? (black)
How many pixels are False Positives? (red)
How many pixels are False Negativess? (cyan)
Given an actually correct relevance mask,
we should also only count the pixels that are relevant,
and color irrelevant pixels dark green.

Example:

python cast_judgement.py ~/awecom/data/clips/swinney1/masking_attempts/training_on_6/swinney1_004600_nonfloor.png ~/r/swinney1/swinney1_004600_nonfloor.png ~/r/swinney1/swinney1_004600_relevant.png ~/awecom/data/clips/swinney1/frames/swinney1_004600.jpg judgement.png original_vs_judgement.gif original_vs_judgement.png

"""

import sys
import PIL
import PIL.Image
import numpy as np
from pathlib import Path
import math
from open_binary_mask_attempt import open_binary_mask_attempt
from get_binary_mask_from_alpha_channel_of_rgba_png import get_binary_mask_from_alpha_channel_of_rgba_png


def do_it(
    attempt_path,
    actual_truth_path,
    relevance_mask_path,
    original_path,
    judgement_png_path,
    compare_gif_path,
    compare_png_path
):
    images = []
    print(attempt_path)
    print(actual_truth_path)
    print(relevance_mask_path)
    print(original_path)
    attempt_binary = open_binary_mask_attempt(attempt_path)
    actual_binary = get_binary_mask_from_alpha_channel_of_rgba_png(actual_truth_path)
    relevance_binary = get_binary_mask_from_alpha_channel_of_rgba_png(relevance_mask_path)
    irrelevance_binary = np.logical_not(relevance_binary)
    original_pil = PIL.Image.open(original_path)

    false_positive_binary = np.logical_and(
        attempt_binary,
        np.logical_not(actual_binary)
    )

    true_positive_binary = np.logical_and(
        attempt_binary,
        np.logical_and(actual_binary, relevance_binary)      
    )

    false_negative_binary = np.logical_and(
        np.logical_not(attempt_binary),
        np.logical_and(actual_binary, relevance_binary)      
    )

    height = attempt_binary.shape[0]
    width = attempt_binary.shape[1]

    output_np = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    output_np[false_positive_binary, :] = [255, 0, 0]
    output_np[irrelevance_binary, :] = [0, 80, 0]
    output_np[true_positive_binary, :] = [80, 80, 80]
    output_np[false_negative_binary, :] = [0, 255, 255]

    output_pil = PIL.Image.fromarray(output_np)

    images.append(output_pil)
    images.append(original_pil)

    images[0].save(compare_gif_path,
               save_all=True, append_images=images[1:], optimize=False, duration=1000, loop=0)

    output_pil.save(judgement_png_path)

    print(f"see {judgement_png_path}")
    
    out_paths = [original_path, judgement_png_path]

    result = PIL.Image.new("RGB", (math.floor(1920/2), 1080))

    for index, out_path in enumerate(out_paths):
        print(f"out path {out_path}")
        double_img = PIL.Image.open(out_path)
        print(f"size of original image {double_img.size}")
        double_img.thumbnail((1920, 1080/2), PIL.Image.ANTIALIAS)
        w, h = double_img.size
        x = 0
        y = index * math.floor(1080/2)
        print(f"pos {x},{y} size {w},{h}")
        result.paste(double_img, (x, y, x + w, y + h))
    
    result.save(compare_png_path)


def main():
    if len(sys.argv) < 7:
        print(usage_message)
        sys.exit(1)

    attempt_path = Path(sys.argv[1])

    actual_truth_path = Path(sys.argv[2])

    relevance_mask_path = Path(sys.argv[3])

    original_path = Path(sys.argv[4])

    judgement_png_path = Path(sys.argv[5])

    compare_gif_path = Path(sys.argv[6])

    compare_png_path = Path(sys.argv[7])
    
    do_it(
        attempt_path=attempt_path,
        actual_truth_path=actual_truth_path,
        relevance_mask_path=relevance_mask_path,
        original_path=original_path,
        judgement_png_path=judgement_png_path,
        compare_gif_path=compare_gif_path,
        compare_png_path=compare_png_path
    )
    

if __name__ == "__main__":
    main()
