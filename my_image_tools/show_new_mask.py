import sys
import PIL
import PIL.Image
import numpy as np
from pathlib import Path
from print_image_in_iterm2 import print_image_in_iterm2

def show_mask(frame_index):

    """
    This is good for when the mask is correct in the alpha channel of one png, but
    """
    alpha_path = Path(
        f"~/awecom/data/clips/gsw1/masking_attempts/fastai_8_bw/gsw1_{frame_index:06d}_nonfloor.png"
    ).expanduser()
    
    rgb_path =  Path(
        f"~/awecom/data/clips/gsw1/frames/gsw1_{frame_index:06d}.jpg"
    ).expanduser()

    assert alpha_path.is_file(), f"No such file {alpha_path}"
    assert rgb_path.is_file(), f"No such file {rgb_path}"
    print(
        f"Taking the alpha channel from {alpha_path} and the rgb info from {rgb_path} and combining them to form a colorized mask:"
    )
    alpha_pil = PIL.Image.open(str(alpha_path))  # the source of the good alpha channel
    alpha_np = np.array(alpha_pil)
    if alpha_np.ndim == 2:
        alpha_channel = alpha_np
    elif alpha_np.ndim == 3:
        assert alpha_np.shape[2] == 4, f"{alpha_path} doesn't have an alpha channel!"
        alpha_channel = alpha_np[:, :, 3]
    else:
        raise Exception()
    rgb_pil = PIL.Image.open(str(rgb_path)).convert(
        "RGBA"
    )  # the source of the rgb information. Even if it doesn't have an alpha channel, make one
    rgba_np = np.array(rgb_pil)
    rgba_np[:, :, 3] = alpha_channel
    combined_pil = PIL.Image.fromarray(rgba_np)
    print_image_in_iterm2(
        image_pil=combined_pil,
    )


def main():
    if len(sys.argv) < 2:
        print(
            """
Usage:
    show_mask <canonical_index_of_gsw1>
"""
        )
        sys.exit(1)

    frame_index = int(sys.argv[1])

    show_mask(frame_index=frame_index)

if __name__ == "__main__":
    main()
