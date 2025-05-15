import sys
import PIL
import PIL.Image
import numpy as np
from pathlib import Path


def flip_alpha_channel(
    original_image_path,
    save_path
):
    """
    Flips the alpha channel.
    Say you want to go from foreground objects opaque, the rest transparent;
    to foreground objects transparent, the rest opaque.
    """
    original_pil = PIL.Image.open(str(original_image_path))
    original_np = np.array(original_pil)
    original_np[:, :, 3] = 255 - original_np[:, :, 3]
    save_pil = PIL.Image.fromarray(original_np)
    save_pil.save(save_path)


def main():
    """
    This is what happens if you run it through the executable flip_alpha_channel
    """
    if len(sys.argv) < 3:
        print("This utility flips the alpha_channel of a rgba png")
        print("Usage:\n\n    flip_alpha_channel <input_rgba.png> <output_rgba.png>\n")
        sys.exit(1)
    original_image_path = Path(sys.argv[1])
    save_path = Path(sys.argv[2])
    assert original_image_path.is_file()
    assert not save_path.is_file()
    flip_alpha_channel(
        original_image_path=original_image_path,
        save_path=save_path
    )

def several():
    """
    This is what happens if you run it through python flip_alpha_channel.py
    """
    mydir = Path("~/awecom/data/clips/den1/masking_attempts/fastai_den1_resnet34_224").expanduser()
    print(mydir)
    for k in range(147000, 651000 + 1, 1000):
        original_image_path = mydir / f"den1_{k:06d}_nonfloor.png"
        save_path = mydir / f"den1_{k:06d}_floor.png"
        flip_alpha_channel(
            original_image_path=original_image_path,
            save_path=save_path
        )


if __name__ == "__main__":
    # main()
    several()
