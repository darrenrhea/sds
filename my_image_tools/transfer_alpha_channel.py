import PIL.Image
import numpy as np
from pathlib import Path
from image_openers import open_alpha_channel_image_as_a_single_channel_grayscale_image

def transfer_alpha_channel(
    alpha_path: Path,
    rgb_path: Path,
    save_path: Path,
):
    """
    Pulls the alpha channel out of a 4 or 1 channel image
    and combines it with the rgb info of another image 
    to create a 4 channel RGBA image.
        """
    print(
        f"Taking the alpha channel from:\n"
        f"{alpha_path}\n"
        f"and the rgb info from:\n"
        f"{rgb_path}\n"
        f"and combining them to form {save_path}\n"
    )

    alpha_np = open_alpha_channel_image_as_a_single_channel_grayscale_image(
        abs_file_path=alpha_path
    )

    rgba_pil = PIL.Image.open(str(rgb_path)).convert(
        "RGBA"
    )  # the source of the rgb information. Even if it doesn't have an alpha channel, make one

    rgba_np = np.array(rgba_pil)
    rgba_np[:, :, 3] = alpha_np
    save_pil = PIL.Image.fromarray(rgba_np)
    save_pil.save(save_path)
