from pathlib import Path
from image_openers import open_alpha_channel_image_as_a_single_channel_grayscale_image
from image_writers import write_grayscale_hw_np_u8_to_png

def extract_alpha_channel(
    alpha_path: Path,
    save_path: Path,
):
    """
    Pulls the alpha channel out of a 4 or 1 channel image
    and saves it as a grayscale image.
    """
    assert isinstance(alpha_path, Path)
    assert isinstance(save_path, Path)
    assert alpha_path.is_file(), f"No such file {alpha_path}"
    assert alpha_path.is_absolute(), f"{alpha_path} is not an absolute path"
    assert save_path.is_absolute(), f"{save_path} is not an absolute path"

    print(
        f"Taking the alpha channel from:\n"
        f"{alpha_path}\n"
        f"and saving as a grayscale image at {save_path}\n"
    )

    alpha_np = open_alpha_channel_image_as_a_single_channel_grayscale_image(
        abs_file_path=alpha_path
    )

    write_grayscale_hw_np_u8_to_png(
        grayscale_hw_np_u8=alpha_np,
        out_abs_file_path=save_path,
        verbose=False
    )

