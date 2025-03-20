from print_data_image_in_iterm2 import print_data_image_in_iterm2
from get_to_image_pil_from_one_of_these import get_to_image_pil_from_one_of_these
from pathlib import Path
import io
import PIL.Image
from typing import Optional

from ptse import ptse


def get_data_of_png_of_image_pil(
    image_pil: PIL.Image.Image,
    scale: float = 1.0,
) -> bytes:
    """
    Write the image_pil to png but in memory / RAM.
    """
    fp = io.BytesIO()
    if scale == 1.0:
        image_pil.save(fp, format="PNG")
    else:
        resized_pil = image_pil.resize(
            size=(
                round(image_pil.width * scale),
                round(image_pil.height * scale),
            ),
            resample=PIL.Image.BILINEAR
        )
        resized_pil.save(fp, format="PNG")
    
    data = fp.getvalue()
    assert isinstance(data, bytes)
    return data


def get_data_of_jpg_of_image_pil(
    image_pil: PIL.Image.Image,
    scale: float = 1.0,
) -> bytes:
    """
    Write the image_pil to JPEG but in memory / RAM.
    """
    fp = io.BytesIO()
    if scale == 1.0:
        resized_pil = image_pil
    else:
        resized_pil = image_pil.resize(
            size=(
                round(image_pil.width * scale),
                round(image_pil.height * scale),
            ),
            resample=PIL.Image.BILINEAR
        )

    if resized_pil.mode == "I;16":
        resized_pil.save(fp, format="PNG")
    else:
        resized_pil.save(fp, format="JPEG")
    
    data = fp.getvalue()
    assert isinstance(data, bytes)
    return data


# print_image_in_iterm2 filename inline base64contents print_filename
#   filename: Filename to convey to client
#   inline: 0 or 1
#   base64contents: Base64-encoded contents
#   print_filename: If non-empty, print the filename
#                   before outputting the image
def print_image_in_iterm2(
    data: Optional[bytes] = None,
    image_path=None,
    grayscale_np_uint8=None,
    grayscale_np_u16=None,
    binary_np_uint8=None,
    rgb_np_uint8=None,
    rgb_chw_np_uint8=None,
    rgba_chw_np_uint8=None,
    rgba_np_uint8=None,
    bgra_np_u8=None,
    bgr_np_u8=None,
    image_pil=None,
    image_file_name=None,
    scale: float = 1.0,
    out: Optional[Path] = None,
    show: bool = True,
    title=None,
):
    # get to image_pil
    if data:
        pass
    elif image_path is not None or image_file_name is not None:  # if we are basically given a file path
        if image_file_name:
            assert (
                isinstance(image_file_name, str)
            ), f"In the case that you give an image_file_name, it must be a str, but it was {type(image_file_name)}"
            image_path = Path(image_file_name).expanduser().resolve()

        else:
            assert image_path is not None
            assert isinstance(image_path, Path), f"image_path must be a Path but it is of type {type(image_path)}"
        # at this point we have an image_path regardless of which of image_path or image_file_name was passed in
        if title is None:
            title = image_path.name

        with open(image_path, "rb") as fp:
            data = fp.read()
    else:
        image_pil, num_channels = get_to_image_pil_from_one_of_these(
            grayscale_np_uint8=grayscale_np_uint8,
            binary_np_uint8=binary_np_uint8,
            rgb_np_uint8=rgb_np_uint8,
            rgb_chw_np_uint8=rgb_chw_np_uint8,
            rgba_chw_np_uint8=rgba_chw_np_uint8,
            rgba_np_uint8=rgba_np_uint8,
            image_pil=image_pil,
            bgra_np_u8=bgra_np_u8,
            bgr_np_u8=bgr_np_u8,
        )
        if out is not None:
            assert isinstance(out, Path), f"out, if not None, must be a Path but it is of type {type(out)} and value {out}"
            assert out.parent.is_dir(), f"ERROR: {out.parent=} must be an extant directory but it isn't."
            image_pil.save(fp=out)
        
        if num_channels == 4 or image_pil.mode == "I;16":
            data = get_data_of_png_of_image_pil(image_pil=image_pil, scale=scale)
        else:
            data = get_data_of_jpg_of_image_pil(image_pil=image_pil, scale=scale)
    
    if title is None:
        title = "image.png"
            
    assert isinstance(title, str)
    if show:
        ptse("")
        print_data_image_in_iterm2(
            data=data,
            title=title
        )
   
