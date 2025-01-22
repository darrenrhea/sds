import PIL.Image
from pathlib import Path
import numpy as np
from print_image_in_iterm2 import print_image_in_iterm2
from typing import Union, Optional


def prii(
    x: Union[str, Path, PIL.Image.Image, bytes, np.array],
    hint: Optional[str] = None,
    out: Optional[Path] = None,
    show: bool = True,
    scale: float = 1.0,
    caption=None  # if you want to say what the image is
):
    """
    TODO: Very repetitive.  Just call print_image_in_term2 once at the end.

    Print the image x regardless of its type.
    You may need to give it a hint like "bgr" if the
    color channels are in bgr order,
    or "binary" if the image is only 0s and 1s (true binary)
    versus 0 xor 255, for which grayscale works fine.
    TODO: torch.Tensor support, outside of dtype u8 such as float support.
    """
    if caption is not None:
        print(f"{caption}")
    if isinstance(x, str):
        print_image_in_iterm2(image_file_name=x, scale=scale, out=out, show=show)
        return
    
    if isinstance(x, Path):
        print_image_in_iterm2(image_path=x, scale=scale, out=out, show=show)
        return
    
    if isinstance(x, PIL.Image.Image):
        print_image_in_iterm2(image_pil=x, scale=scale, out=out, show=show)
        return

    if isinstance(x, bytes):
        print_image_in_iterm2(data=x, scale=scale, out=out, show=show)
        return

   
    

    if isinstance(x, np.ndarray):
        if x.ndim == 3:
            if hint is None:
                channels = "rgb"  # by default we assume that the image is in rgb or rgba
            else:
                assert isinstance(hint, str)
                if "rgb" in hint:
                    channels = "rgb"
                elif "bgr" in hint:
                    channels = "bgr"
                else:
                    assert False, f"ERROR: How is {hint=} for a 3 dimensional image?"
            assert channels in ["rgb", "bgr"], "colorspace should be determined by now?!"

            if hint is not None and "chw" in hint:
                channel_index = 0
            if hint is not None and "hwc" in hint:
                channel_index = 2
            elif x.shape[2] <= 4:
                channel_index = 2
            elif x.shape[0] <= 4:
                channel_index = 0
            else:
                raise ValueError(f"We don't understand which index to use for channel because {x.shape=}")
            has_alpha_channel = x.shape[channel_index] == 4

            if channels == "rgb":
                if channel_index == 0 and not has_alpha_channel:
                    print_image_in_iterm2(rgb_chw_np_uint8=x, scale=scale, out=out, show=show)
                elif channel_index == 0 and has_alpha_channel:
                    print_image_in_iterm2(rgba_chw_np_uint8=x, scale=scale, out=out, show=show)
                elif channel_index == 2 and not has_alpha_channel:
                    print_image_in_iterm2(rgb_np_uint8=x, scale=scale, out=out, show=show)
                elif channel_index == 2 and has_alpha_channel:
                    print_image_in_iterm2(rgba_np_uint8=x, scale=scale, out=out, show=show)
            elif channels == "bgr":
                if channel_index == 0 and not has_alpha_channel:
                    raise NotImplementedError("bgr with chw not implemented yet")
                elif channel_index == 0 and has_alpha_channel:
                    raise NotImplementedError("bgra with chw not implemented yet")
                elif channel_index == 2 and not has_alpha_channel:
                    print_image_in_iterm2(bgr_np_u8=x, scale=scale, out=out, show=show)
                elif channel_index == 2 and has_alpha_channel:
                    print_image_in_iterm2(bgra_np_u8=x, scale=scale, out=out, show=show)

        elif x.ndim == 2:
            if hint is not None and "binary" in hint:
                print_image_in_iterm2(binary_np_uint8=x, scale=scale, out=out, show=show)
            else:
                print_image_in_iterm2(grayscale_np_uint8=x, scale=scale, out=out, show=show)
        else:
            raise ValueError(f"x.ndim must be 2 or 3 but it is {x.ndim=}")
        
