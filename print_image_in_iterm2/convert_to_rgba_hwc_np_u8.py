import PIL.Image
from pathlib import Path
import numpy as np
from typing import Optional


def convert_to_rgba_hwc_np_u8(
    x: str | Path | PIL.Image.Image | np.ndarray,
    hint: Optional[str] = None
) -> np.ndarray:
    """
    Converts
    practically anything
    that is interpretable as an rgba rasterized/pixelated image
    to a numpy array of shape (height, width, 4) and dtype uint8.
    Weird stuff (like bgr channel order) you may need to provide a hint for,
    like hint="bgr"
    TODO: use the hint when it is provided to determine the colorspace and channel order.
    TODO: Does not seem to work if passed an rgb_hwc_np_u8 with 3 channels??!!
    """
    if isinstance(x, np.ndarray):
        assert x.ndim == 3, f"ERROR: {x.ndim=} is not 3"
        if x.dtype != np.uint8:
            raise ValueError(f"ERROR: {x.dtype=} is not np.uint8")
        if x.shape[2] not in [3, 4]:
            raise ValueError(f"ERROR: {x.shape=} does not have 3 or 4 channels")
        if x.shape[2] == 4:
            rgba_hwc_np_u8 = x
        if x.shape[2] == 3:
            rgba_hwc_np_u8 = np.zeros(
                (x.shape[0], x.shape[1], 4),
                dtype=np.uint8
            )
            rgba_hwc_np_u8[:, :, :3] = x
            rgba_hwc_np_u8[:, :, 3] = 255
    else:
        if hint is not None:
            raise NotImplementedError("ERROR: hint is not implemented yet for convert_to_rgba_hwc_np_u8.")
        image_pil = None
        image_path = None
        rgba_hwc_np_u8 = None
        if isinstance(x, Path):
            image_path = x
        elif isinstance(x, str):
            image_path = Path(x).expanduser().resolve()
            assert image_path.exists(), f"ERROR: {image_path=} does not exist"
        
        # at this point, every possible manner of getting a file path has happened,
        # so we try to open it as a PIL x.
        if image_path is not None:
            image_pil = PIL.Image.open(image_path)
        
        if isinstance(x, PIL.Image.Image):
            image_pil = x
        
        # at this point we have tried to get an x pil every way we can,
        # so if any of those worked, we can convert it to a
        # height x width x channel numpy array of uint8s:
        if image_pil is not None:
            rgba_hwc_np_u8 = np.array(image_pil.convert("RGBA"))
        
    

    assert rgba_hwc_np_u8.dtype == np.uint8, f"ERROR: {rgba_hwc_np_u8.dtype=} is not np.uint8"
    assert rgba_hwc_np_u8.ndim == 3, f"ERROR: {rgba_hwc_np_u8.ndim=} is not 3"
    assert rgba_hwc_np_u8.shape[2] == 4, f"ERROR: {rgba_hwc_np_u8.shape=} does not have 4 channels"
    
    if rgba_hwc_np_u8 is not None:
        return rgba_hwc_np_u8
    else:
        raise ValueError(f"ERROR: {x=} was not a path, PIL x, or 3 or 4 channel hwc uint8 numpy array")


