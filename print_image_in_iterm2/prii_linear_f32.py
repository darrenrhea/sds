from convert_linear_f32_to_u8 import (
     convert_linear_f32_to_u8
)
from prii import (
     prii
)
import numpy as np


def prii_linear_f32(
    x: np.ndarray,
    caption=None,
    out=None,
):
    assert x.dtype == np.float32
    np_u8 = convert_linear_f32_to_u8(x)
    prii(np_u8, out=out, caption=caption)