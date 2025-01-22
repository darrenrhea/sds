import numpy as np


def translated_paste_onto_blank_canvas(
    desired_height: int,
    desired_width: int,
    top_layer_rgba_np_uint8,
    i0: int,
    j0: int
):
    """
    Paste a possibly very differently-sized cutout image (might be wider or narrower, might be taller or shorter) onto
    a transparent, desired size canvas that is desired_height x desired_width,
    such that the top left corner pixel of the cutout image will be pasted onto the (i0, j0) pixel in the canvas.
    """
    assert isinstance(i0, int), f"i0 is not an int: {type(i0)=} and value {i0=}"
    assert isinstance(j0, int), f"j0 is not an int: {type(j0)=} and value {j0=}"
    h = desired_height
    w = desired_width
    if top_layer_rgba_np_uint8.ndim == 3:
        canvas_sized_np_uint8 = np.zeros(
            shape=(
                h,
                w,
                top_layer_rgba_np_uint8.shape[2]
            ),
            dtype=np.uint8
        )
    elif top_layer_rgba_np_uint8.ndim == 2:
        canvas_sized_np_uint8 = np.zeros(
            shape=(
                h,
                w
            ),
            dtype=np.uint8
        )
    
    clip_height = top_layer_rgba_np_uint8.shape[0]
    clip_width = top_layer_rgba_np_uint8.shape[1]

    # there are several ways to paste with no intersection with the canvas.
    if i0 >= h:
        return canvas_sized_np_uint8
    if j0 >= w:
        return canvas_sized_np_uint8
    if i0 <= -clip_height:
        return canvas_sized_np_uint8
    if j0 <= -clip_width:
        return canvas_sized_np_uint8

  

    # [a, b) = [0, clip_height) intersect [-i0, -i0 + height)

    # [A, B) = [0, h) intersect [i0, i0 + clip_height)
   
    a = max(0, -i0)
    b = min(h - i0, clip_height)
    c = max(0, -j0)
    d = min(clip_width, w - j0)
    A = max(0, i0)
    B = min(h, i0 + clip_height)
    C = max(0, j0)
    D = max(min(w, j0 + clip_width), 0)
    assert B - A == b - a
    assert A >= 0
    assert B >= 0, f"{B=} because {h=} and {i0=} and {clip_height=}"
    assert C >= 0
    assert D >= 0
    assert A <= B
    assert C <= D
    assert B <= h
    assert D <= w
    canvas_sized_np_uint8[A:B, C:D, ...] = top_layer_rgba_np_uint8[a:b, c:d, ...]
    assert canvas_sized_np_uint8.shape[0] == h
    return canvas_sized_np_uint8

