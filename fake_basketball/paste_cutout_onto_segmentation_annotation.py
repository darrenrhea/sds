from draw_marker_on_np_u8 import (
     draw_marker_on_np_u8
)
from prii import (
     prii
)
from translated_feathered_paste import (
     translated_feathered_paste
)

import numpy as np
from typing import Tuple



def paste_cutout_onto_segmentation_annotation(
    # The fake annotation so far.
    # to begin this may be a copy of an actual annotation,
    # then this proc mutates it to get faker with every paste:
    fake_annotation_so_far_rgba_np_u8: np.ndarray,
    bottom_xy: Tuple[int, int],  # an origin point on the fake-annotation-so-far.
    cutout_rgba_np_u8: np.ndarray, # the cutout, already loaded into memory
    top_xy: Tuple[int, int], # a point on the cutout that will be placed on top of bottom_xy
    verbose=False
) -> None:  # acts by mutating fake_annotation_so_far_rgba_np_u8
    """
    See also paste_cutouts_onto_segmentation_annotations,
    which for-loops this over various cutouts and various
    actual segmentation annotations to generate a lot of
    fake data.

    This takes in an actual LED annotation and a cutout,
    say of a basketball player, and forms a
    relevant-masked fake segmentation annotation.
    """
    assert isinstance(top_xy[0], int)
    assert isinstance(top_xy[1], int)
    assert isinstance(bottom_xy[0], int), f"{type(bottom_xy[0])=} but should be int"
    assert isinstance(bottom_xy[1], int), f"{type(bottom_xy[1])=} but should be int"
    bottom_layer_color_np_uint8 = fake_annotation_so_far_rgba_np_u8[:, :, :3]
    actual_mask = fake_annotation_so_far_rgba_np_u8[:, :, 3]

    if verbose:
        temp = bottom_layer_color_np_uint8.copy()
        draw_marker_on_np_u8(xy=bottom_xy, victim=temp, r=200)
        prii(temp, caption="bottom_layer_color_np_uint8 with bottom_xy origin point indicated by green cross:")

    if verbose:
        temp = cutout_rgba_np_u8.copy()
        draw_marker_on_np_u8(xy=top_xy, victim=temp)
        prii(temp, caption="cutout_rgba_np_u8 with origin point indicated by green cross:")


    fake_original_np_u8, translated_cutout_mask = translated_feathered_paste(
        bottom_layer_color_np_uint8=bottom_layer_color_np_uint8,
        top_layer_rgba_np_uint8=cutout_rgba_np_u8,
        top_xy=top_xy,
        bottom_xy=bottom_xy
    )

    # Surely this is wrong but whatever.
    # transparency multiplicative, so:
    # opacity total = 1 - (1-o1)(1-o2)
    fake_mask = np.maximum(translated_cutout_mask, actual_mask)

    if verbose:
        prii(fake_original_np_u8, caption="The resulting fake original image caused by pasting:")

    # mutate it:
    fake_annotation_so_far_rgba_np_u8[:, :, 3] = fake_mask
    fake_annotation_so_far_rgba_np_u8[:, :, :3] = fake_original_np_u8

