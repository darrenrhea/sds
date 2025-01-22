from prii_named_xy_points_on_image import (
     prii_named_xy_points_on_image
)
from open_as_hwc_rgba_np_uint8 import (
     open_as_hwc_rgba_np_uint8
)
from paste_cutout_onto_segmentation_annotation import (
     paste_cutout_onto_segmentation_annotation
)
from prii import prii


def test_paste_cutout_onto_segmentation_annotation():
    bottom_xy = (601, 465)
    top_xy = (373, 1503)

    fake_annotation_so_far_rgba_np_u8 = open_as_hwc_rgba_np_uint8(
        image_path="fixtures/fake_annotation_so_far_rgba_np_u8.png"
    )

    cutout_rgba_np_u8 = open_as_hwc_rgba_np_uint8(
        image_path="fixtures/cutout_rgba_np_u8.png"
    )

    prii_named_xy_points_on_image(
        image=cutout_rgba_np_u8,
        name_to_xy={
            "top_xy": top_xy
        }
    )

    prii_named_xy_points_on_image(
        image=fake_annotation_so_far_rgba_np_u8,
        name_to_xy={
            "bottom_xy": bottom_xy
        }
    )

    paste_cutout_onto_segmentation_annotation(
        # this gets mutated with every paste:
        fake_annotation_so_far_rgba_np_u8=fake_annotation_so_far_rgba_np_u8, 
        bottom_xy=bottom_xy,
        # a cutout of a basketball player, ball, some other foreground object.
        cutout_rgba_np_u8=cutout_rgba_np_u8,
        top_xy=top_xy,     
    )
    
    prii(fake_annotation_so_far_rgba_np_u8)


if __name__ == "__main__":
    test_paste_cutout_onto_segmentation_annotation()