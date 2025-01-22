from feathered_paste_for_images_of_the_same_size import (
     feathered_paste_for_images_of_the_same_size
)
import cv2
from get_top_xy_for_cutout import (
     get_top_xy_for_cutout
)
from get_valid_cutout_kinds import (
     get_valid_cutout_kinds
)
from scale_image_and_point_together import (
     scale_image_and_point_together
)
from prii_named_xy_points_on_image import (
     prii_named_xy_points_on_image
)
from PasteableCutout import (
     PasteableCutout
)
from typing import Any, Dict, List
from CameraParameters import (
     CameraParameters
)
from augment_cutout import (
     augment_cutout
)
from paste_cutout_onto_segmentation_annotation import (
     paste_cutout_onto_segmentation_annotation
)
import numpy as np
from make_placement_descriptors_for_this_cutout_kind import (
     make_placement_descriptors_for_this_cutout_kind
)


def paste_multiple_cutouts_onto_one_camera_posed_segmentation_annotation(
    league: str,
    context_id: str,
    cutouts_by_kind: Dict[str, List[PasteableCutout]],
    rgba_np_u8: np.ndarray,  # this is not violated by this procedure.
    camera_pose: CameraParameters,
    cutout_kind_to_transform: dict[str, Any], # what albumentations augmentation to use per kind of cutout
    cutout_kind_to_num_cutouts_to_paste: Dict[str, int],  # per kind, how many cutouts to paste onto an actual annotation to make a fake annotation
) -> np.array:
    """
    This is going for the augment-the-cutout right before pasting design.
    This makes one fake/synthetic annotation
    out of one actual segmentation annotation
    by pasting multiple cutouts onto the actual annotation.
    See paste_cutout_onto_segmentation_annotations.py
    """
    assert isinstance(rgba_np_u8, np.ndarray), f"rgba_np_u8 is not a np.ndarray, but a {type(rgba_np_u8)}"
    assert rgba_np_u8.ndim == 3, f"ERROR: {rgba_np_u8.shape=}"
    assert rgba_np_u8.shape[2] == 4, f"ERROR: {rgba_np_u8.shape=}"
    valid_cutout_kinds = get_valid_cutout_kinds()

    # At the beginning, the fake annotation is just a defensive copy of the actual annotation, but successive pastes will mutate it:
    real_rgb = rgba_np_u8[:, :, :3].copy()
    fake_annotation_so_far_rgba_np_u8 = rgba_np_u8.copy()
    
    
    # pick which cutouts to add to the actual annotation, right now all cutouts are added to all actual annotations:  
    
    # use the camera pose to get some xys where to paste the cutouts:

    # print(camera_pose)
    photograph_width_in_pixels = rgba_np_u8.shape[1]
    photograph_height_in_pixels = rgba_np_u8.shape[0]
    
    unsorted_placement_descriptors = []
    for cutout_kind in valid_cutout_kinds:
        placement_descriptors_for_this_kind = make_placement_descriptors_for_this_cutout_kind(
            league=league,
            context_id=context_id,
            cutout_kind=cutout_kind,
            camera_pose=camera_pose,
            photograph_width_in_pixels=photograph_width_in_pixels,
            photograph_height_in_pixels=photograph_height_in_pixels,
            num_cutouts_to_paste=cutout_kind_to_num_cutouts_to_paste[cutout_kind],
        )
        unsorted_placement_descriptors.extend(placement_descriptors_for_this_kind)

    # sensible to sort by distance from camera, so that the cutouts that are closer to the camera are pasted first:
    placement_descriptors = sorted(unsorted_placement_descriptors, key=lambda x: x["distance_from_camera"], reverse=True)
    # color_print_json(placement_descriptors)
    
   
    print_bottom_xys = False
    if print_bottom_xys:
        prii_named_xy_points_on_image(
            image=rgba_np_u8,
            name_to_xy={
                f"{placement_descriptor['cutout_kind']}_{placement_descriptor['index_within_kind']}"
                :
                placement_descriptor["bottom_xy"]
                for placement_descriptor in placement_descriptors
            }
        )

    for placement_descriptor in placement_descriptors:
        bottom_xy = placement_descriptor["bottom_xy"]
        how_many_pixels_is_six_feet_at_that_point = placement_descriptor["how_many_pixels_is_six_feet_at_that_point"]
        cutout_kind = placement_descriptor["cutout_kind"]
        cutouts_of_that_kind = cutouts_by_kind[cutout_kind]
        # pick a cutout of that kind to add to the actual annotation:
        cutout_index = np.random.randint(0, len(cutouts_of_that_kind))
        cutout = cutouts_of_that_kind[cutout_index]
        assert cutout.kind == cutout_kind, "ERROR: cutout.kind != cutout_kind?!"
        original_cutout_rgba_np_u8 = cutout.rgba_np_u8
        
       
        # use the cutout metadata to get their toe point,
        # unless it's a ball, in which case we just pick a random point
        # in front of the screen:
        unaugmented_top_xy = get_top_xy_for_cutout(
            cutout=cutout
        )
        # augment live/online: This makes it more likely to never be the same twice.
        cutout_rgba_np_u8, top_xy = augment_cutout(
            rgba_np_u8=original_cutout_rgba_np_u8,
            top_xy=unaugmented_top_xy,
            transform=cutout_kind_to_transform[cutout_kind],
        )

        # scale the cutout so that people are like six feet tall:
        scale_factor = how_many_pixels_is_six_feet_at_that_point / cutout.how_many_pixels_is_six_feet
        scaled_cutout_rgba_np_u8, scaled_top_xy = scale_image_and_point_together(
            rgba_np_u8=cutout_rgba_np_u8,
            top_xy=top_xy,
            scale_factor=scale_factor,
        )
            
        paste_cutout_onto_segmentation_annotation(
            # this gets mutated with every paste:
            fake_annotation_so_far_rgba_np_u8=fake_annotation_so_far_rgba_np_u8, 
            bottom_xy=bottom_xy,
            # a cutout of a basketball player, ball, some other foreground object.
            cutout_rgba_np_u8=scaled_cutout_rgba_np_u8,
            top_xy=scaled_top_xy,     
        )

    # all pasting is done, do some post-processing:
    harshly_pasted_rgb  = fake_annotation_so_far_rgba_np_u8[:, :, :3].copy()
    fake_alpha = fake_annotation_so_far_rgba_np_u8[:, :, 3].copy()

    if np.random.randint(0, 2) == 0:
        # these have to be ints:
        xstd = np.random.randint(1, 3+1)
        ystd = np.random.randint(1, 3+1)
        blurred_alpha = cv2.blur(
            fake_alpha,
            (xstd, ystd)
        )
    else:
        blurred_alpha = fake_alpha

    top_layer_color_np_uint8 = np.zeros_like(fake_annotation_so_far_rgba_np_u8)
    top_layer_color_np_uint8[:, :, :3] = harshly_pasted_rgb
    top_layer_color_np_uint8[:, :, 3] = blurred_alpha

    blended = feathered_paste_for_images_of_the_same_size(
        bottom_layer_color_np_uint8=real_rgb,
        top_layer_rgba_np_uint8=top_layer_color_np_uint8,
    )

    blurred_rgb = blended[:, :, :3].copy()

    fake_rgba = np.concatenate(
        [blurred_rgb, fake_alpha[:, :, np.newaxis]],
        axis=2
    )

    return fake_rgba
   