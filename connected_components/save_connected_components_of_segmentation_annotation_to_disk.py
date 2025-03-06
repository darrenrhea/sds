from connected_component_masks import (
     connected_component_masks
)
from prii import (
     prii
)
from write_rgb_and_alpha_to_png import (
     write_rgb_and_alpha_to_png
)
from make_rgba_from_original_and_mask_paths import (
     make_rgba_from_original_and_mask_paths
)
import numpy as np
from pathlib import Path


def save_connected_components_of_segmentation_annotation_to_disk(
    # the original image, has historically been a .jpg and will be until the migration to 16-bit .pngs happens.
    original_path: Path,
    # where to find the importantpeople segmentation convention mask:
    mask_path: Path,
    # where to save them all:
    out_dir_path: Path,
    desired: str  # say "big" or "small"
) -> None:
    """
    Cuts out connected components and saves them to disk.

    A floor_not_floor mask has many connected components
    that are either a single person or a clump-of-persons
    THAT DOESNT TOUCH THE BOUNDARY OF THE IMAGE.

    All such cutouts are saved to disk can be automatically cut-out.
    
    A human being can then quickly iterate through the resulting
    folder of maybe-cutouts, rejecting tidbits by deleting them
    and labeling the rest with its kind (clump, referee, player, coach, etc.)
    """

    rgba = make_rgba_from_original_and_mask_paths(
        original_path=original_path,
        mask_path=mask_path,
        flip_mask=False,
        quantize=True
    )

    full_size_height = rgba.shape[0]
    full_size_width = rgba.shape[1]
    original_np = rgba[:, :, :3]    
   

   
    alpha = rgba[:, :, 3]
    # print_image_in_iterm2(grayscale_np_uint8=alpha_uint8)
    # flip it to indicate where the LED screens are:
    binary = (alpha > 32).astype(np.uint8)
 
    connected_components = connected_component_masks(
        binary_hw_np_u8=binary
    )

    connected_components = sorted(
        connected_components,
        key=lambda c: (c["xmin"], c["ymin"], c["measure"]),
        reverse=False
    )
    biggest_size = max([c["measure"] for c in connected_components])
    print(f"biggest_size {biggest_size}")
    if desired == "big":
        connected_components = [
            c for c in connected_components
            if c["measure"] >= biggest_size
        ]
    else:
        connected_components = [
            c for c in connected_components
            if c["measure"] < biggest_size
        ]

    annotation_id = original_path.stem
    for component_index, component in enumerate(connected_components):
        margin_of_error = 200
        xmin = component["xmin"]
        xmax = component["xmax"]
        ymin = component["ymin"]
        ymax = component["ymax"]
        
        # BEGIN we don't want people who are half off the left side of the image:
        if xmin == 0:
            continue        
        if xmax == full_size_width:
            continue
        if ymin == 0:
            continue
        if ymax == full_size_height:
            continue
        # ENDOF we don't want people who are half off the left side of the image.

        xmin = max(0, xmin - margin_of_error)
        ymin = max(0, ymin - margin_of_error)
        xmax = min(full_size_width, xmax + margin_of_error)
        ymax = min(full_size_height, ymax + margin_of_error)

        annotation_id = original_path.stem

        cutout_id = f"{annotation_id}_{component_index:02d}"
 
        rgba_cutout_out_file_path = out_dir_path / f"{cutout_id}.png"

        discrete_mask = component["mask"] * 255
        alpha_hw_np_u8 = np.minimum(
            discrete_mask,
            alpha,
        )
        write_rgb_and_alpha_to_png(
            rgb_hwc_np_u8=original_np,
            alpha_hw_np_u8=alpha_hw_np_u8,
            out_abs_file_path=rgba_cutout_out_file_path,
        )
