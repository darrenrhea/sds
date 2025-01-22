from write_rgb_and_alpha_to_png import (
     write_rgb_and_alpha_to_png
)
from make_rgba_from_original_and_mask_paths import (
     make_rgba_from_original_and_mask_paths
)
import cv2
import numpy as np
from pathlib import Path


def get_connected_components_of_mask(
    # the original image, has historically been a .jpg and will be until the migration to  16-bit .pngs happens.
    original_path: Path,
    # where to find the importantpeople segmentation convention mask:
    mask_path: Path,
    # where to save them all:
    out_dir_path: Path,
    desired: str
) -> None:
    """
    Cuts the importantpeople out into connected components and saves them to disk
    as grayscale masks.

    An (importantpeople segmentation convention) mask has many connected components
    that are either a single person or a clump-of-persons.

    All such 4K basketball importantpeople masks
    can be automatically turned into
    cutouts by saving the mask of each connected component.
    
    A human being can then quickly iterate through the resulting
    folder of maybe-cutouts, rejecting tidbits by deleting them
    and labeling the rest with its kind (clump, referee, player, coach, etc.)

    It is naive to think that two connected components can
    always be separated by a bounding box.  Although
    polygons (not necessarily convex) can separate connected components,
    the programming complexity is too high for now.
    so we save the full mask of each connected component to disk.
    """

    rgba = make_rgba_from_original_and_mask_paths(
        original_path=original_path,
        mask_path=mask_path,
        flip_mask=False,
        quantize=True
    )

    # prii(rgba)

    full_size_height = rgba.shape[0]
    full_size_width = rgba.shape[1]
    original_np = rgba[:, :, :3]    
   

   
    alpha = rgba[:, :, 3]
    # print_image_in_iterm2(grayscale_np_uint8=alpha_uint8)
    # flip it to indicate where the LED screens are:
    binary = (alpha > 127).astype(np.uint8)

    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    
    connected_components = []
    for i in range(numLabels):
        if i == 0:
            continue  # the opposite is label 0?
        
        mask = (labels == i).astype(np.uint8) * 255

        x0, y0, bbox_width, bbox_height, measure = stats[i]

        print(f"{measure=}")
        # prii(mask)

        component = dict(
            xmin=x0,
            xmax=x0 + bbox_width,
            ymin=y0,
            ymax=y0 + bbox_height,
            label=i,
            measure=measure,
            mask=mask
        )
        connected_components.append(component)

    connected_components = sorted(
        connected_components,
        key=lambda c: (c["xmin"], c["ymin"], c["measure"]),
        reverse=False
    )

    if desired == "big":
        connected_components = [
            c for c in connected_components
            if c["measure"] > 10000
        ]
    else:
        connected_components = [
            c for c in connected_components
            if c["measure"] < 4000
        ]

    

   

    annotation_id = original_path.stem
    for component_index, component in enumerate(connected_components):
        margin_of_error = 200
        xmin = component["xmin"]
        xmax = component["xmax"]
        ymin = component["ymin"]
        ymax = component["ymax"]
        xmin = max(0, xmin - margin_of_error)
        ymin = max(0, ymin - margin_of_error)
        xmax = min(full_size_width, xmax + margin_of_error)
        ymax = min(full_size_height, ymax + margin_of_error)

        original_name = original_path.name
        mask_name = mask_path.name
        annotation_id = original_path.stem

        cutout_id = f"{annotation_id}_{component_index:02d}"

        # This is less useful than we thought it would be, and it interferes with the json
        # that stores the cutout's metadata like kind, league, team, toe point, six_feet_above_that, etc.
        # jsonable = dict(
        #     camera_pose={},
        #     cutout_name=cutout_id,
        #     source_original=original_name,
        #     source_mask=mask_name,
        #     xmin=int(xmin),
        #     xmax=int(xmax),
        #     ymin=int(ymin),
        #     ymax=int(ymax),
        #     annotation_id=annotation_id,
        #     component_index=component_index
        # )

        # bj.color_print_json(jsonable)
        # json_path = out_dir_path / f"{cutout_id}.json"
        # bj.dump(obj=jsonable, fp=json_path)
        
        rgba_cutout_out_file_path = out_dir_path / f"{cutout_id}.png"



        mask = component["mask"]
        
        print(f"pri {rgba_cutout_out_file_path}")

        write_rgb_and_alpha_to_png(
            rgb_hwc_np_u8=original_np[ymin:ymax, xmin:xmax, :],
            alpha_hw_np_u8=mask[ymin:ymax, xmin:xmax],
            out_abs_file_path=rgba_cutout_out_file_path,
        )

   


