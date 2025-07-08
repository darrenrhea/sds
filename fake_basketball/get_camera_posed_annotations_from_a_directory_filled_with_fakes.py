from make_map_from_annotation_id_to_camera_pose import (
     make_map_from_annotation_id_to_camera_pose
)
from color_correct_rgb_via_polynomial_coefficients_linear_f32_in_and_out import (
     color_correct_rgb_via_polynomial_coefficients_linear_f32_in_and_out
)
from convert_u8_to_linear_f32 import (
     convert_u8_to_linear_f32
)
from augment_texture import (
     augment_texture
)
from get_a_random_ad_they_sent_us_file_path import (
     get_a_random_ad_they_sent_us_file_path
)
from get_ad_name_to_paths_that_do_need_color_correction import (
     get_ad_name_to_paths_that_do_need_color_correction
)
from get_ad_name_to_paths_that_dont_need_color_correction import (
     get_ad_name_to_paths_that_dont_need_color_correction
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from convert_from_rgb_hwc_np_u8_to_rgba_hwc_np_nonlinear_f32 import (
     convert_from_rgb_hwc_np_u8_to_rgba_hwc_np_nonlinear_f32
)
from get_a_random_adrip_file_path import (
     get_a_random_adrip_file_path
)
from get_a_random_adrip_as_rgba_hwc_np_nonlinear_f32 import (
     maybe_augment_adrip
)
from insert_fake_ads_plain_version import (
     insert_fake_ads_plain_version
)
from insert_fake_ads_they_sent_to_us import (
     insert_fake_ads_they_sent_to_us
)
import pprint as pp
from get_rgba_hwc_np_f32_from_texture_id import (
     get_rgba_hwc_np_f32_from_texture_id
)
from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image
)
from add_noise_via_jpg_lossyness import (
     add_noise_via_jpg_lossyness
)
from create_ij_displacement_and_weight_pairs import (
     create_ij_displacement_and_weight_pairs
)
from blur_both_original_and_mask_u8 import (
     blur_both_original_and_mask_u8
)
from choose_by_percent import (
     choose_by_percent
)
from load_color_correction_from_json import (
     load_color_correction_from_json
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from get_world_coordinate_descriptors_of_ad_placements import (
     get_world_coordinate_descriptors_of_ad_placements
)
from make_relevance_mask_for_led_boards import (
     make_relevance_mask_for_led_boards
)
from write_rgb_hwc_np_u8_to_png import (
     write_rgb_hwc_np_u8_to_png
)
from write_grayscale_hw_np_u8_to_png import (
     write_grayscale_hw_np_u8_to_png
)
from write_rgba_hwc_np_u8_to_png import (
     write_rgba_hwc_np_u8_to_png
)
from get_augmentation_for_texture import (
     get_augmentation_for_texture
)
from add_camera_pose_to_annotations import (
     add_camera_pose_to_annotations
)
from get_approved_annotations_from_these_repos import (
     get_approved_annotations_from_these_repos
)
import argparse
import numpy as np
import time
from pathlib import Path
from prii import (
     prii
)
from get_valid_cutout_kinds import (
     get_valid_cutout_kinds
)
from better_json import (
     color_print_json
)
from group_cutouts_by_kind import (
     group_cutouts_by_kind
)
from prii_named_xy_points_on_image import (
     prii_named_xy_points_on_image
)
from get_cutouts import (
     get_cutouts
)
from make_rgba_from_original_and_mask_paths import (
     make_rgba_from_original_and_mask_paths
)
from paste_multiple_cutouts_onto_one_camera_posed_segmentation_annotation import (
     paste_multiple_cutouts_onto_one_camera_posed_segmentation_annotation
)
from get_cutout_augmentation import (
     get_cutout_augmentation
)

import pprint

def get_camera_posed_annotations_from_a_directory_filled_with_fakes(
    directory: Path,
    video_frame_annotations_metadata_sha256: str
):
    """
    Get camera posed annotations from a directory of fakes.
    """

    map_from_annotation_id_to_camera_pose = make_map_from_annotation_id_to_camera_pose(
        video_frame_annotations_metadata_sha256=video_frame_annotations_metadata_sha256
    )
    pprint.pprint(map_from_annotation_id_to_camera_pose)

    s = "_original.jpg"
    L = len(s)
    directory = directory.resolve()
    camera_posed_annotations = []
    for file_path in directory.glob("*_original.jpg"):
        
        original_file_path = file_path
        name = file_path.name
        assert name.endswith(s)
        removed = name[:-L]
        typical_fake_part = "_fake723718586022605"
        fl = len(typical_fake_part)
        fake_part = removed[-fl:]
        # print(f"{fake_part=}")
        assert fake_part.startswith("_fake")
        annotation_id = removed[:-fl]
        reconstructed = annotation_id + fake_part + "_original.jpg"
        assert name == reconstructed, f"ERROR: {name=} {reconstructed=}"
        mask_file_path = directory / (annotation_id + fake_part + "_nonfloor.png")
        assert mask_file_path.is_file()

        camera_pose = map_from_annotation_id_to_camera_pose.get(
            annotation_id
        )
        if camera_pose is None:
            continue

        camera_posed_annotation = dict(
            original_file_path=original_file_path,
            annotation_id=annotation_id,
            mask_file_path=mask_file_path,
            camera_pose=camera_pose,
        )
        camera_posed_annotations.append(camera_posed_annotation)
    
    return camera_posed_annotations


if __name__ == "__main__":
    directory = Path("/shared/fake_nba/underneaths2")
    video_frame_annotations_metadata_sha256 = "99bc2c688a6bd35f08b873495d062604e0b954244e6bb20f5c5a76826ac53524"

    camera_posed_annotations = get_camera_posed_annotations_from_a_directory_filled_with_fakes(
        video_frame_annotations_metadata_sha256=video_frame_annotations_metadata_sha256,
        directory=directory,
    )

    print(f"Got {len(camera_posed_annotations)} camera posed annotations.")