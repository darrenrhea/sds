import sys
from show_insertion_description_id_without_color_correction import (
     show_color_correction_result_on_insertion_description_id
)
from get_color_correction_polynomial_coefficients_from_from_to_mapping_array_f64 import (
     get_color_correction_polynomial_coefficients_from_from_to_mapping_array_f64
)
from get_from_to_mapping_array_f64_from_insertion_description_id import (
     get_from_to_mapping_array_f64_from_insertion_description_id
)
import pprint as pp

from get_ad_placement_descriptor_from_jsonable import (
     get_ad_placement_descriptor_from_jsonable
)
from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from get_video_frame_path_from_clip_id_and_frame_index import (
     get_video_frame_path_from_clip_id_and_frame_index
)
from insert_ad_into_camera_posed_original_video_frame import (
     insert_ad_into_camera_posed_original_video_frame
)
from pathlib import Path
import numpy as np
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from prii import (
     prii
)
from CameraParameters import (
     CameraParameters
)
import better_json as bj

from color_correct_rgb_hwc_np_u8_image_via_polynomial_coefficients import (
     color_correct_rgb_hwc_np_u8_image_via_polynomial_coefficients
)
import pyperclip


def itvsaa_insert_the_very_same_ad_again():
    # an insertion is a bundle of:
    # an original frame to insert into
    # as well as its camera pose
    # and a mask of the image that we use to eliminate people and objects that
    # occlude the LED board
    # A png of image that was been displayed on the the LED board at that time
    # A description of a subrectangle of that image, 
    # because actually we insert only a subrectangle of the image
   
    # insertion_description_id = "6a9a25fb-9fbc-4fc2-9e14-cb6105b3d249"
    # insertion_description_id = "41bb4d0b-d2cf-4d15-8482-abb6493520ba"
    insertion_description_id = sys.argv[1]
    # sometimes you want to force certain pairs hard:
    additional_from_to_pairs = np.array(
        [
            [
                [255, 255, 255], [240, 247, 255],
            ],
            [
                [32, 32, 37], [0, 12, 50],
            ],
            [
                [195, 29, 41], [133, 33, 79],
            ],
        ]
    )
    additional_from_to_pairs = additional_from_to_pairs.astype(np.float64) / 255.0
    additional_from_to_pairs = additional_from_to_pairs.repeat(100000, axis=0)

    degree = 2
    from_to_mapping_array_f64s = []
    for insertion_description_id in insertion_description_ids:
        from_to_mapping_array_f64 = get_from_to_mapping_array_f64_from_insertion_description_id(
            insertion_description_id=insertion_description_id
        )
        from_to_mapping_array_f64s.append(from_to_mapping_array_f64)
    
    from_to_mapping_array_f64s.append(additional_from_to_pairs)

    from_to_mapping_array_f64 = np.concatenate(from_to_mapping_array_f64s, axis=0)

    coefficients = get_color_correction_polynomial_coefficients_from_from_to_mapping_array_f64(
        degree=degree,
        from_to_mapping_array_f64=from_to_mapping_array_f64,
    )
    for insertion_description_id in insertion_description_ids:
        show_color_correction_result_on_insertion_description_id(
            insertion_description_id=insertion_description_id,
            degree=degree,
            coefficients=coefficients
        )
   

 
if __name__ == "__main__":
    itvsaa_insert_the_very_same_ad_again()