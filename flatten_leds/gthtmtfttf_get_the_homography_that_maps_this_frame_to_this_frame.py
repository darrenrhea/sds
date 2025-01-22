from print_red import (
     print_red
)
import textwrap
from find_homography_from_2d_correspondences import (
     find_homography_from_2d_correspondences
)
import numpy as np
from prii_named_xy_points_on_image import (
     prii_named_xy_points_on_image
)
import pprint
from gwklfcafi_get_well_known_landmarks_from_clip_id_and_frame_index import (
     gwklfcafi_get_well_known_landmarks_from_clip_id_and_frame_index
)
from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)


def gthtmtfttf_get_the_homography_that_maps_this_frame_to_this_frame(
    clip_id: str,
    src_frame_index: int,
    dst_frame_index: int,
    explain: bool = False,
): 
    landmark_name_to_xy_coordinate_in_src_image = (
        gwklfcafi_get_well_known_landmarks_from_clip_id_and_frame_index(
    
            clip_id=clip_id,
            frame_index=src_frame_index
        )
    )

    if explain:
        print(f"For frame {src_frame_index} the landmarks are:")
        pprint.pprint(
            landmark_name_to_xy_coordinate_in_src_image
        )

    landmark_name_to_xy_coordinate_in_dst_image = (
        gwklfcafi_get_well_known_landmarks_from_clip_id_and_frame_index(
    
            clip_id=clip_id,
            frame_index=dst_frame_index
        )
    )
    
    
    common_landmark_names = (
        set(landmark_name_to_xy_coordinate_in_src_image.keys())
        &
        set(landmark_name_to_xy_coordinate_in_dst_image.keys())
    )


    if explain:
        print(f"For frame {dst_frame_index} the landmarks are:")

        pprint.pprint(
            landmark_name_to_xy_coordinate_in_dst_image
        )

        print("The landmarks that are visible in both frames are:")
        pprint.pprint(common_landmark_names)
    
    possible = (
        len(landmark_name_to_xy_coordinate_in_dst_image) >= 4
    )

    if not possible:
        print_red("Not possible")
        explain = True

   

    if explain:
        src_image = get_original_frame_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=src_frame_index,
        )

        dst_image = get_original_frame_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=dst_frame_index,
        )
        # print("src image:")
        # prii_named_xy_points_on_image(
        #     name_to_xy=landmark_name_to_xy_coordinate_in_src_image,
        #     image=src_image,
        #     output_image_file_path= None,
        #     default_color=(0, 255, 0),  # green is the default
        #     dont_show=False,
        # )
        print("dst image:")
        prii_named_xy_points_on_image(
            name_to_xy=landmark_name_to_xy_coordinate_in_dst_image,
            image=dst_image,
            output_image_file_path= None,
            default_color=(0, 255, 0),  # green is the default
            dont_show=False,
        )

    if not possible:
        return None
    
    # we make n x 2 numpy arrays of the observation (x_pixel, y_pixel) locations of the landmarks they have in common:
    src_points_np = np.array(
        [
            landmark_name_to_xy_coordinate_in_src_image[landmark_name]
            for landmark_name in common_landmark_names
        ]
    )

    dst_points_np = np.array(
        [
            landmark_name_to_xy_coordinate_in_dst_image[landmark_name]
            for landmark_name in common_landmark_names
        ]
    )
    dct = find_homography_from_2d_correspondences(
        src_points=src_points_np,
        dst_points=dst_points_np,
        ransacReprojThreshold=20.0,
    )
    
    assert (
        dct["success"],
    ), textwrap.dedent(
        f"""\
        We failed to find the homography from
        frame {src_frame_index}
        to
        frame {dst_frame_index}
        """
    )

    H = dct["homography_3x3"]

    assert isinstance(H, np.ndarray)
    assert H.shape == (3, 3)

    return H


if __name__ == "__main__":
    gthtmtfttf_get_the_homography_that_maps_this_frame_to_this_frame(
        clip_id="brewcub",
        src_frame_index=23094,  # we manually made the polygon to cover the crowd area
        dst_frame_index=41500,
    )