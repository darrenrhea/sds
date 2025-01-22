from quantize_colors_via_kmeans import (
     quantize_colors_via_kmeans
)
from gather_colors_on_indicator import (
     gather_colors_on_indicator
)
from get_approved_annotations_from_these_repos import (
     get_approved_annotations_from_these_repos
)
from pathlib import Path
import numpy as np
import PIL
import PIL.Image
from CameraParameters import CameraParameters
from open_as_rgba_hwc_np_u8 import (
     open_as_rgba_hwc_np_u8
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from render_ads_on_subregion import (
     render_ads_on_subregion
)
from get_world_coordinate_descriptors_of_ad_placements import (
     get_world_coordinate_descriptors_of_ad_placements
)
from draw_euroleague_landmarks import (
     draw_euroleague_landmarks
)
from feathered_paste_for_images_of_the_same_size import (
     feathered_paste_for_images_of_the_same_size
)
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from get_euroleague_geometry import (
     get_euroleague_geometry
)
from prii import (
     prii
)

def find_color_mapping_from_samples(
    original_rgba_np_u8: np.ndarray,
    real_indicator: np.ndarray,
    fake_rgba_np_u8: np.ndarray,
    fake_indicator: np.ndarray
):
    """
    Given an actual image of an LED ad and the indicator of where the ad is,
    and given the fake image inserted into the real image,
    try to find a color mapping from the fake images colors (which tend to be very saturated)
    to the real colors caused by LED optoelectronics, camera optoelectronics, and lighting conditions.
    """
    real_rgb_values = gather_colors_on_indicator(
        rgb_or_rgba_np_u8=original_rgba_np_u8,
        indicator=real_indicator
    )
    
    real_cluster_indices, real_centroids = quantize_colors_via_kmeans(
        rgb_values=real_rgb_values
    )
    
    _, real_counts = np.unique(real_cluster_indices, return_counts=True)

    print("real_centroids:")
    print(real_centroids)
    print(real_counts)
    assert (
        list(real_counts) == sorted(list(real_counts),reverse=True)
    ), "ERROR: The real_counts should be sorted in descending order."

    fake_rgb_values = gather_colors_on_indicator(
        rgb_or_rgba_np_u8=fake_rgba_np_u8,
        indicator=fake_indicator
    )

    fake_cluster_indices, fake_centroids = quantize_colors_via_kmeans(
        rgb_values=fake_rgb_values
    )
    _, fake_counts = np.unique(fake_cluster_indices, return_counts=True)

    print("fake_centroids:")
    print(fake_centroids)

    assert (
        list(fake_counts) == sorted(list(fake_counts),reverse=True)
    ), "ERROR: The fake_counts should be sorted in descending order."


    print("We suggest this map from led_videos to reality:")
    print(f"{fake_centroids[0]} => {real_centroids[0]}")
    print(f"{fake_centroids[1]} => {real_centroids[1]}")
    print(f"{fake_centroids[2]} => {real_centroids[2]}")

    num_color_mappings = 3
    color_map = np.zeros(
        (num_color_mappings, 2, 3),
        dtype=np.uint8
    )

    color_map[0, 0] = fake_centroids[0]
    color_map[0, 1] = real_centroids[0]
    color_map[1, 0] = fake_centroids[1]
    color_map[1, 1] = real_centroids[1]
    color_map[2, 0] = fake_centroids[2]
    color_map[2, 1] = real_centroids[2]
    bigger = color_map.repeat(200 ,axis=0).repeat(200, axis=1)

    prii(bigger)


   

