from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)
import pprint as pp
from get_world_coordinate_descriptors_of_ad_placements import (
     get_world_coordinate_descriptors_of_ad_placements
)
from prii import (
     prii
)
from rip_world_points import (
     rip_world_points
)
from get_euroleague_geometry import (
     get_euroleague_geometry
)
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from forward_project_world_points import (
     forward_project_world_points
)
import numpy as np
from scipy.ndimage import map_coordinates
from CameraParameters import CameraParameters

def test_rip_world_points_1():

    clip_id = "hou-sas-2024-10-17-sdi"
    frame_index = 160000


    camera_pose = get_camera_pose_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )

    # ad_placement_descriptors = get_world_coordinate_descriptors_of_ad_placements(
    #     clip_id=clip_id,
    #     with_floor_as_giant_ad=False,
    #     overcover_by=0.0,
    # )

    # # remove the LEDBRD0 ad placement
    # ad_placement_descriptors = [
    #     x
    #     for x in ad_placement_descriptors
    #     if x.name == "LEDBRD1"
    # ]
    # ad_placement_descriptor = ad_placement_descriptors[0]
    # origin = ad_placement_descriptor.origin
    # u = ad_placement_descriptor.u
    # v = ad_placement_descriptor.v
    # height = ad_placement_descriptor.height
    # width = ad_placement_descriptor.width
    # pp.pprint(ad_placement_descriptor)


    rgb_hwc_np_u8 = get_original_frame_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )
    prii(rgb_hwc_np_u8)

    num_x_samples = 3840
    num_y_samples = 1920
    xs = np.linspace(-16.0, 0.0, num_x_samples)
    ys = np.linspace(8.5, -7.5, num_y_samples)

    x, y = np.meshgrid(
        ys,
        xs,
        indexing="ij"
    )
    xyzs = []
    for i in range(num_y_samples):
        for j in range(num_x_samples):
            xyzs.append(
                [xs[j], ys[i], 0.0]
            )
    xyzs = np.array(xyzs)


    # xyzs = np.stack(
    #     [
    #         x.reshape(-1),
    #         y.reshape(-1),
    #         np.zeros(shape=(x.size,)),
    #     ],
    #     axis=1
    # )

    rgba_values_f32 = rip_world_points(
        rgb_hwc_np_u8=rgb_hwc_np_u8,
        camera_pose=camera_pose,
        xyzs=xyzs,
    )
    
    out_rgba_hwc_np_u8 = np.round(rgba_values_f32).clip(0, 255).astype(np.uint8).reshape(num_y_samples, num_x_samples, 4)
    out_rgba_hwc_np_u8[:, :, 3] = 255
    print(f"{out_rgba_hwc_np_u8.shape=}")
    prii(out_rgba_hwc_np_u8)
    

if __name__ == "__main__":
    test_rip_world_points_1()