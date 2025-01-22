from get_world_coordinate_descriptors_of_ad_placements import (
     get_world_coordinate_descriptors_of_ad_placements
)
from prii import (
     prii
)
from rip_world_points import (
     rip_world_points
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
import numpy as np

def test_rip_world_points_1():

    clip_id = "bay-zal-2024-03-15-mxf-yadif"
    frame_index = 94960


    camera_pose = get_camera_pose_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )
    ad_placement_descriptors = get_world_coordinate_descriptors_of_ad_placements(
        clip_id=clip_id,
        with_floor_as_giant_ad=False,
        overcover_by=0.0,
    )

    # remove the LEDBRD0 ad placement
    ad_placement_descriptors = [
        x
        for x in ad_placement_descriptors
        if x.name == "LEDBRD1"
    ]
    ad_placement_descriptor = ad_placement_descriptors[0]
    origin = ad_placement_descriptor.origin
    u = ad_placement_descriptor.u
    v = ad_placement_descriptor.v
    height = ad_placement_descriptor.height
    width = ad_placement_descriptor.width

    num_x_samples = 720
    num_y_samples = 96
    
    u_samples = np.linspace(0.0, width, num_x_samples)
    v_samples = np.linspace(height, 0.0, num_y_samples)

    xyzs = []
    for i in range(num_y_samples):
        for j in range(num_x_samples):
            p = origin + u_samples[j] * u + v_samples[i] * v
            xyzs.append(p)
            
    xyzs = np.array(xyzs)
    
    shared_dir = get_the_large_capacity_shared_directory()

    rgb_hwc_np_u8 = open_as_rgb_hwc_np_u8(
        image_path=shared_dir / "clips" / clip_id / "frames" / f"{clip_id}_{frame_index:06d}_original.jpg"
    )
    prii(rgb_hwc_np_u8)




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
    print(f"{out_rgba_hwc_np_u8.shape=}")
    prii(out_rgba_hwc_np_u8)
    

if __name__ == "__main__":
    test_rip_world_points_1()