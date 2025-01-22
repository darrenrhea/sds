from calculate_metrics import (
     calculate_metrics
)
from AdPlacementDescriptor import (
     AdPlacementDescriptor
)
import pprint as pp
import numpy as np
from CameraParameters import (
     CameraParameters
)

from render_then_get_from_to_mapping_array import (
     render_then_get_from_to_mapping_array
)

def mutate_ad_placement_descriptor_and_camera_pose(
        ad_placement_descriptor,
        camera_pose,
):
    new_u = ad_placement_descriptor.u.copy()
    new_v = ad_placement_descriptor.v.copy()
    new_origin = ad_placement_descriptor.origin.copy()

    new_width = ad_placement_descriptor.width
    new_height = ad_placement_descriptor.height
    
    new_origin += 0.01 * np.array(
        [
            np.random.randn(),
            np.random.randn(),
            np.random.randn(),
        ]
    )
    new_width += 0.01 * np.random.randn()
    new_height += 0.01 * np.random.randn()

    new_ad_placement_descriptor = AdPlacementDescriptor(
        name="LED",
        origin=new_origin,
        u=new_u,
        v=new_v,
        height=new_height,
        width=new_width
    )

    new_rod = camera_pose.rod.copy() + 0.0001 * np.random.randn()
    new_loc = camera_pose.loc.copy() + 0.0001 * np.random.randn()
    new_f = camera_pose.f + 0.001 * np.random.randn()
    new_k1 = camera_pose.k1 + 0.001 * np.random.randn()
    new_k2 = camera_pose.k2 + 0.001 * np.random.randn()

    new_camera_pose = CameraParameters(
        rod=new_rod,
        loc=new_loc,
        f=new_f,
        k1=new_k1,
        k2=new_k2,
    )

    return new_ad_placement_descriptor, new_camera_pose


def optimize_self_insertion(
    original_rgb_np_u8: np.ndarray,
    camera_pose: CameraParameters,
    mask_for_regression_hw_np_u8: np.ndarray,
    texture_rgba_np_f32: np.ndarray,
    ad_placement_descriptor: AdPlacementDescriptor,
    max_iters: int,
):
    u = ad_placement_descriptor.u
    v = ad_placement_descriptor.v
    origin = ad_placement_descriptor.origin
    width = ad_placement_descriptor.width
    height = ad_placement_descriptor.height

    ad_placement_descriptor = AdPlacementDescriptor(
        name="LED",
        origin=origin,
        u=u,
        v=v,
        height=height,
        width=width
    )

    from_to_mapping_array_f64 = render_then_get_from_to_mapping_array(
        original_rgb_np_u8=original_rgb_np_u8,
        camera_pose=camera_pose,
        mask_for_regression_hw_np_u8=mask_for_regression_hw_np_u8,
        ad_placement_descriptor=ad_placement_descriptor,
        texture_rgba_np_f32=texture_rgba_np_f32,
        use_linear_light=False,
    )

    best_corr = calculate_metrics(
        from_to_mapping_array_f64=from_to_mapping_array_f64
    )
    print(f"Initially, {best_corr=} is caused by")
    pp.pprint(ad_placement_descriptor)    
    pp.pprint(camera_pose)
    
    cntr = 0
    while True:
        new_ad_placement_descriptor, new_camera_pose = mutate_ad_placement_descriptor_and_camera_pose(
            ad_placement_descriptor=ad_placement_descriptor,
            camera_pose=camera_pose,
        )


        from_to_mapping_array_f64 = render_then_get_from_to_mapping_array(
            camera_pose=new_camera_pose,
            ad_placement_descriptor=ad_placement_descriptor,
            texture_rgba_np_f32 = texture_rgba_np_f32,
            original_rgb_np_u8=original_rgb_np_u8,
            mask_for_regression_hw_np_u8=mask_for_regression_hw_np_u8,
            use_linear_light=False,
        )

        corr = calculate_metrics(
            from_to_mapping_array_f64=from_to_mapping_array_f64
        )
        if corr > best_corr:
            best_corr = corr
            ad_placement_descriptor = new_ad_placement_descriptor   
            camera_pose = new_camera_pose
            print(f"Improved to {best_corr=}")
            pp.pprint(ad_placement_descriptor)    
            pp.pprint(camera_pose)
        cntr += 1
        if cntr >= max_iters:
            break

    return ad_placement_descriptor, camera_pose
    
