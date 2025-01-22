from ffsi_flatten_for_segmentation_inference_implementation2 import (
     ffsi_flatten_for_segmentation_inference_implementation2
)
from get_world_coordinate_descriptors_of_ad_placements import (
     get_world_coordinate_descriptors_of_ad_placements
)
from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)


def fasf_flatten_a_single_frame(
    clip_id: str,
    frame_index: int,
    rip_height: int,
    rip_width: int,
    board_id: str,
):
    """
    camera-pose based.
    TODO: Make this able to handle several boards?
    Or maybe the caller should just call this function several times?
    """
    assert isinstance(clip_id, str)
    assert isinstance(frame_index, int)
    assert isinstance(rip_height, int)
    assert isinstance(rip_width, int)
    assert isinstance(board_id, str)
    
    ad_placement_descriptors = get_world_coordinate_descriptors_of_ad_placements(
        clip_id=clip_id,
        with_floor_as_giant_ad=False,
        overcover_by=0.0
    )

    assert len(ad_placement_descriptors) == 1
    ad_placement_descriptor = ad_placement_descriptors[0]

    assert board_id == "board0", "Currently only board0 is supported"

    # pp.pprint(ad_placement_descriptor)

    ad_origin = ad_placement_descriptor.origin
    u = ad_placement_descriptor.u
    v = ad_placement_descriptor.v
    ad_height = ad_placement_descriptor.height
    ad_width = ad_placement_descriptor.width

    # we want them to all have the same width as well
    # rip_width = int(np.round(rip_height * ad_width / ad_height / 32)) * 32

    # print(f"flattening {clip_id=} {frame_index=}")
    # Determine where to save the flattened image and the visibility mask / onscreen mask:

    original_rgb_hwc_np_u8 = (
        get_original_frame_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index,
        )
    )

    camera_pose = get_camera_pose_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )

    (
        flattened_rgb,
        visibility_mask
    ) = ffsi_flatten_for_segmentation_inference_implementation2(
        original_rgb_hwc_np_u8=original_rgb_hwc_np_u8,
        camera_pose=camera_pose,
        ad_origin=ad_origin,
        u=u,
        v=v,
        ad_height=ad_height,
        ad_width=ad_width,
        rip_height=rip_height,
        rip_width=rip_width,
    )
    
    return (
        flattened_rgb,
        visibility_mask
    )
