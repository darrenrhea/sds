from get_flat_original_path import (
     get_flat_original_path
)
from get_visibility_mask_path import (
     get_visibility_mask_path
)
from write_grayscale_hw_np_u8_to_png import (
     write_grayscale_hw_np_u8_to_png
)
from ffsi_flatten_for_segmentation_inference_implementation2 import (
     ffsi_flatten_for_segmentation_inference_implementation2
)
from write_rgb_hwc_np_u8_to_png import (
     write_rgb_hwc_np_u8_to_png
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


def flatten_clip_id_from_a_to_b(
    clip_id: str,
    first_frame_index: int,
    last_frame_index: int,
    step: int,
    rip_height: int,
    rip_width: int,
    skip_already_done: bool = False
):
    """
    TODO: Make this able to handle several boards?
    Or maybe the caller should just call this function several times?
    """
    assert isinstance(clip_id, str)
    assert isinstance(first_frame_index, int)
    assert isinstance(last_frame_index, int)
    assert isinstance(step, int)
    assert step > 0
    assert isinstance(rip_height, int)
    assert isinstance(rip_width, int)
    
    ad_placement_descriptors = get_world_coordinate_descriptors_of_ad_placements(
        clip_id=clip_id,
        with_floor_as_giant_ad=False,
        overcover_by=0.0
    )

    assert len(ad_placement_descriptors) == 1
    ad_placement_descriptor = ad_placement_descriptors[0]

    board_id = "board0"
    # pp.pprint(ad_placement_descriptor)

    ad_origin = ad_placement_descriptor.origin
    u = ad_placement_descriptor.u
    v = ad_placement_descriptor.v
    ad_height = ad_placement_descriptor.height
    ad_width = ad_placement_descriptor.width

    # we want them to all have the same width as well
    # rip_width = int(np.round(rip_height * ad_width / ad_height / 32)) * 32

    for frame_index in range(first_frame_index, last_frame_index + 1, step):        
        # print(f"flattening {clip_id=} {frame_index=}")
        # Determine where to save the flattened image and the visibility mask / onscreen mask:
        flattened_original_file_path = get_flat_original_path(
            clip_id=clip_id,
            frame_index=frame_index,
            board_id=board_id,
            rip_width=rip_width,
            rip_height=rip_height,
        )

        visibility_mask_file_path = get_visibility_mask_path(
            clip_id=clip_id,
            frame_index=frame_index,
            board_id=board_id,
            rip_width=rip_width,
            rip_height=rip_height,
        )

        if (
            skip_already_done
            and
            flattened_original_file_path.exists()
            and
            visibility_mask_file_path.exists()
        ):
            print(f"{flattened_original_file_path} already exists, skipping")
            print(f"{visibility_mask_file_path} already exists, skipping")
            continue
    
        original_rgb_hwc_np_u8 = get_original_frame_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index,
        )

        camera_pose = get_camera_pose_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index,
        )

       
        # H = get_homography_for_flattening(
        #     photograph_height_in_pixels=photograph_height_in_pixels,
        #     photograph_width_in_pixels=photograph_width_in_pixels,
        #     camera_pose=camera_pose,
        #     ad_origin=ad_origin,
        #     u=u,
        #     v=v,
        #     ad_height=ad_height,
        #     ad_width=ad_width,
        #     rip_height=rip_height,
        #     rip_width=rip_width,
        # )


        # led_board_index = 0
        # save_homography_file_path = out_dir / f"{clip_id}_{frame_index:06d}_homography_{led_board_index}.json"
        # save_homography_and_rip_size_as_json(
        #     H=H,
        #     rip_height=rip_height,
        #     rip_width=rip_width,
        #     out_path=save_homography_file_path,
        # )


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
       
        

        # prii(flattened_rgb)
        flattened_original_file_path.parent.mkdir(exist_ok=True, parents=True)
        write_rgb_hwc_np_u8_to_png(
            rgb_hwc_np_u8=flattened_rgb,
            out_abs_file_path=flattened_original_file_path
        )
        print(f"pri {flattened_original_file_path}")

        write_grayscale_hw_np_u8_to_png(
            grayscale_hw_np_u8=visibility_mask,
            out_abs_file_path=visibility_mask_file_path
        )

        print(f"pri {visibility_mask_file_path}")


       
