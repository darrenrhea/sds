from faiaccpb_flatten_and_infer_and_compose_camera_pose_based import (
     faiaccpb_flatten_and_infer_and_compose_camera_pose_based
)
from prii import (
     prii
)
from grirosffmi_get_ram_in_ram_out_segmenter_from_final_model_id import (
     grirosffmi_get_ram_in_ram_out_segmenter_from_final_model_id
)
import argparse
import textwrap
import os
from uttfmiopm_union_together_the_flat_masks_into_one_perspective_mask import (
     uttfmiopm_union_together_the_flat_masks_into_one_perspective_mask
)
from prii_rgb_and_alpha import (
     prii_rgb_and_alpha
)
from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from get_flat_mask_path import (
     get_flat_mask_path
)
from get_world_coordinate_descriptors_of_ad_placements import (
     get_world_coordinate_descriptors_of_ad_placements
)
from write_grayscale_hw_np_u8_to_png import (
     write_grayscale_hw_np_u8_to_png
)
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image
)




def priseg_cli_tool():

    argp = argparse.ArgumentParser(
        description="show the segmentation inference for a model of flatten then infer type",
        usage=textwrap.dedent(
            """
            export m=slflatpatch30
            export c=slday8game1
            priseg 0
            # or, override the clip_id and final_model_id:
            # priseg -m slflatpatch30 -c slday3game1 0
            """
        )
    )

    argp.add_argument(
        "frame_indices",
        nargs="+",
        type=int,
        help="a list of frame indices to infer. The final_model_id is the environment variable m and the clip_id is the environment variable c",
    )
    
    args = argp.parse_args()
    
    frame_indices = args.frame_indices

    try:  
        final_model_id = os.environ["m"]
    except KeyError:
        raise ValueError("please set the environment variable m to the final_model_id like:\n\nexport m=slflatpatch30")
    
    try:
        clip_id = os.environ["c"]
    except KeyError:
        raise ValueError("please set the environment variable c to the clip_id like:\n\nexport c=slday8game1")

    assert (
        clip_id not in ["brewcub"]
    ), "This does not work for baseball, because it is camera-pose / lens distortion based."

    print(f"For clip_id {clip_id} and final_model_id {final_model_id}, we will show the segmentation for frame_indices {frame_indices}")
    
    ram_in_ram_out_segmenter = grirosffmi_get_ram_in_ram_out_segmenter_from_final_model_id(
        final_model_id=final_model_id
    )

    board_ids = [
        "board0",
    ]
    board_id_to_rip_height = {
         "board0": 256
    }
    
    board_id_rip_width = {
        "board0": 4268
    }

    rip_height = 256
    rip_width = 4268

    
    photograph_height_in_pixels = 1080
    photograph_width_in_pixels = 1920
   
    
    for frame_index in frame_indices:
        print(f"Reconstituting {frame_index=}")

        rgb_hwc_np_u8 = get_original_frame_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index
        )
        prii(rgb_hwc_np_u8, caption="rgb_hwc_np_u8:")
        
        camera_pose = get_camera_pose_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index,
        )

        ad_placement_descriptors = get_world_coordinate_descriptors_of_ad_placements(
            clip_id=clip_id,
            with_floor_as_giant_ad=False,
            overcover_by=0.0
        )

        flat_inference_file_path = get_flat_mask_path(
            clip_id=clip_id,
            frame_index=frame_index,
            final_model_id=model_id,
            rip_width=rip_width,
            rip_height=rip_height,
            board_id="board0"
        )
        
        flat_mask_hw_np_u8 = open_alpha_channel_image_as_a_single_channel_grayscale_image(
            abs_file_path=flat_inference_file_path
        )

        assert len(ad_placement_descriptors) == 1
        pairs_of_flat_mask_and_ad_descriptor = [
            (flat_mask_hw_np_u8, ad_placement_descriptors[0]),
        ]

        mask_hw_np_u8 = uttfmiopm_union_together_the_flat_masks_into_one_perspective_mask(
            pairs_of_flat_mask_and_ad_descriptor=pairs_of_flat_mask_and_ad_descriptor,
            camera_pose=camera_pose,
            photograph_height_in_pixels=photograph_height_in_pixels,
            photograph_width_in_pixels=photograph_width_in_pixels,
        )

        prii(mask_hw_np_u8, caption="mask_hw_np_u8")

        prii_rgb_and_alpha(
            rgb_hwc_np_u8=rgb_hwc_np_u8,
            alpha_hw_np_u8=mask_hw_np_u8,
        )

        
