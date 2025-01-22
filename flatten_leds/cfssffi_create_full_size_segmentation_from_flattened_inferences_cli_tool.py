from uttfmiopm_union_together_the_flat_masks_into_one_perspective_mask import (
     uttfmiopm_union_together_the_flat_masks_into_one_perspective_mask
)
from prii_rgb_and_alpha import (
     prii_rgb_and_alpha
)
from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)
import textwrap
from prii import prii
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from get_flat_mask_path import (
     get_flat_mask_path
)
from get_world_coordinate_descriptors_of_ad_placements import (
     get_world_coordinate_descriptors_of_ad_placements
)
import argparse
from write_grayscale_hw_np_u8_to_png import (
     write_grayscale_hw_np_u8_to_png
)
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image
)



def cfssffi_create_full_size_segmentation_from_flattened_inferences_cli_tool():
    argp = argparse.ArgumentParser(
        description=textwrap.dedent(
            """\
            """
        ),
        usage=textwrap.dedent(
            """\
            cfssffi_create_full_size_segmentation_from_flattened_inferences -c slday8game1 -m slflatpatch30 -a 0 -b 10
            cfssffi_create_full_size_segmentation_from_flattened_inferences -c slday8game1 -m slflatpatch30 -a 0 -b 10
            """
        )
    )
    argp.add_argument(
        "-c", "--clip_id",
        required=True,
        help="The clip_id to flatten."
    )
    argp.add_argument(
        "-m", "--final_model_id",
        required=True,
        help="The final_model_id to used for ad board segmentation."
    )
    argp.add_argument(
        "-a", "--first_frame_index",
        required=True,
        type=int,
        help="The first frame index to flatten."
    )
    argp.add_argument(
        "-b", "--last_frame_index",
        required=True,
        type=int,
        help="The last frame index to flatten."
    )

    argp.add_argument(
        "-s", "--frame_step",
        required=False,
        default=1,
        type=int,
        help="how many to increment the frame_index by"
    )

    args = argp.parse_args()
    clip_id = args.clip_id
    first_frame_index = args.first_frame_index
    last_frame_index = args.last_frame_index
    model_id = args.final_model_id
    frame_step = args.frame_step

    shared_dir = get_the_large_capacity_shared_directory()
    inferences_dir = shared_dir / "inferences"

    rip_height = 256
    rip_width = 4268

    
    photograph_height_in_pixels = 1080
    photograph_width_in_pixels = 1920
   
    
    for frame_index in range(first_frame_index, last_frame_index + 1, frame_step): 
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
        
        save = False

        if save:
            save_mask_file_path =  inferences_dir / f"{clip_id}_{frame_index:06d}_{model_id}.png"

            

            write_grayscale_hw_np_u8_to_png(
                grayscale_hw_np_u8=mask_hw_np_u8,
                out_abs_file_path=save_mask_file_path
            )
            print(f"pri {save_mask_file_path}")
        

  
