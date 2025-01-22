from prii import prii
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from get_flat_mask_path import (
     get_flat_mask_path
)
from utsi_unflatten_the_segmentation_inferences_implementation2 import (
     utsi_unflatten_the_segmentation_inferences_implementation2
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



def cfssffi_create_full_size_segmentation_from_flattened_inferences():
    argp = argparse.ArgumentParser()
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

    args = argp.parse_args()
    clip_id = args.clip_id
    first_frame_index = args.first_frame_index
    last_frame_index = args.last_frame_index
    model_id = args.final_model_id

   
    shared_dir = get_the_large_capacity_shared_directory()
    inferences_dir = shared_dir / "inferences"

    rip_height = 256
    rip_width = 4268

    
    photograph_height_in_pixels = 1080
    photograph_width_in_pixels = 1920
   
    
    for frame_index in range(first_frame_index, last_frame_index+1): 
        print(f"Reconstituting {frame_index=}")
        camera_pose = get_camera_pose_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index,
        )

        ad_placement_descriptors = get_world_coordinate_descriptors_of_ad_placements(
            clip_id=clip_id,
            with_floor_as_giant_ad=False,
            overcover_by=0.0
        )

        assert len(ad_placement_descriptors) == 1
        ad_placement_descriptor = ad_placement_descriptors[0]


        # pp.pprint(ad_placement_descriptor)

        ad_origin = ad_placement_descriptor.origin
        u = ad_placement_descriptor.u
        v = ad_placement_descriptor.v
        ad_height = ad_placement_descriptor.height
        ad_width = ad_placement_descriptor.width

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

        mask_hw_np_u8 = utsi_unflatten_the_segmentation_inferences_implementation2(
            flat_mask_hw_np_u8=flat_mask_hw_np_u8,
            camera_pose=camera_pose,
            rip_height=rip_height,
            rip_width=rip_width,
            ad_origin=ad_origin,
            u=u,
            v=v,
            ad_height=ad_height,
            ad_width=ad_width,
            photograph_height_in_pixels=photograph_height_in_pixels,
            photograph_width_in_pixels=photograph_width_in_pixels,
        )

        # prii(mask_hw_np_u8, caption="mask_hw_np_u8")
        save_mask_file_path =  inferences_dir / f"{clip_id}_{frame_index:06d}_{model_id}.png"

        

        write_grayscale_hw_np_u8_to_png(
            grayscale_hw_np_u8=mask_hw_np_u8,
            out_abs_file_path=save_mask_file_path
        )
        print(f"pri {save_mask_file_path}")
        

  
if __name__ == "__main__":
    cfssffi_create_full_size_segmentation_from_flattened_inferences()