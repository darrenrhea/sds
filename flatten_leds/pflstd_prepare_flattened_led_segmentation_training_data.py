from prii import (
     prii
)
from ffst_flatten_for_segmentation_training_implementation2 import (
     ffst_flatten_for_segmentation_training_implementation2
)
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from ffst_flatten_for_segmentation_training import (
     ffst_flatten_for_segmentation_training
)
from write_grayscale_hw_np_u8_to_png import (
     write_grayscale_hw_np_u8_to_png
)
from write_rgb_np_u8_to_png import (
     write_rgb_np_u8_to_png
)
from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image
)
from get_approved_annotations_from_these_repos import (
     get_approved_annotations_from_these_repos
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
import numpy as np


def pflstd_prepare_flattened_led_segmentation_training_data():
    rip_height = 256
    # We might want them to all have a width divisible by 32
    # since it is cookie-cuttered into 512x256 patches,
    # the exact width may not matter
    # rip_width = int(np.round(rip_height * ad_width / ad_height / 32)) * 32

    rip_width = 4268
    shared_dir = get_the_large_capacity_shared_directory()

    clip_id_to_use_to_determine_led_board_locations = "slgame1" # bos-mia-2024-04-21-mxf"
    out_dir = shared_dir / "flattened_training_data/summer_league_2024"
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"{out_dir=!s}")

    repo_ids_to_use =  [
        # these first two have the lighting condition:
        # "bay-zal-2024-03-15-mxf-yadif_led",
        # "bay-mta-2024-03-22-mxf_led",  # YELLOW UNIFORMS!
        # "maccabi_fine_tuning",  # YELLOW UNIFORMS!
        # "maccabi1080i_led",  # YELLOW UNIFORMS!
        # "skweek_led",
        # "denizbank_led",
        # "munich1080i_led",
        # "bos-mia-2024-04-21-mxf_led",
        # "bos-dal-2024-06-09-mxf_led",
        # "dal-bos-2024-06-12-mxf_led",
        # "dal-bos-2024-01-22-mxf_led",
        "slgame1_led",
        "slday8game1_led",
    ]

    # 389500 didn't get wide enough on the left:
    # 391000 has something weird going on, there was an error in the area of a quadrilateral calculation.
    # 442500 is barely visible, similarly 614500, similarly 617500
    approved_annotations = get_approved_annotations_from_these_repos(
        repo_ids_to_use=repo_ids_to_use
    )


    approved_annotations = [
        x for x in approved_annotations
        # if x["frame_index"] == 112000
        #if x["clip_id"] == "slday8game1"
    ]
    
    # pp.pprint(approved_annotations)
    # np.random.shuffle(approved_annotations)
    
    ad_placement_descriptors = get_world_coordinate_descriptors_of_ad_placements(
        clip_id=clip_id_to_use_to_determine_led_board_locations,
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


    bl = ad_origin
    br = ad_origin + ad_width * u
    tl = ad_origin + ad_height * v
    tr = ad_origin + ad_height * v + ad_width * u

    print(f"{ad_origin=}")
    print(f"{u=}")
    print(f"{v=}")
    print(f"{ad_height=}")
    print(f"{ad_width=}")

    print(f"{bl=}")
    print(f"{br=}")
    print(f"{tl=}")
    print(f"{tr=}")

   
    

    for approved_annotation in approved_annotations:        
        clip_id = approved_annotation["clip_id"]
        frame_index = approved_annotation["frame_index"]
        mask_file_path = approved_annotation["mask_file_path"]

        print(f"{clip_id=} {frame_index=}")
        
        mask_hw_np_u8 = open_alpha_channel_image_as_a_single_channel_grayscale_image(
            abs_file_path=mask_file_path
        )

        original_rgb_hwc_np_u8 = get_original_frame_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index,
        )

        camera_pose = get_camera_pose_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index,
        )

       
        
        (
            flattened_rgb,
            visibility_mask,
            flattened_mask
        ) = ffst_flatten_for_segmentation_training_implementation2(
            ad_origin=ad_origin,
            u=u,
            v=v,
            ad_height=ad_height,
            ad_width=ad_width,
            camera_pose=camera_pose,
            original_rgb_hwc_np_u8=original_rgb_hwc_np_u8,
            mask_hw_np_u8=mask_hw_np_u8,
            rip_height=rip_height,
            rip_width=rip_width,
        )

        save_flattened_original_file_path = out_dir / f"{clip_id}_{frame_index:06d}_original.jpg"
        save_visibility_mask_file_path = out_dir / f"{clip_id}_{frame_index:06d}_relevance.png"
        save_flattened_mask_file_path = out_dir / f"{clip_id}_{frame_index:06d}_nonfloor.png"

        # prii(flattened_rgb, out=save_flattened_original_file_path)
        # prii(visibility_mask, out=save_visibility_mask_file_path)
        # prii(flattened_mask, out=save_flattened_mask_file_path)
        write_rgb_np_u8_to_png(
            rgb_hwc_np_u8=flattened_rgb,
            out_abs_file_path=save_flattened_original_file_path
        )

        write_grayscale_hw_np_u8_to_png(
            grayscale_hw_np_u8=visibility_mask,
            out_abs_file_path=save_visibility_mask_file_path
        )

        write_grayscale_hw_np_u8_to_png(
            grayscale_hw_np_u8=flattened_mask,
            out_abs_file_path=save_flattened_mask_file_path
        )

        
if __name__ == "__main__":
    pflstd_prepare_flattened_led_segmentation_training_data()