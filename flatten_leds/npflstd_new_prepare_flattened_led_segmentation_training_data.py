from prii import (
     prii
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from decide_what_to_flatten import (
     decide_what_to_flatten
)
from ffst_flatten_for_segmentation_training_implementation2 import (
     ffst_flatten_for_segmentation_training_implementation2
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
from get_world_coordinate_descriptors_of_ad_placements import (
     get_world_coordinate_descriptors_of_ad_placements
)
import numpy as np


def pflstd_prepare_flattened_led_segmentation_training_data():
    rip_height = 256

    ad_placement_descriptors = get_world_coordinate_descriptors_of_ad_placements(
        clip_id="bos-mia-2024-04-21-mxf",
        with_floor_as_giant_ad=False,
        overcover_by=0.2
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

    # we want them to all have a width divisible by 32
    # since it is cookie-cuttered into 256x256 patches,
    # the exact width may not matter
    rip_width = int(np.round(rip_height * ad_width / ad_height / 32)) * 32

    records = decide_what_to_flatten()

    for record in records:
        
        # clip_id = record["clip_id"]
        # frame_index = record["frame_index"]

        camera_pose = record["camera_pose"]

        original_file_path = record["original_file_path"]
        mask_file_path = record["mask_file_path"]

        save_original_file_path = record["save_original_file_path"]
        save_relevance_file_path = record["save_relevance_file_path"]
        save_mask_file_path = record["save_mask_file_path"]

      
      
        
        mask_hw_np_u8 = open_alpha_channel_image_as_a_single_channel_grayscale_image(
            abs_file_path=mask_file_path
        )

        original_rgb_hwc_np_u8 = open_as_rgb_hwc_np_u8(
            image_path=original_file_path
        )

        # camera_pose = get_camera_pose_from_clip_id_and_frame_index(
        #     clip_id=clip_id,
        #     frame_index=frame_index,
        # )

       
        
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

        # prii(flattened_rgb)
        # prii(visibility_mask)
        # prii(flattened_mask)

        write_rgb_np_u8_to_png(
            rgb_hwc_np_u8=flattened_rgb,
            out_abs_file_path=save_original_file_path
        )

        write_grayscale_hw_np_u8_to_png(
            grayscale_hw_np_u8=visibility_mask,
            out_abs_file_path=save_relevance_file_path
        )

        write_grayscale_hw_np_u8_to_png(
            grayscale_hw_np_u8=flattened_mask,
            out_abs_file_path=save_mask_file_path
        )
        
        print(save_original_file_path)
        print(save_relevance_file_path)
        print(save_mask_file_path)

    print(f"Successfully flattened {len(records)}")

if __name__ == "__main__":
    pflstd_prepare_flattened_led_segmentation_training_data()