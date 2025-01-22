from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from write_grayscale_hw_np_u8_to_png import (
     write_grayscale_hw_np_u8_to_png
)
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from make_relevance_mask_for_led_boards import (
     make_relevance_mask_for_led_boards
)
from get_world_coordinate_descriptors_of_ad_placements import (
     get_world_coordinate_descriptors_of_ad_placements
)
from get_approved_annotations_from_these_repos import (
     get_approved_annotations_from_these_repos
)
import shutil



def add_multiple_copies_of_the_training_data():
    # shared_dir = Path("/shared").expanduser()
    shared_dir = get_the_large_capacity_shared_directory()
    # shared_dir = Path("~/a").expanduser()


    dst_dir = shared_dir / "fake_game5" / "human_annotated_multiplied"
    dst_dir.mkdir(exist_ok=True)
    temp_dir = shared_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    # pairs_of_repo_name_and_num_approved = [
    #     ("bay-czv-2024-03-01_led", 20), # same
    #     ("bay-efs-2023-12-20_led", 20),  # should be 5 more here, pul it
    #     ("bay-mta-2024-03-22-mxf_led", 77),
    #     ("bay-mta-2024-03-22-part1-srt_led", 46),  # the new stuff
    #     ("bay-zal-2024-03-15-yt_led", 37), # same
    #     ("maccabi_fine_tuning", 117),  # same
    #     ("maccabi1080i_led", 49),
    #     ("munich1080i_led", 221),
    # ]

    repo_ids_to_use = [
        # white away team stuff:
        # "bay-zal-2024-03-15-mxf-yadif_led",
        # "munich1080i_led",
        # "bay-czv-2024-03-01_led",
        # "bay-efs-2023-12-20_led",
        # "bay-mta-2024-03-22-mxf_led",
        # "bay-mta-2024-03-22-part1-srt_led",
        # "bay-zal-2024-03-15-yt_led",
        # "maccabi_fine_tuning",
        # "maccabi1080i_led",
        "bos-mia-2024-04-21-mxf_led",
        "bos-dal-2024-06-09-mxf_led",
        # "dal-bos-2024-01-22-mxf_led",
        # "dal-bos-2024-06-12-mxf_led",
    ]

    actual_annotations = get_approved_annotations_from_these_repos(
        repo_ids_to_use=repo_ids_to_use
    )
    
    # this is only to get the ad placements in 3D:
    clip_id = "bos-mia-2024-04-21-mxf"

    ad_placement_descriptors = get_world_coordinate_descriptors_of_ad_placements(
        clip_id=clip_id,
        with_floor_as_giant_ad=False,
        overcover_by=0.4,
    )

    for actual_annotation in actual_annotations:
        mask_file_path = actual_annotation["mask_file_path"]
        assert mask_file_path.is_file()
        original_file_path = actual_annotation["original_file_path"]
        assert original_file_path.is_file()
        frame_index = actual_annotation["frame_index"]
        clip_id = actual_annotation["clip_id"]

        camera_pose = get_camera_pose_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index
        )

        camera_posed_original_video_frame = dict(
            original_file_path=original_file_path,
            frame_index=frame_index,
            clip_id=clip_id,
            camera_pose=camera_pose,
        )

        relevance_mask = make_relevance_mask_for_led_boards(
            camera_posed_original_video_frame=camera_posed_original_video_frame,
            ad_placement_descriptors=ad_placement_descriptors
        )
        relevance_path = temp_dir / f"{mask_file_path.name[:-13]}_relevance.png"
        print(relevance_path)
        write_grayscale_hw_np_u8_to_png(
            out_abs_file_path=relevance_path,
            grayscale_hw_np_u8=relevance_mask,
        )
        

        src_file_paths = [
            mask_file_path,
            original_file_path,
            relevance_path,
        ]
        for src_file_path in src_file_paths:
            for copy_index in range(7):
                variant_name = f"anothercopy-{copy_index:02d}-{src_file_path.name}"
                print(f"{variant_name=}")
                dst_file_path = dst_dir / variant_name
                print(f"Copying\n{src_file_path}\nto\n{dst_file_path}\n")
                shutil.copy(
                    src=src_file_path,
                    dst=dst_file_path,
                )


if __name__ == "__main__":
    add_multiple_copies_of_the_training_data()
  


