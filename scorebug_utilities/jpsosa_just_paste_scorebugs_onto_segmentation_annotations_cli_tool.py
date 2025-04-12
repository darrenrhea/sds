import pprint
import random
import sys
from color_print_json import color_print_json
from get_random_scorebug_rgba_np_u8 import (
     get_random_scorebug_rgba_np_u8
)
from open_as_rgba_hwc_np_u8 import (
     open_as_rgba_hwc_np_u8
)
from feathered_paste_for_images_of_the_same_size import (
     feathered_paste_for_images_of_the_same_size
)
from open_sha256s_as_rgba_hwc_np_u8s import open_sha256s_as_rgba_hwc_np_u8s
from osofpsaj_open_sha256_or_file_path_str_as_json import osofpsaj_open_sha256_or_file_path_str_as_json
from upload_file_path_to_s3_file_uri import (
     upload_file_path_to_s3_file_uri
)
from print_green import (
     print_green
)
from store_file_by_sha256_in_s3 import (
     store_file_by_sha256_in_s3
)
from download_the_files_with_these_sha256s import (
     download_the_files_with_these_sha256s
)
from make_rgba_hwc_np_u8_from_rgb_and_alpha import (
     make_rgba_hwc_np_u8_from_rgb_and_alpha
)
from make_python_internalized_video_frame_annotation import (
     make_python_internalized_video_frame_annotation
)
import argparse
import textwrap
import time
from blur_rgb_hwc_np_linear_f32 import (
     blur_rgb_hwc_np_linear_f32
)
from write_rgb_hwc_np_u8_to_jpg import (
     write_rgb_hwc_np_u8_to_jpg
)
from convert_linear_f32_to_u8 import (
     convert_linear_f32_to_u8
)
from convert_u8_to_linear_f32 import (
     convert_u8_to_linear_f32
)
from pathlib import Path
from insert_quads_into_camera_posed_image_behind_mask import (
     insert_quads_into_camera_posed_image_behind_mask
)
from prii import (
     prii
)
import numpy as np
import better_json as bj
from write_rgba_hwc_np_u8_to_png import (
     write_rgba_hwc_np_u8_to_png
)


def jpsosa_just_paste_scorebugs_onto_segmentation_annotations_cli_tool():
    """
    Just add random scorebugs to the images in the directory.
    """
    argp = argparse.ArgumentParser(
        description="mffnfabianfu_make_fake_floor_not_floor_annotations_by_inserting_a_new_floor_underneath",
        usage=textwrap.dedent(
            """\
            


            jpsosa_just_paste_scorebugs_onto_segmentation_annotations \\
            --in_dir ~/cleandata \\
            --scorebug_config 9122a594ad39bf7d194597ecd3dc7867b395e95617aaa35ed7fdb100c471ace4 \\
            --out_dir ~/withscorebugs \\
            --print_in_iterm2

            

            """
        )
    )
    argp.add_argument(
        "--in_dir",
        help="The directory where you want to save the fake segmentation annotation images",
        required=True,
    )
    argp.add_argument(
        "--scorebug_config",
        help="Either a json5 file configuring the scorebugs or the sha256 thereof.",
        required=True,
    )
    argp.add_argument(
        "--out_dir",
        help="The directory where you want to save the fake segmentation annotation images",
        required=True,
    )
    argp.add_argument(
        "--print_in_iterm2",
        help="print the images in iterm2.",
        action="store_true",
    )

    
    opt = argp.parse_args()
    scorebug_config_json_file_or_sha256 = opt.scorebug_config
    print_in_iterm2 = opt.print_in_iterm2
    out_dir = Path(opt.out_dir)
    in_dir = Path(opt.in_dir).resolve()
    in_dir.mkdir(parents=True, exist_ok=True)
    print(f"Taking (hopefully scorebug-free) annotations from {in_dir=}")

    
   
  
    scorebug_config = osofpsaj_open_sha256_or_file_path_str_as_json(
        sha256_or_local_file_path_str=scorebug_config_json_file_or_sha256
    )
    print("We will protect these scorebugs:")
    color_print_json(scorebug_config)
        
    scorebug_sha256s = []
    for scorebug in scorebug_config["scorebugs"]:
        scorebug_sha256s.append(scorebug["scorebug_sha256"])
     
    # unique-ify:
    scorebug_sha256s = list(set(scorebug_sha256s))
    
    # download the files:
    download_the_files_with_these_sha256s(
        sha256s_to_download=scorebug_sha256s,
        max_workers=10,
        verbose=True,
    )
    sys.exit(0)

    scorebug_rgbas = open_sha256s_as_rgba_hwc_np_u8s(
        sha256s=scorebug_sha256s
    )
    
    out_dir.mkdir(exist_ok=True, parents=True)
     print(f"Writing to directory {out_dir=}")
    
    
    num_laps = 2
    start_time = time.time()
    num_completed = 0
    out_of = len(local_file_pathed_annotations) * num_laps
    for _ in range(num_laps):
        for local_file_pathed_annotation in local_file_pathed_annotations:
            work_item = make_python_internalized_video_frame_annotation(
                local_file_pathed_annotation=local_file_pathed_annotation
            )
            clip_id = work_item["clip_id"]
            frame_index = work_item["frame_index"]
            original_rgb_np_u8 = work_item["original_rgb_np_u8"]
            floor_not_floor_hw_np_u8 = work_item["floor_not_floor_hw_np_u8"]
            camera_pose = work_item["camera_pose"]

            assert (
                camera_pose.f > 0
            ), f"camera_pose.f is 0.0 for {clip_id=} {frame_index=} i.e. file {work_item['camera_pose_json_file_path']}"
            
            assert (
                camera_pose.loc[1] < 180.0
            ), f"camera loc y is > 180.0 for {clip_id=} {frame_index=} i.e. file {work_item['camera_pose_json_file_path']}"

            dct = get_a_floor_texture_with_random_shadows_and_lights(
                floor_id=floor_id,
                asset_repos_dir=asset_repos_dir,
            )
            color_corrected_texture_rgba_np_linear_f32 = dct["color_corrected_texture_rgba_np_linear_f32"]
            floor_placement_descriptor = dct["floor_placement_descriptor"]
            del dct

            textured_ad_placement_descriptors = []
            floor_placement_descriptor.texture_rgba_np_f32 = color_corrected_texture_rgba_np_linear_f32
            textured_ad_placement_descriptors.append(floor_placement_descriptor)
            assert len(textured_ad_placement_descriptors) == 1
            
            # move to linear light:
            original_rgb_np_linear_f32 = convert_u8_to_linear_f32(
                original_rgb_np_u8
            )
            
            assert floor_not_floor_hw_np_u8  is not None
            assert isinstance(floor_not_floor_hw_np_u8 , np.ndarray)
            
        
            inserted_with_color_correction_linear_f32 = insert_quads_into_camera_posed_image_behind_mask(
                anti_aliasing_factor=2,
                use_linear_light=True, # this should be true, because of all the averaging processes going on.
                original_rgb_np_linear_f32=original_rgb_np_linear_f32,
                camera_pose=camera_pose,
                mask_hw_np_u8=floor_not_floor_hw_np_u8,
                textured_ad_placement_descriptors=textured_ad_placement_descriptors,
            )
            assert inserted_with_color_correction_linear_f32.dtype == np.float32
            assert inserted_with_color_correction_linear_f32.shape[2] == 3

            if np.random.randint(0, 2) == 0:
                blended_rgb_hwc_np_linear_f32 = blur_rgb_hwc_np_linear_f32(
                    rgb_hwc_np_linear_f32=inserted_with_color_correction_linear_f32,
                    sigma_x=0.5,
                    sigma_y=0.5,
                )
            else:
                blended_rgb_hwc_np_linear_f32 = inserted_with_color_correction_linear_f32

            blended_rgb_hwc_np_linear_u8 = convert_linear_f32_to_u8(
                blended_rgb_hwc_np_linear_f32
            )

            rgba_hwc_np_u8 = make_rgba_hwc_np_u8_from_rgb_and_alpha(
                rgb=blended_rgb_hwc_np_linear_u8,
                alpha=floor_not_floor_hw_np_u8,
            )

            do_paste_cutouts = True
            if do_paste_cutouts:
                # choose how many of each kind somehow:
                cutout_kind_to_num_cutouts_to_paste = dict(
                    player=np.random.randint(0, 12),
                    referee=np.random.randint(0, 3),
                    coach=np.random.randint(0, 3),
                    ball=np.random.randint(0, 10),
                    led_screen_occluding_object=np.random.randint(0, 2),
                )
                league = "nba"
                pasted_rgba_np_u8 = paste_multiple_cutouts_onto_one_camera_posed_segmentation_annotation(
                    league=league,
                    context_id=context_id,
                    cutouts_by_kind=cutouts_by_kind,
                    rgba_np_u8=rgba_hwc_np_u8,  # this is not violated by this procedure.
                    camera_pose=camera_pose,  # to get realistics locations and sizes we need to know the camera pose.
                    cutout_kind_to_transform=cutout_kind_to_transform, # what albumentations augmentation to use per kind of cutout
                    cutout_kind_to_num_cutouts_to_paste=cutout_kind_to_num_cutouts_to_paste
                )

            else:
                print("we are not pasting cutouts this time.")
                pasted_rgba_np_u8 = rgba_hwc_np_u8

            annotation_id = f"{clip_id}_{frame_index:06d}"
            rid = np.random.randint(0, 1_000_000_000_000_000)
            fake_annotation_id = f"{annotation_id}_fake{rid:015d}"

            fake_original_path = out_dir / f"{fake_annotation_id}_original.jpg"
            fake_mask_path = out_dir / f"{fake_annotation_id}_nonfloor.png"

            scorebug_rgba = random.choice(scorebug_rgbas)

            with_scorebug_rgb = feathered_paste_for_images_of_the_same_size(
                bottom_layer_color_np_uint8=pasted_rgba_np_u8[:, :, 0:3],
                top_layer_rgba_np_uint8=scorebug_rgba,
            )

            with_scorebug_rgba_np_u8 = np.zeros(
                (pasted_rgba_np_u8.shape[0], pasted_rgba_np_u8.shape[1], 4),
                dtype=np.uint8
            )
            with_scorebug_rgba_np_u8[:, :, 0:3] = with_scorebug_rgb
            with_scorebug_rgba_np_u8[:, :, 3] = np.maximum(
                scorebug_rgba[:, :, 3],
                pasted_rgba_np_u8[:, :, 3],
            )

            write_rgb_hwc_np_u8_to_jpg(
                rgb_hwc_np_u8=with_scorebug_rgba_np_u8[:, :, 0:3],
                out_abs_file_path=fake_original_path,
                verbose=True
            )

            # write_grayscale_hw_np_u8_to_png(
            #     grayscale_hw_np_u8=floor_not_floor_hw_np_u8 ,
            #     out_abs_file_path=fake_mask_path,
            #     verbose=False,
            # )

            write_rgba_hwc_np_u8_to_png(
                rgba_hwc_np_u8=with_scorebug_rgba_np_u8,
                out_abs_file_path=fake_mask_path,
                verbose=False
            )
            
            # many different computers can be generating and uploading to s3 at the same time:
            if records_s3_dir_uri is not None:
                # store products in s3
                fake_original_sha256 = store_file_by_sha256_in_s3(
                    fake_original_path
                )

                fake_mask_sha256 = store_file_by_sha256_in_s3(
                    fake_mask_path
                )

                record = dict(
                    clip_id=clip_id,
                    frame_index=frame_index,
                    fake_original_sha256=fake_original_sha256,
                    fake_mask_sha256=fake_mask_sha256,
                )
                
                record_file_path = records_dir_path / f"{fake_annotation_id}.json"

                bj.dump(
                    obj=record,
                    fp=record_file_path
                )

                print_green(f"pri {record_file_path}")
                
                record_s3_file_uri = f"{records_s3_dir_uri}{fake_annotation_id}.json"
                
                upload_file_path_to_s3_file_uri(
                    file_path=record_file_path,
                    s3_file_uri=record_s3_file_uri,
                    expected_hexidecimal_sha256=None,
                    verbose=False,
                )

                
            # write_rgb_and_alpha_to_png(
            #     rgb_hwc_np_u8=pasted_rgba_np_u8,
            #     alpha_hw_np_u8=floor_not_floor_hw_np_u8,
            #     out_abs_file_path=fake_mask_path,
            #     verbose=False
            # )

            if print_in_iterm2:
                # prii_linear_f32(
                #     x=blended_rgb_hwc_np_linear_f32,
                #     caption="this is the final color corrected result",
                # )
                
                prii(
                    x=fake_original_path,
                    caption="this is the synthetic / fake original:",
                )

                prii(
                    x=fake_mask_path,
                    caption="this is the synthetic / fake mask:",
                )

                prii(
                    x=original_rgb_np_u8,
                    caption="this is the original frame:",
                )
            
            num_completed += 1
            duration = time.time() - start_time
            print(f"So far, it has been {duration/60} minutes")
            seconds_per_item = duration / num_completed
            estimated_remaining = (out_of - num_completed) * seconds_per_item
            print(f"completed {num_completed} / {out_of}")
            print(f"Estimated time remaining: {estimated_remaining/60} minutes.")




