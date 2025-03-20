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
from group_cutouts_by_kind import (
     group_cutouts_by_kind
)
from get_cutouts import (
     get_cutouts
)
from paste_multiple_cutouts_onto_one_camera_posed_segmentation_annotation import (
     paste_multiple_cutouts_onto_one_camera_posed_segmentation_annotation
)
from get_a_floor_texture_with_random_shadows_and_lights import (
     get_a_floor_texture_with_random_shadows_and_lights
)
from get_valid_floor_ids import (
     get_valid_floor_ids
)
from gfpfwmbasoafoafps_get_file_path_from_what_might_be_a_sha256_of_a_file_or_a_file_path_str import (
     gfpfwmbasoafoafps_get_file_path_from_what_might_be_a_sha256_of_a_file_or_a_file_path_str
)
from is_sha256 import (
     is_sha256
)
from make_python_internalized_video_frame_annotation import (
     make_python_internalized_video_frame_annotation
)
from get_local_file_pathed_annotations import (
     get_local_file_pathed_annotations
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
from get_cutout_augmentation import (
     get_cutout_augmentation
)


      
def mffnfabianfu_make_fake_floor_not_floor_annotations_by_inserting_a_new_floor_underneath_cli_tool():
    """
    This inserts a color-corrected floor texture underneath the players etc. in a floor-not-floor annotated frame
    to make fake training data.
    """
    argp = argparse.ArgumentParser(
        description="mffnfabianfu_make_fake_floor_not_floor_annotations_by_inserting_a_new_floor_underneath",
        usage=textwrap.dedent(
            """\
            You may need to ask someone who knows what the sha256 of all the approved video frame annotations is,
            usually Darren.

            conda activate sds

            mffnfabianfu_make_fake_floor_not_floor_annotations_by_inserting_a_new_floor_underneath \\
            --floor_id 24-25_ALL_STAR \\
            --video_frame_annotations 37deb6dd165db2a0b1d1ea42ecffa1f1161656526ebc7b1fb0410f37718649b2 \\
            --out_dir ~/a/crap \\
            --records_out_dir ~/records \\
            --print_in_iterm2
    
            or you can use a local .json5 file like:

            rm -rf ~/a/crap/*

            mffnfabianfu_make_fake_floor_not_floor_annotations_by_inserting_a_new_floor_underneath \\
            --floor_id 24-25_ALL_STAR \\
            --video_frame_annotations ~/temp/my_video_annotations.json5 \\
            --out_dir ~/a/crap \\
            --print_in_iterm2
            
            where

            my_video_annotations.json5 is a JSON5 file of video_frame_annotations_metadata like:

            
            [
                {
                    "clip_id": "den1",
                    "frame_index": 421000,
                    "label_name_to_sha256": {
                        "camera_pose": "95c8ad68915e9c9956de970b82e08a258551423a3c3b8f952dc3cc6b3a26310e",
                        "original": "9f6fb0a79c094c44a0112a676909e4a9331c6c7334b44f42a7031f5ef8a006fe",
                        "floor_not_floor": "3ec993256576525887bd5007deddb681eaa3cb6e34391edd412c7adb84d1ea58",
                    },
                },
            ]

            
            chunk0="37deb6dd165db2a0b1d1ea42ecffa1f1161656526ebc7b1fb0410f37718649b2",
            chunk1="f2ffa2041832a30582b2e3bfe9b609480f433a194cae53dfc13ecd2485ef634d",
            chunk2="09b0d09f4309313c70484847313b3c12b22c2f2923aa565cc71d261372fb7221",
            chunk3="d551828911e76b7e7ac2eae6a61dc96ac67791a28cc66baefd82ec3614b8f303",


            """
        )
    )
    argp.add_argument(
        "--floor_id",
        help="The floor_id of the floor that you want to insert underneath the players etc.",
        required=True,
    )
    argp.add_argument(
        "--video_frame_annotations",
        help="Either a json5 file of video_frame_annotations_metadata or the sha256 thereof.",
        required=True,
    )
    argp.add_argument(
        "--scorebug_config",
        help="Either a json5 file configuring the scorebugs or the sha256 thereof.",
        required=True,
    )
    argp.add_argument(
        "--out_dir",
        help="The directory where you want to save the fake annotations images",
        required=True,
    )
    argp.add_argument(
        "--records_out_dir",
        help="The directory where you want to save the records of each fake annotations",
        required=True,
    )
    argp.add_argument(
        "--records_s3_dir_uri",
        help="s3 directory where you want to save the records of each fake annotations",
        required=False,
    )
    argp.add_argument(
        "--asset_repos_dir",
        help="The directory into which you clone a bunch of asset repos.",
        required=True,
    )
    argp.add_argument(
        "--print_in_iterm2",
        help="print the images in iterm2.",
        action="store_true",
    )

    
    opt = argp.parse_args()
    video_frame_annotations_json_file_or_sha256 = opt.video_frame_annotations
    scorebug_config_json_file_or_sha256 = opt.scorebug_config
    print_in_iterm2 = opt.print_in_iterm2
    floor_id = opt.floor_id
    out_dir = Path(opt.out_dir)
    records_dir_path = Path(
        opt.records_out_dir
    )
    records_dir_path.mkdir(exist_ok=True)
    asset_repos_dir = Path(
        opt.asset_repos_dir
    )
    assert asset_repos_dir.exists()
    jersey_dir = asset_repos_dir / "jersey_ids"

    records_s3_dir_uri = opt.records_s3_dir_uri
    if records_s3_dir_uri is not None:
        assert records_s3_dir_uri.startswith("s3://")
        assert records_s3_dir_uri.endswith("/")
    # context_id = "dallas_mavericks"
    # context_id = "boston_celtics"
    context_id = "nba_floor_not_floor_pasting"



    
    cutout_dirs_str = [
    
        "nba_misc_cutouts_approved/coaches",
        "nba_misc_cutouts_approved/coach_kidd",
        "nba_misc_cutouts_approved/randos",
        # "nba_misc_cutouts_approved/referees",  now that we are doing BAL we need yellow shirt referees
        # "bal_cutouts_approved/referees",
        "nba_misc_cutouts_approved/balls",
        "nba_misc_cutouts_approved/objects",
        "allstar2025_cutouts_approved/phx_lightblue",
        "denver_nuggets_cutouts_approved/icon",
        "denver_nuggets_cutouts_approved/statement",
        "houston_cutouts_approved/icon",
        "houston_cutouts_approved/statement",

        # "/shared/r/houston_cutouts_approved/statement",
        
        # "/shared/r/miami_heat_cutouts_approved/statement",
        # "/shared/r/cleveland_cavaliers_cutouts_approved/statement",
        
        # "/shared/r/dallas_mavericks_cutouts_approved/association",
        # "/shared/r/boston_celtics_cutouts_approved/statement",

        # # June 12:
        # "/shared/r/dallas_mavericks_cutouts_approved/statement",
        # "/shared/r/boston_celtics_cutouts_approved/association",

        # # June 17:
        # "/shared/r/dallas_mavericks_cutouts_approved/association",
        # "/shared/r/boston_celtics_cutouts_approved/icon",


        # "/shared/r/efs_cutouts_approved/white_with_blue_and_orange",
        # "/shared/r/athens_cutouts_approved/white_with_green",  # seem quality
        # "/shared/r/zalgiris_cutouts_approved/white_with_green",  # seem quality
        # "/shared/r/munich_cutouts_approved/maroon_uniform_shoes",  # just one guy
        # "/shared/r/munich_cutouts_approved/maroon_uniforms",  # color seems off
        # "/shared/r/munich_cutouts_approved/balls",
        # "/shared/r/munich_cutouts_approved/coaches_faithful",
        # "/shared/r/munich_cutouts_approved/referees_faithful",
    ]

    cutout_dirs = [
        asset_repos_dir / x for x in cutout_dirs_str
    ]
    for x in cutout_dirs:
        print(x)
        assert x.is_dir(), f"{x} does not exist"

    diminish_cutouts_for_debugging = True
    sport = "basketball"
    league = "nba"
    cutouts = (  # a list of PastableCutout objects
        get_cutouts(
            sport=sport,
            league=league,
            jersey_dir=jersey_dir,
            cutout_dirs=cutout_dirs,
            diminish_for_debugging=diminish_cutouts_for_debugging
        )
    )

    cutouts_by_kind = group_cutouts_by_kind(
        sport=sport,
        cutouts=cutouts,
    ) 

    valid_floor_ids = get_valid_floor_ids()

    assert (
        floor_id in valid_floor_ids
    ), f"{floor_id=} is not valid. Valid values are {valid_floor_ids=}"

  
    video_frame_annotations_metadata = osofpsaj_open_sha256_or_file_path_str_as_json(
        sha256_or_local_file_path_str=video_frame_annotations_json_file_or_sha256
    )
  
    scorebug_config = osofpsaj_open_sha256_or_file_path_str_as_json(
        sha256_or_local_file_path_str=scorebug_config_json_file_or_sha256
    )
    print("We will protect these scorebugs:")
    color_print_json(scorebug_config)
    
    # BEGIN checking that the video_frame_annotations_metadata is valid, and also gather mentioned sha256s
    mentioned_sha256s = set()
    assert isinstance(video_frame_annotations_metadata, list)
    for annotation in video_frame_annotations_metadata:
        assert isinstance(annotation, dict)
        assert "clip_id" in annotation
        assert isinstance(annotation["clip_id"], str)
        assert "frame_index" in annotation
        assert isinstance(annotation["frame_index"], int)
        assert "label_name_to_sha256" in annotation
        assert isinstance(annotation["label_name_to_sha256"], dict)
        for label_name in ["original", "camera_pose", "floor_not_floor"]:
            maybe_sha256 = annotation["label_name_to_sha256"][label_name]
            assert maybe_sha256 is None or is_sha256(maybe_sha256)
            if maybe_sha256 is not None:
                mentioned_sha256s.add(maybe_sha256)
    mentioned_sha256s = list(mentioned_sha256s)
    # ENDOF checking that the video_frame_annotations_metadata is valid:
    
    scorebug_sha256s = []
    for scorebug in scorebug_config["scorebugs"]:
        scorebug_sha256s.append(scorebug["scorebug_sha256"])
        
    mentioned_sha256s.extend(scorebug_sha256s)
    
    # unique-ify:
    mentioned_sha256s = list(set(mentioned_sha256s))
    
    # download the files:
    download_the_files_with_these_sha256s(
        sha256s_to_download=mentioned_sha256s,
        max_workers=10,
        verbose=True,
    )

    scorebug_rgbas = open_sha256s_as_rgba_hwc_np_u8s(
        sha256s=scorebug_sha256s
    )
    for x in scorebug_rgbas:
        prii(x)
    # video_frame_annotations_metadata_sha256 = "99bc2c688a6bd35f08b873495d062604e0b954244e6bb20f5c5a76826ac53524"

    cutout_kind_to_transform = dict(
        player=get_cutout_augmentation("player"),
        referee=get_cutout_augmentation("referee"),
        coach=get_cutout_augmentation("coach"),
        ball=get_cutout_augmentation("ball"),
        led_screen_occluding_object=get_cutout_augmentation("led_screen_occluding_object"),
    )

   

    print("Gathering appropriate local_file_pathed_annotations")

    # For the task of sticking a floor underneath the a floor_not_floor annotation, we need the following labels:
    desired_labels = set(["camera_pose", "floor_not_floor", "original"])

    local_file_pathed_annotations = get_local_file_pathed_annotations(
        video_frame_annotations_metadata=video_frame_annotations_metadata,
        desired_labels=desired_labels,
        desired_leagues=["nba"],
        max_num_annotations=None,
        print_in_iterm2=False,
        print_inadequate_annotations = False,
    )

    out_dir.mkdir(exist_ok=True, parents=True)
    
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




