from color_correct_rgb_via_polynomial_coefficients_linear_f32_in_and_out import (
     color_correct_rgb_via_polynomial_coefficients_linear_f32_in_and_out
)
from convert_u8_to_linear_f32 import (
     convert_u8_to_linear_f32
)
from augment_texture import (
     augment_texture
)
from get_a_random_ad_they_sent_us_file_path import (
     get_a_random_ad_they_sent_us_file_path
)
from get_ad_name_to_paths_that_do_need_color_correction import (
     get_ad_name_to_paths_that_do_need_color_correction
)
from get_ad_name_to_paths_that_dont_need_color_correction import (
     get_ad_name_to_paths_that_dont_need_color_correction
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from convert_from_rgb_hwc_np_u8_to_rgba_hwc_np_nonlinear_f32 import (
     convert_from_rgb_hwc_np_u8_to_rgba_hwc_np_nonlinear_f32
)
from get_a_random_adrip_file_path import (
     get_a_random_adrip_file_path
)
from get_a_random_adrip_as_rgba_hwc_np_nonlinear_f32 import (
     maybe_augment_adrip
)
from insert_fake_ads_plain_version import (
     insert_fake_ads_plain_version
)
from insert_fake_ads_they_sent_to_us import (
     insert_fake_ads_they_sent_to_us
)
import pprint as pp
from get_rgba_hwc_np_f32_from_texture_id import (
     get_rgba_hwc_np_f32_from_texture_id
)
from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image
)
from add_noise_via_jpg_lossyness import (
     add_noise_via_jpg_lossyness
)
from create_ij_displacement_and_weight_pairs import (
     create_ij_displacement_and_weight_pairs
)
from blur_both_original_and_mask_u8 import (
     blur_both_original_and_mask_u8
)
from choose_by_percent import (
     choose_by_percent
)
from load_color_correction_from_json import (
     load_color_correction_from_json
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from get_world_coordinate_descriptors_of_ad_placements import (
     get_world_coordinate_descriptors_of_ad_placements
)
from make_relevance_mask_for_led_boards import (
     make_relevance_mask_for_led_boards
)
from write_rgb_hwc_np_u8_to_png import (
     write_rgb_hwc_np_u8_to_png
)
from write_grayscale_hw_np_u8_to_png import (
     write_grayscale_hw_np_u8_to_png
)
from write_rgba_hwc_np_u8_to_png import (
     write_rgba_hwc_np_u8_to_png
)
from get_augmentation_for_texture import (
     get_augmentation_for_texture
)
from add_camera_pose_to_annotations import (
     add_camera_pose_to_annotations
)
from get_approved_annotations_from_these_repos import (
     get_approved_annotations_from_these_repos
)
import argparse
import numpy as np
import time
from pathlib import Path
from prii import (
     prii
)
from get_valid_cutout_kinds import (
     get_valid_cutout_kinds
)
from better_json import (
     color_print_json
)
from group_cutouts_by_kind import (
     group_cutouts_by_kind
)
from prii_named_xy_points_on_image import (
     prii_named_xy_points_on_image
)
from get_cutouts import (
     get_cutouts
)
from make_rgba_from_original_and_mask_paths import (
     make_rgba_from_original_and_mask_paths
)
from paste_multiple_cutouts_onto_one_camera_posed_segmentation_annotation import (
     paste_multiple_cutouts_onto_one_camera_posed_segmentation_annotation
)
from get_cutout_augmentation import (
     get_cutout_augmentation
)


def gabofla_generate_a_bunch_of_fake_led_annotations():
    """
    This is the final caller / entrypoint for making fake data.

    You must configure some things:

    0. where in 3D space, to place objects like balls, basketball players, coaches, referees.
    
    This happens in the procedure: get_random_world_coordinates_for_a_cutout_of_this_kind
    
    We often want to place things close enough to the LED board that they
    are fairly likely to occlude the LED board, to teach the neural network a lesson
    about how to tell the diffence between the LED board and other things.

    A change of units, like from meters to feet,
    as happens when going from Euroleague to the NBA of NCAA,
    will require a change in the world coordinates.

    So choose a league, i.e. one the the strings:
    "euroleague" or "nba" or "britishbasketball",
    to determine the units of measure.

    1. where to stick the generated fake annotations.
    This is controlled by the command line arguments to
    gabofa_generate_a_bunch_of_fake_annotations.py
    i.e. give a directory at the command line like:

    mkdir -p /shared/fake_euroleague/bkt
    python gabofa_generate_a_bunch_of_fake_annotations.py bkt /shared/fake_euroleague/bkt
    
    

    You have to choose to insert either ad-rips or ads they sent to us.
    This is controlled by the ad_insertion_method variable below being set to "ad_rips" or "ads_they_sent_to_us".

    # Check which ad_rips will be used via:

    python get_ad_name_to_paths_that_dont_need_color_correction.py

    2. Which ads to insert.  Set ad_id to "bkt", "skweek", or "denizbank", for euroleague, or for the NBA do 
    ls -1 ~/r/nba_ads to see the list of ad jpgs that are allowed.
    For example, set ad_id to Playoffs_Title because
    the NBA ad ~/r/nba_ads/Playoffs_Title.jpg exists.
    In particular you better have the nba_ads repo cloned and recent into ~/r.
    We are moving away from this soon, so that people will not have to clone
    a bunch of darren repos.

    3. Choose a color correction polynomial.  This is the sha256 hash of an
    aws-s3-stored json file
    that states the total degree and coefficients of the multivariate polynomial
    that maps the 3D rgb color space cube [0, 1]^3 of ads they sent to us
    into the 3D color space cube rgb [0, 1]^3 of how it actually is written down by
    camera 1, due to various opto-electronic transfers such as:
    1. rgb values in the RAM of the LED board driving computer to actual brightness of the LEDs
    2. plus ambient lighting of the LED board, which makes (0,0,0) black impossible
    3. composed with the recording opto-electronic transfer function, i.e. 
    light received by the camera sensor transfers to rgb values in the  RAM of the camera's computer,
    on to SDI.
    We don't think the ansatz of being a multivariable polynomial is very restrictive, because
    presumably the color correction map is continuous, so by Stone-Weierstrass, we can approximate it
    arbitrarily well with a polynomial on [0,1]^3.

    Set by color_correction_sha256 = "bd545cba8ac10558b8a5a4eeba40bc3be9f1e809975fd7e6ad38d6a3ac598140"

    
    4. which cutouts to use.  This is controlled by cutout_dirs_str below.
    Do we want to include what player cutouts?
    Currently every player, referee, coach, ball, and led_screen_occluding_object must have a json5 file along side it,
    with the same base name, that has xy points of the toe and 6 feet above that (the top of there head for instance),
    or in the case of the ball, the bottom, center, and top of the ball.
    This is so we can place them in 3D space with an appropriate scale.

    5. which human annotations to start with.
    The foundation of fake data is human annotations, where a human
    has erased the capable-of-shining-light-LED-part of the LED board.
    This is controlled by the list repo_ids_to_use below.
    In particular, you have to have a lot of these "darren-repos" cloned into ~/r.
    We are trying to get away from this because it is a pain for everyone else,
    and darren as well, to clone all these repos.
    """

    # BEGIN set debug flags:
    only_use_one_human_annotation_for_debugging = False
    you_want_to_see_the_cutouts_for_debugging = False
    diminish_cutouts_for_debugging = False
    # ENDOF set debug flags.

    # context_id = "dallas_mavericks"
    # context_id = "boston_celtics"
    context_id = "summer_league_2024"

    ad_insertion_method = "ad_rips"
    # ad_insertion_method = "ads_they_sent_to_us"

    start_time = time.time()

    flip_flop_dir = Path("~/ff").expanduser()
    
    # The choice of league is important for setting the units of measure:
    # feet for NBA and NCAA, meters for euroleague and britishbasketball:
    league = "nba"

    

    # BEGIN command line argument parsing:
    argp = argparse.ArgumentParser()
    argp.add_argument("ad_id", type=str)
    argp.add_argument("out_dir", type=str)

    # if you want to 
    argp.add_argument("--prii", action="store_true")
    opt = argp.parse_args()
    ad_id = opt.ad_id
    

    prii_them_out_in_the_terminal = opt.prii
    out_dir = Path(opt.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    assert out_dir.is_dir(), f"ERROR: {out_dir=} is not a directory."
    # BEGIN command line argument parsing:
    
    if False:
        # Uses linear light, and gets the emerald buck and ESPN red right, dark background:
        color_correction_sha256 = "4edceff5771335b7a64b1507fa1d31f38f5148f71322092c4db5ecd8ec6e985b"
        color_correction_json_path = get_file_path_of_sha256(sha256=color_correction_sha256)
        
        degree, coefficients = load_color_correction_from_json(
            json_path=color_correction_json_path
        )
    else:
        rm, r_boost = 1.36, 0.01
        gm, g_boost = 1.99, 0.01
        bm, b_boost = 2.79, 0.01

        degree = 1
        coefficients = np.array(
            [
                [ r_boost, g_boost, b_boost],
                [ rm, 0.0, 0.0],
                [0.0,  gm,  0.0],
                [0.0,  0.0,  bm],
            ]
        )
    
    # BEGIN configure which human annotations to start from, and which ones to use for what:

    
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
        "slgame1_led",
        # "dal-bos-2024-06-12-mxf_led",
        # "dal-bos-2024-01-22-mxf_led",
    ]
    for repo_ids in repo_ids_to_use:
        print(repo_ids)
    
    approved_annotations = get_approved_annotations_from_these_repos(
        repo_ids_to_use=repo_ids_to_use
    )
    approved_annotations = [
        x for x in approved_annotations
        # if x["frame_index"] == 628500
    ]

    if only_use_one_human_annotation_for_debugging:
        approved_annotations = approved_annotations[:1]
    
    # pp.pprint(approved_annotations)
    # sys.exit(1)
    
        
    instructions_anna = {
        "paste_cutouts": True,
        "insert_ads": False,
    }

    instructions_random = {
        "paste_cutouts": 1.0,
        "insert_ads": 1.0,
    }

    named_sets_of_camera_posed_annotations_with_instructions = [
        # ["everything", approved_annotations, instructions_random],
        ["anna", approved_annotations, instructions_anna],  # just ad cutouts
    ]

    # LBYL: Look Before You Leap
    for name_of_set, annotations, instructions in named_sets_of_camera_posed_annotations_with_instructions:
        print(f"adding camera parameters to {name_of_set} annotations")
        add_camera_pose_to_annotations(annotations=annotations)
        for x in annotations:
            assert "camera_pose" in x, f"ERROR: {x=} does not have a camera_pose key."


    for name_of_set, camera_posed_annotations, instructions in named_sets_of_camera_posed_annotations_with_instructions:
        for x in camera_posed_annotations:
            assert "camera_pose" in x, f"ERROR: {x=} does not have a camera_pose key."
    
    # ENDOF configure which human annotations to start from, and which ones to use for what:


    # BEGIN INSERTED ADS CONFIGURATION:
    albu_transform_for_textures = get_augmentation_for_texture(
        ad_insertion_method=ad_insertion_method
    )
    # ENDOF INSERTED ADS CONFIGURATION.


    # BEGIN cutouts configuration:
    # which cutouts are used is pretty much determined by the cutout directories used:
    cutout_dirs_str = [
        "~/r/nba_misc_cutouts_approved/coaches",
        "~/r/nba_misc_cutouts_approved/coach_kidd",
        "~/r/nba_misc_cutouts_approved/randos",
        "~/r/nba_misc_cutouts_approved/referees",
        "~/r/nba_misc_cutouts_approved/balls",
        "~/r/nba_misc_cutouts_approved/objects",
        "~/r/miami_heat_cutouts_approved/statement",
        "~/r/cleveland_cavaliers_cutouts_approved/statement",
        
        "~/r/dallas_mavericks_cutouts_approved/association",
        "~/r/boston_celtics_cutouts_approved/statement",

        # June 12:
        "~/r/dallas_mavericks_cutouts_approved/statement",
        "~/r/boston_celtics_cutouts_approved/association",

        # June 17:
        "~/r/dallas_mavericks_cutouts_approved/association",
        "~/r/boston_celtics_cutouts_approved/icon",


        # "~/r/efs_cutouts_approved/white_with_blue_and_orange",
        # "~/r/athens_cutouts_approved/white_with_green",  # seem quality
        # "~/r/zalgiris_cutouts_approved/white_with_green",  # seem quality
        # "~/r/munich_cutouts_approved/maroon_uniform_shoes",  # just one guy
        # "~/r/munich_cutouts_approved/maroon_uniforms",  # color seems off
        # "~/r/munich_cutouts_approved/balls",
        # "~/r/munich_cutouts_approved/coaches_faithful",
        # "~/r/munich_cutouts_approved/referees_faithful",
    ]
    # someone has to choose the cutout augmentations for each kind of cutout:

    cutout_kind_to_transform = dict(
        player=get_cutout_augmentation("player"),
        referee=get_cutout_augmentation("referee"),
        coach=get_cutout_augmentation("coach"),
        ball=get_cutout_augmentation("ball"),
        led_screen_occluding_object=get_cutout_augmentation("led_screen_occluding_object"),
    )

    # ENDOF cutouts configuration.

    
    ad_name_to_paths_that_dont_need_color_correction = \
    get_ad_name_to_paths_that_dont_need_color_correction()

    ad_name_to_paths_that_do_need_color_correction = \
    get_ad_name_to_paths_that_do_need_color_correction()

    pp.pprint(ad_name_to_paths_that_do_need_color_correction)
    # the ad_id specified is ignored when doing rips, probably because they are named wrongly.
    # ad_name_to_paths_that_dont_need_color_correction = {
    #     k: v
    #     for k, v in ad_name_to_paths_that_dont_need_color_correction.items()
    #     if k == ad_id
    # }

    ad_name_to_paths_that_do_need_color_correction = {
        k: v
        for k, v in ad_name_to_paths_that_do_need_color_correction.items()
        if k == ad_id
    }
    if ad_insertion_method == "ad_rips":
        assert len(ad_name_to_paths_that_dont_need_color_correction) > 0

    if ad_insertion_method == "ads_they_sent_to_us": 
        assert len(ad_name_to_paths_that_do_need_color_correction) > 0

    cutout_dirs = [Path(x).expanduser() for x in cutout_dirs_str]

    cutouts = get_cutouts(
        cutout_dirs=cutout_dirs,
        diminish_for_debugging=diminish_cutouts_for_debugging
    )
    cutouts_by_kind = group_cutouts_by_kind(
        cutouts=cutouts
    ) 

    print("Here are the cutouts we are going to paste onto the actual annotations:")
    print("You better check them for coverage of all uniform possibilities and adequate variety.")
    valid_cutout_kinds = get_valid_cutout_kinds()
    
    if you_want_to_see_the_cutouts_for_debugging:
        for kind in valid_cutout_kinds:
            print(f"All cutouts of kind {kind}:")
            for cutout in cutouts_by_kind[kind]:
                color_print_json(cutout.metadata)
                prii_named_xy_points_on_image(
                    image=cutout.rgba_np_u8,
                    name_to_xy=cutout.metadata["name_to_xy"]
                )
    
    mirror_floor_texture_rgba_np_f32 = get_rgba_hwc_np_f32_from_texture_id(
        texture_id="23-24_BOS_CORE",
        blur_radius=(18.0, 4.0),
        use_linear_light=True
    )
    
    mirror_floor_texture_rgba_np_f32 = np.flip(mirror_floor_texture_rgba_np_f32, axis=0)

    
    # prii_linear_f32(mirror_floor_texture_rgba_np_f32)

    # setup is over, for loop:

    total_num = 0
    for name_of_set, camera_posed_annotations, instructions in named_sets_of_camera_posed_annotations_with_instructions:
        total_num += len(camera_posed_annotations)
    print(f"going to make {total_num} fake annotations")

    fake_counter = 0
    for name_of_set, camera_posed_annotations, instructions in named_sets_of_camera_posed_annotations_with_instructions:
        for camera_posed_annotation in camera_posed_annotations:  
            annotation_id = camera_posed_annotation["annotation_id"]   
            clip_id = camera_posed_annotation["clip_id"]
            frame_index = camera_posed_annotation["frame_index"]
            camera_pose = camera_posed_annotation["camera_pose"]
            mask_file_path = camera_posed_annotation["mask_file_path"]
            original_file_path = camera_posed_annotation["original_file_path"]
            
            # TODO: might use original_file_path to get the original image:
            original_rgb_hwc_np_u8 = open_as_rgb_hwc_np_u8(original_file_path)
            # original_rgb_hwc_np_u8 = get_original_frame_from_clip_id_and_frame_index(
            #     clip_id=clip_id,
            #     frame_index=frame_index,
            # )

            mask_hw_np_u8 = open_alpha_channel_image_as_a_single_channel_grayscale_image(
                abs_file_path=mask_file_path
            )

            # overcover the led boards for relevance masks:
            ad_placement_descriptors_for_relevance_masks = get_world_coordinate_descriptors_of_ad_placements(
                clip_id=clip_id,
                with_floor_as_giant_ad=False,
                overcover_by=0.4,
            )

            ad_placement_descriptors = get_world_coordinate_descriptors_of_ad_placements(
                clip_id=clip_id,
                with_floor_as_giant_ad=False,
                overcover_by=0.2
            )
            
            if prii_them_out_in_the_terminal:
                print("This is the actual annotation we are going to start with:")
                prii(camera_posed_annotation["original_file_path"])
                

            insert_ads_instructions = instructions.get("insert_ads")

            if insert_ads_instructions is True:
                do_insert_fake_ads = True
            elif insert_ads_instructions is False:
                do_insert_fake_ads = False
            else:
               assert isinstance(insert_ads_instructions, float)
               do_insert_fake_ads = (np.random.rand() < insert_ads_instructions)

            paste_cutouts_instructions = instructions.get("paste_cutouts")
            if paste_cutouts_instructions is True:
                do_paste_cutouts = True
            elif paste_cutouts_instructions is False:
                do_paste_cutouts = False
            else:
               assert isinstance(paste_cutouts_instructions, float)
               do_paste_cutouts = (np.random.rand() < paste_cutouts_instructions)

            if do_insert_fake_ads:
                if ad_insertion_method == "ad_rips":
                    print("Doing plain insertion, usually for rips")
                    
                    adrip_file_path = get_a_random_adrip_file_path(
                        ad_name_to_paths_that_dont_need_color_correction
                    )

                    unaugmented_adrip_rgb_hwc_np_u8 = \
                    open_as_rgb_hwc_np_u8(adrip_file_path)

                    augmented_adrip_rgb_hwc_np_u8 = \
                    maybe_augment_adrip(
                        albu_transform=albu_transform_for_textures,
                        unaugmented_adrip_rgb_hwc_np_u8=unaugmented_adrip_rgb_hwc_np_u8
                    )

                    final_color_ad_texture_rgba_np_nonlinear_f32 = \
                    convert_from_rgb_hwc_np_u8_to_rgba_hwc_np_nonlinear_f32(
                        rgb_hwc_np_u8=augmented_adrip_rgb_hwc_np_u8
                    )
 
                    
                    # prii_nonlinear_f32(final_color_ad_texture_rgba_np_nonlinear_f32)

                    rgb_hwc_np_u8 = insert_fake_ads_plain_version(
                        ad_placement_descriptors=ad_placement_descriptors,
                        original_rgb_hwc_np_u8=original_rgb_hwc_np_u8,
                        mask_hw_np_u8=mask_hw_np_u8,
                        final_color_ad_texture_rgba_np_nonlinear_f32=final_color_ad_texture_rgba_np_nonlinear_f32,
                        camera_pose=camera_pose,
                    )
                elif ad_insertion_method == "ads_they_sent_to_us":
                    print("Inserting ideal ads they sent to us after color correction, noise and light and blur augmentation, etc.")
                    # get the ad they sent over to us.
                    # maybe augment it
                    # go to linear_f32 for blurring and stuff
                    # insert it

                    ad_they_sent_us_file_path = get_a_random_ad_they_sent_us_file_path(
                        ad_name_to_paths_that_do_need_color_correction=ad_name_to_paths_that_do_need_color_correction
                    )

                    ad_they_sent_us_rgb_hwc_np_u8 = \
                    open_as_rgb_hwc_np_u8(ad_they_sent_us_file_path)

                    # For now, we augment while still in rgb_hwc_np_u8:
                    augmented_ad_they_sent_us_rgb_hwc_np_u8 = augment_texture(
                        rgb_np_u8=ad_they_sent_us_rgb_hwc_np_u8,
                        transform=albu_transform_for_textures,
                    )

                    augmented_ad_they_sent_us_rgb_hwc_np_linear_f32 = \
                    convert_u8_to_linear_f32(
                        augmented_ad_they_sent_us_rgb_hwc_np_u8
                    )

                    corrected_ad_they_sent_us_rgb_hwc_np_linear_f32 = \
                    color_correct_rgb_via_polynomial_coefficients_linear_f32_in_and_out(
                        degree=degree,
                        coefficients=coefficients,
                        rgb_hwc_np_linear_f32=augmented_ad_they_sent_us_rgb_hwc_np_linear_f32
                    )

                    rgb_hwc_np_u8 = insert_fake_ads_they_sent_to_us(
                        original_rgb_hwc_np_u8=original_rgb_hwc_np_u8,
                        mask_hw_np_u8=mask_hw_np_u8,
                        camera_pose=camera_pose,
                        ad_placement_descriptors=ad_placement_descriptors,
                        final_color_ad_rgb_np_linear_f32=corrected_ad_they_sent_us_rgb_hwc_np_linear_f32,
                        verbose=False,
                        flip_flop_dir=flip_flop_dir,
                    )
                else:
                    raise Exception(f"ERROR: {ad_insertion_method=} is not a valid ad_insertion_method.")
                
                # ad insertion does not change the mask:
                rgba_hwc_np_u8 = np.concatenate(
                    [
                        rgb_hwc_np_u8,
                        mask_hw_np_u8[:, :, np.newaxis]
                    ],
                    axis=2
                )
            else:
                print("Due to coin flip, we are not inserting any ads into the ad boards at this time.")
                rgba_hwc_np_u8 = make_rgba_from_original_and_mask_paths(
                    original_path=camera_posed_annotation["original_file_path"],
                    mask_path=camera_posed_annotation["mask_file_path"],
                    flip_mask=False,
                    quantize=False
                )
            
            if prii_them_out_in_the_terminal:
                print("This is the result of any ad insertion:")
                prii(rgba_hwc_np_u8[:, :, :3])
    
            # choose where to save the fake annotation:
            rid = np.random.randint(0, 1_000_000_000_000_000)
            fake_annotation_id = f"{annotation_id}_fake{rid:015d}"
            fake_original_out_path = out_dir / f"{fake_annotation_id}_original.png"
            fake_rgba_out_path = out_dir / f"{fake_annotation_id}_nonfloor.png"
            fake_relevance_mask_out_path = out_dir / f"{fake_annotation_id}_relevance.png"
            
            relevance_mask = make_relevance_mask_for_led_boards(
                camera_posed_original_video_frame=camera_posed_annotation,
                ad_placement_descriptors=ad_placement_descriptors_for_relevance_masks,
            )

            write_grayscale_hw_np_u8_to_png(
                grayscale_hw_np_u8=relevance_mask,
                out_abs_file_path=fake_relevance_mask_out_path
            )
    
            if do_paste_cutouts:
                # choose how many of each kind somehow:
                cutout_kind_to_num_cutouts_to_paste = dict(
                    player=np.random.randint(0, 12),
                    referee=np.random.randint(0, 3),
                    coach=np.random.randint(0, 3),
                    ball=np.random.randint(0, 10),
                    led_screen_occluding_object=np.random.randint(0, 2),
                )

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
                print("Due to coin flip, we are not pasting cutouts this time.")
                pasted_rgba_np_u8 = rgba_hwc_np_u8


            # blur the fake annotation:
            radius = choose_by_percent(
                value_prob_pairs=[
                    (0, 0.50),
                    (1, 0.40),
                    (2, 0.10),
                    # (3, 0.05),
                    # (4, 0.05),
                    # (5, 0.05),
                    # (6, 0.05),
                ]
            )


            ij_displacement_and_weight_pairs = create_ij_displacement_and_weight_pairs(
                radius=radius
            )
            blurred_rgba_hwc_np_u8 = blur_both_original_and_mask_u8(
                rgba_np_u8=pasted_rgba_np_u8,
                ij_displacement_and_weight_pairs=ij_displacement_and_weight_pairs,
            )
            blurred_mask = blurred_rgba_hwc_np_u8[:, :, 3:4]
            assert blurred_mask.shape == (1080, 1920, 1)
            # prii(blurred_rgba_hwc_np_u8[:, :, :3], caption="prior to JPEG noise:", out=Path("prior_to_jpeg.png").resolve())
            jpeg_quality = choose_by_percent(
                value_prob_pairs=[
                    (95, 0.25),
                    (80, 0.25),
                    (70, 0.10),
                    # (60, 0.10),
                    # (50, 0.10),
                    # (40, 0.10),
                    # (30, 0.10),
                ]
            )

            noisy_rgb_hwc_np_u8 = add_noise_via_jpg_lossyness(
                rgb_hwc_np_u8=blurred_rgba_hwc_np_u8[:, :, :3],
                jpeg_quality=jpeg_quality,
            )

            noisy_rgba_np_u8 = np.concatenate(
                [
                    noisy_rgb_hwc_np_u8,
                    blurred_mask,
                ],
                axis=2,
            )

            write_rgba_hwc_np_u8_to_png(
                rgba_hwc_np_u8=noisy_rgba_np_u8,
                out_abs_file_path=fake_rgba_out_path
            )
           
            write_rgb_hwc_np_u8_to_png(
                rgb_hwc_np_u8=noisy_rgba_np_u8[:, :, :3],
                out_abs_file_path=fake_original_out_path
            )
            print(fake_original_out_path)
            fake_counter += 1
            
            duration = time.time() - start_time

            print(f"made {fake_counter} out of {total_num } fake annotations so far in {duration} seconds.")
            print(f"That is {fake_counter / duration * 60} fake annotations per minute.")

            if prii_them_out_in_the_terminal:
                print("The produced fake segmentation-annotation-datapoint is a triplet made these three images:")
                prii(fake_original_out_path, caption=f"{fake_original_out_path}")
                prii(fake_rgba_out_path, caption=f"{fake_rgba_out_path}")
                prii(fake_relevance_mask_out_path, caption=f"{fake_relevance_mask_out_path}")



if __name__ == "__main__":
    gabofla_generate_a_bunch_of_fake_led_annotations()