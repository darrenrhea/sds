from get_camera_posed_annotations_from_a_directory_filled_with_fakes import (
     get_camera_posed_annotations_from_a_directory_filled_with_fakes
)
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


def poosa_paste_objects_onto_segmentation_annotations():
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
    python poosa_paste_objects_onto_segmentation_annotations /shared/fake_nba/pasted
    
    

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

    5. which segmentation  annotations to start with.
    The foundation of fake data is human annotations, where a human
    has erased the capable-of-shining-light-LED-part of the LED board.
    This is controlled by the list repo_ids_to_use below.
    In particular, you have to have a lot of these "darren-repos" cloned into ~/r.
    We are trying to get away from this because it is a pain for everyone else,
    and darren as well, to clone all these repos.
    """

    directory = Path("/shared/fake_nba/underneaths2")
    video_frame_annotations_metadata_sha256 = "99bc2c688a6bd35f08b873495d062604e0b954244e6bb20f5c5a76826ac53524"

    camera_posed_annotations = get_camera_posed_annotations_from_a_directory_filled_with_fakes(
        video_frame_annotations_metadata_sha256=video_frame_annotations_metadata_sha256,
        directory=directory,
    )

    print(f"Got {len(camera_posed_annotations)} camera posed annotations.")

    # BEGIN set debug flags:
    only_use_one_human_annotation_for_debugging = False
    you_want_to_see_the_cutouts_for_debugging = False
    diminish_cutouts_for_debugging = False
    # ENDOF set debug flags.

    # context_id = "dallas_mavericks"
    # context_id = "boston_celtics"
    # context_id = "summer_league_2024"
    # TODO: this should be pasting_distribution_context_id or simply pasting_spacial_distribution_id
    context_id = "nba_floor_not_floor_pasting"

    start_time = time.time()

    # The choice of league is important for setting the units of measure:
    # feet for NBA and NCAA, meters for euroleague and britishbasketball:
    sport = "basketball"
    league = "nba"

    # BEGIN command line argument parsing:
    argp = argparse.ArgumentParser()
    argp.add_argument("out_dir", type=str)

    # if you want to 
    argp.add_argument("--prii", action="store_true")
    opt = argp.parse_args()
    

    prii_them_out_in_the_terminal = opt.prii
    out_dir = Path(opt.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    assert out_dir.is_dir(), f"ERROR: {out_dir=} is not a directory."
    # BEGIN command line argument parsing:
    
   

    
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
        if x["frame_index"] == 628500
    ]

    if only_use_one_human_annotation_for_debugging:
        approved_annotations = approved_annotations[:1]
    
    # pp.pprint(approved_annotations)
    # sys.exit(1)
    
   


    for x in camera_posed_annotations:
        assert "camera_pose" in x, f"ERROR: {x=} does not have a camera_pose key."


   



    # BEGIN cutouts configuration:
    # which cutouts are used is pretty much determined by the cutout directories used:
    cutout_dirs_str = [
        "~/r/nba_misc_cutouts_approved/coaches",
        # "~/r/nba_misc_cutouts_approved/coach_kidd",
        # "~/r/nba_misc_cutouts_approved/randos",
        "~/r/nba_misc_cutouts_approved/referees",
        "~/r/nba_misc_cutouts_approved/balls",
        "~/r/nba_misc_cutouts_approved/objects",
        # "~/r/miami_heat_cutouts_approved/statement",
        "~/r/houston_cutouts_approved/icon",
        "~/r/lac_cutouts_approved/city",
        "~/r/lac_cutouts_approved/city",
        "~/r/lac_cutouts_approved/city",
        "~/r/lac_cutouts_approved/city",
        
        # "~/r/cleveland_cavaliers_cutouts_approved/statement",
        
        # "~/r/dallas_mavericks_cutouts_approved/association",
        # "~/r/boston_celtics_cutouts_approved/statement",

        # # June 12:
        # "~/r/dallas_mavericks_cutouts_approved/statement",
        # "~/r/boston_celtics_cutouts_approved/association",

        # # June 17:
        # "~/r/dallas_mavericks_cutouts_approved/association",
        # "~/r/boston_celtics_cutouts_approved/icon",


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

    cutout_dirs = [Path(x).expanduser() for x in cutout_dirs_str]
    asset_repos_dir = Path("~/r").expanduser()
    jersey_dir = asset_repos_dir / "jersey_ids"
    cutouts = get_cutouts(
        cutout_dirs=cutout_dirs,
        diminish_for_debugging=diminish_cutouts_for_debugging,
        league=league,
        sport=sport,
        jersey_dir=jersey_dir,
    )
    cutouts_by_kind = group_cutouts_by_kind(
        sport=sport,
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
    

    # setup is over, for loop:

    total_num = len(camera_posed_annotations)
    print(f"going to make {total_num} fake annotations")

    fake_counter = 0

    for camera_posed_annotation in camera_posed_annotations:  
        original_file_path = camera_posed_annotation["original_file_path"]
        mask_file_path = camera_posed_annotation["mask_file_path"]
        camera_pose = camera_posed_annotation["camera_pose"]
        # this is for explaining where a pasted-upon segmentation came from:
        annotation_id = camera_posed_annotation["annotation_id"]

        # choose where to save the fake annotation:
        rid = np.random.randint(0, 1_000_000_000_000_000)
        fake_annotation_id = f"{annotation_id}_fake{rid:015d}"
        fake_original_out_path = out_dir / f"{fake_annotation_id}_original.png"
        fake_rgba_out_path = out_dir / f"{fake_annotation_id}_nonfloor.png"
        # fake_relevance_mask_out_path = out_dir / f"{fake_annotation_id}_relevance.png"

        # open the original and mask together as rgba:
        rgba_hwc_np_u8 = make_rgba_from_original_and_mask_paths(
            original_path=original_file_path,
            mask_path=mask_file_path,
            flip_mask=False,
            quantize=False
        )

        if prii_them_out_in_the_terminal:
            print("This is the actual annotation we are going to start with:")
            prii(rgba_hwc_np_u8)

        # choose how many of each kind of cutout to paste somehow:
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

        print(f"made {fake_counter} out of {total_num} fake annotations so far in {duration} seconds.")
        print(f"That is {fake_counter / duration * 60} fake annotations per minute.")
        print(f"So it should be done in {(total_num - fake_counter) / fake_counter * duration / 60} minutes.")

        if prii_them_out_in_the_terminal:
            print("The produced fake segmentation-annotation-datapoint is a pair made these two images:")
            prii(fake_original_out_path, caption=f"{fake_original_out_path}")
            prii(fake_rgba_out_path, caption=f"{fake_rgba_out_path}")



if __name__ == "__main__":
    poosa_paste_objects_onto_segmentation_annotations()