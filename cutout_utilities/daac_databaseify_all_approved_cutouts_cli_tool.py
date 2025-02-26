from group_cutouts_by_kind import (
     group_cutouts_by_kind
)
from get_cutouts import (
     get_cutouts
)
import argparse
import textwrap
from get_cutout_augmentation import (
     get_cutout_augmentation
)


      
def daac_databaseify_all_approved_cutouts_cli_tool():
    """
    This is a command line tool that takes a bunch of folders on your local hard-drive, usually pulled-to-latest-versino repos of human-approved cutouts,
    and converts them into a json database of cutouts for better querying (filtering, aggregations) and better deployment to computers that don't want to pull a bunch of repos.
    """
    argp = argparse.ArgumentParser(
        description=textwrap.dedent(
            """\
            Converts a bunch of folders on your local hard-drive, usually pulled-to-latest-versino repos of human-approved cutouts,
            into a json database of cutouts for better querying (filtering, aggregations) and better deployment
            to instances that don't want to pull repos.
            """
        ),
        usage=textwrap.dedent(
            """\
            conda activate sds

            daac_databaseify_all_approved_cutouts \\
            --out_dir ~/a/crap \\
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
        "--out_file",
        help="The directory where you want to save the fake annotations.",
        required=True,
    )
    argp.add_argument(
        "--print_in_iterm2",
        help="print the images in iterm2.",
        action="store_true",
    )
    opt = argp.parse_args()
    video_frame_annotations_json_file_or_sha256 = opt.video_frame_annotations
    print_in_iterm2 = opt.print_in_iterm2
    floor_id = opt.floor_id
    out_file = Path(opt.out_file).resolve().expanduser()

    # context_id = "dallas_mavericks"
    # context_id = "boston_celtics"
    context_id = "nba_floor_not_floor_pasting"

    cutout_dirs_str = [
        "~/r/nba_misc_cutouts_approved/coaches",
        "~/r/nba_misc_cutouts_approved/coach_kidd",
        "~/r/nba_misc_cutouts_approved/randos",
        "~/r/nba_misc_cutouts_approved/referees",
        "~/r/nba_misc_cutouts_approved/balls",
        "~/r/nba_misc_cutouts_approved/objects",
        "~/r/allstar2025_cutouts_approved/phx_lightblue",
        "~/r/denver_nuggets_cutouts_approved/icon",
        "~/r/denver_nuggets_cutouts_approved/statement",
        "~/r/houston_cutouts_approved/icon",

        # "~/r/houston_cutouts_approved/statement",
        
        # "~/r/miami_heat_cutouts_approved/statement",
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

    cutout_dirs = [Path(x).expanduser() for x in cutout_dirs_str]
    diminish_cutouts_for_debugging = False
    sport = "basketball"
    league = "nba"
    cutouts = get_cutouts(
        sport=sport,
        league=league,
        cutout_dirs=cutout_dirs,
        diminish_for_debugging=diminish_cutouts_for_debugging
    )
    cutouts_by_kind = group_cutouts_by_kind(
        sport=sport,
        cutouts=cutouts
    ) 

    cutout_kind_to_transform = dict(
        player=get_cutout_augmentation("player"),
        referee=get_cutout_augmentation("referee"),
        coach=get_cutout_augmentation("coach"),
        ball=get_cutout_augmentation("ball"),
        led_screen_occluding_object=get_cutout_augmentation("led_screen_occluding_object"),
    )

    # # choose how many of each kind somehow:
    # cutout_kind_to_num_cutouts_to_paste = dict(
    #     player=np.random.randint(0, 12),
    #     referee=np.random.randint(0, 3),
    #     coach=np.random.randint(0, 3),
    #     ball=np.random.randint(0, 10),
    #     led_screen_occluding_object=np.random.randint(0, 2),
    # )
    # league = "nba"
    # pasted_rgba_np_u8 = paste_multiple_cutouts_onto_one_camera_posed_segmentation_annotation(
    #     league=league,
    #     context_id=context_id,
    #     cutouts_by_kind=cutouts_by_kind,
    #     rgba_np_u8=rgba_hwc_np_u8,  # this is not violated by this procedure.
    #     camera_pose=camera_pose,  # to get realistics locations and sizes we need to know the camera pose.
    #     cutout_kind_to_transform=cutout_kind_to_transform, # what albumentations augmentation to use per kind of cutout
    #     cutout_kind_to_num_cutouts_to_paste=cutout_kind_to_num_cutouts_to_paste
    # )




