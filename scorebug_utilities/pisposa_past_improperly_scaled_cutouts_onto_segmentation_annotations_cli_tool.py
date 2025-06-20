from print_yellow import (
     print_yellow
)
from get_cutout_dirs_str_for_nba import (
     get_cutout_dirs_str_for_nba
)
from prii_named_xy_points_on_image import (
     prii_named_xy_points_on_image
)
from get_valid_cutout_kinds import (
     get_valid_cutout_kinds
)
from group_cutouts_by_kind import (
     group_cutouts_by_kind
)
from get_cutouts import (
     get_cutouts
)
from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from get_clip_id_and_frame_index_from_mask_file_name import (
     get_clip_id_and_frame_index_from_mask_file_name
)
from get_datapoint_path_tuples_from_list_of_dataset_folders import (
     get_datapoint_path_tuples_from_list_of_dataset_folders
)
import pprint
import random
import sys
from color_print_json import color_print_json
from feathered_paste_for_images_of_the_same_size import (
     feathered_paste_for_images_of_the_same_size
)
from open_sha256s_as_rgba_hwc_np_u8s import open_sha256s_as_rgba_hwc_np_u8s
from osofpsaj_open_sha256_or_file_path_str_as_json import osofpsaj_open_sha256_or_file_path_str_as_json
from print_green import (
     print_green
)

from download_the_files_with_these_sha256s import (
     download_the_files_with_these_sha256s
)
from make_rgba_hwc_np_u8_from_rgb_and_alpha import (
     make_rgba_hwc_np_u8_from_rgb_and_alpha
)
import argparse
import textwrap
import time
from write_rgb_hwc_np_u8_to_jpg import (
     write_rgb_hwc_np_u8_to_jpg
)
from pathlib import Path
from prii import (
     prii
)
import numpy as np
from write_rgba_hwc_np_u8_to_png import (
     write_rgba_hwc_np_u8_to_png
)


def pisposa_past_improperly_scaled_cutouts_onto_segmentation_annotations_cli_tool():
    """
    Just add randomly scaled cutouts to segmentation annotations.
    Dirty, but not camera-poses available.
    """
    start_time = time.time()
    argp = argparse.ArgumentParser(
        description="pisposa_past_improperly_scaled_cutouts_onto_segmentation_annotations",
        usage=textwrap.dedent(
            """\
            


            pisposa_past_improperly_scaled_cutouts_onto_segmentation_annotations \\
            --in_dir ~/cleandata \\
            --out_dir ~/withreferees \\
            --print_in_iterm2
            """
        )
    )
    argp.add_argument(
        "--in_dir",
        nargs="+",
        help="The directory(s) where you want to save the fake segmentation annotation images",
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
    print_in_iterm2 = opt.print_in_iterm2
    out_dir = Path(opt.out_dir).resolve()
    in_dirs = [Path(p).resolve() for p in opt.in_dir]
    diminish_cutouts_for_debugging = True  # TODO should we make a flag?

    
    print_green("Taking (hopefully scorebug-free) segmentation annotations from these input directories:")
    for in_dir in in_dirs:
        print_green(f"    {in_dir}")

    # BEGIN get cutouts:
    asset_repos_dir = Path(
        "~/r"
    ).expanduser()
    assert asset_repos_dir.exists()
    jersey_dir = asset_repos_dir / "jersey_ids"

   
    context_id = "nba_floor_not_floor_pasting"

    cutout_dirs_str = get_cutout_dirs_str_for_nba()

    cutout_dirs = [
        asset_repos_dir / x for x in cutout_dirs_str
    ]
    for x in cutout_dirs:
        print(x)
        assert x.is_dir(), f"{x} does not exist"

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
   
    
    stop_time = time.time()
    print(f"Loading {len(cutouts)} cutouts into RAM in {stop_time - start_time} seconds.")

    cutouts_by_kind = group_cutouts_by_kind(
        cutouts=cutouts,
        sport="basketball"
    ) 

    print("Here are the cutouts we are going to paste onto the actual annotations:")
    print("You better check them for coverage of all uniform possibilities and adequate variety.")
    valid_cutout_kinds = get_valid_cutout_kinds()
    
    for kind in valid_cutout_kinds:
        print_green(f"All cutouts of kind {kind}:")
        for cutout in cutouts_by_kind[kind]:
            color_print_json(cutout.metadata)
            prii_named_xy_points_on_image(
                image=cutout.rgba_np_u8,
                name_to_xy=cutout.metadata["name_to_xy"]
            )

    # ENDOF get cutouts

   
    
    

    datapoint_path_tuples = get_datapoint_path_tuples_from_list_of_dataset_folders(
        dataset_folders=in_dirs
    )
    for original_file_path, mask_file_path, _ in datapoint_path_tuples:
        clip_id, frame_index = get_clip_id_and_frame_index_from_mask_file_name(
            file_name=mask_file_path.name
        )
        print(f"{clip_id=}")
        print(f"{frame_index=}")
        print(f"{original_file_path=}")
        print(f"{mask_file_path=}")
        assert original_file_path.is_file(), f"{original_file_path} is not a file"
        assert mask_file_path.is_file(), f"{mask_file_path} is not a file"
        print("\n"*5)
    

    out_dir.mkdir(exist_ok=True, parents=True)
    print_yellow(
        f"We will be writing to th is directory:    {out_dir=}"
    )
    
    
    num_laps = 1
    start_time = time.time()
    num_completed = 0
    out_of = len(datapoint_path_tuples) * num_laps
    for _ in range(num_laps):
        for original_file_path, mask_file_path, _ in datapoint_path_tuples:
            clip_id, frame_index = get_clip_id_and_frame_index_from_mask_file_name(
                file_name=mask_file_path.name
            )
           
            original_rgb_np_u8 = open_as_rgb_hwc_np_u8(
                image_path=original_file_path
            )
            mask_hw_np_u8 = open_alpha_channel_image_as_a_single_channel_grayscale_image(
                abs_file_path=mask_file_path
            )

            prii(
                x=original_rgb_np_u8,
                caption="this is the original frame:",
            )
            prii(
                x=mask_hw_np_u8,
                caption="this is the mask:",
            )

            rgba_hwc_np_u8 = make_rgba_hwc_np_u8_from_rgb_and_alpha(
                rgb=original_rgb_np_u8,
                alpha=mask_hw_np_u8,
            )
            prii(
                x=rgba_hwc_np_u8,
                caption="this is the original segmentation annotation:",
            )

         

            annotation_id = f"{clip_id}_{frame_index:06d}"
            rid = np.random.randint(0, 1_000_000_000_000_000)
            fake_annotation_id = f"{annotation_id}_fake{rid:015d}"

            fake_original_path = out_dir / f"{fake_annotation_id}_original.jpg"
            fake_mask_path = out_dir / f"{fake_annotation_id}_nonfloor.png"

            # somehow paste the cutouts onto the original frame

            with_scorebug_rgb = feathered_paste_for_images_of_the_same_size(
                bottom_layer_color_np_uint8=original_rgb_np_u8,
                top_layer_rgba_np_uint8=rgba_hwc_np_u8
            )

            with_scorebug_rgba_np_u8 = np.zeros(
                (original_rgb_np_u8.shape[0], original_rgb_np_u8.shape[1], 4),
                dtype=np.uint8
            )
            with_scorebug_rgba_np_u8[:, :, 0:3] = with_scorebug_rgb
            with_scorebug_rgba_np_u8[:, :, 3] = np.maximum(
                scorebug_rgba[:, :, 3],
                mask_hw_np_u8,
            )

            write_rgb_hwc_np_u8_to_jpg(
                rgb_hwc_np_u8=with_scorebug_rgba_np_u8[:, :, 0:3],
                out_abs_file_path=fake_original_path,
                verbose=True
            )
            write_rgba_hwc_np_u8_to_png(
                rgba_hwc_np_u8=with_scorebug_rgba_np_u8,
                out_abs_file_path=fake_mask_path,
                verbose=True
            )

           

            if print_in_iterm2:
               
                prii(
                    x=original_rgb_np_u8,
                    caption="this is the original frame:",
                )
                
                prii(
                    x=fake_original_path,
                    caption="this is the synthetic / fake original:",
                )

                prii(
                    x=fake_mask_path,
                    caption="this is the synthetic / fake mask:",
                )

               
            num_completed += 1
            duration = time.time() - start_time
            print(f"So far, it has been {duration/60} minutes")
            seconds_per_item = duration / num_completed
            estimated_remaining = (out_of - num_completed) * seconds_per_item
            print(f"completed {num_completed} / {out_of}")
            print(f"Estimated time remaining: {estimated_remaining/60} minutes.")




if __name__ == "__main__":
    # for those who want to run it without the installing the executable called
    # pisposa_past_improperly_scaled_cutouts_onto_segmentation_annotations
    pisposa_past_improperly_scaled_cutouts_onto_segmentation_annotations_cli_tool()