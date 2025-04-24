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
        nargs="+",
        help="The directory(s) where you want to save the fake segmentation annotation images",
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
    print(opt.in_dir)
    in_dirs = [Path(p).resolve() for p in opt.in_dir]
    
    print_green("Taking (hopefully scorebug-free) segmentation annotations from these input directories:")
    for in_dir in in_dirs:
        print_green(f"    {in_dir}")
    
    

    datapoint_path_tuples = get_datapoint_path_tuples_from_list_of_dataset_folders(
        list_of_dataset_folders=in_dirs
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
    
    # scorebug_sha256s = []
    scorebug_sha256s.append(
        "1c2c63f087f422d7059f044633bc3a42c1001547f6cc473517da81389b680d27"
    )

    scorebug_rgbas = open_sha256s_as_rgba_hwc_np_u8s(
        sha256s=scorebug_sha256s
    )
    for i, scorebug_rgba in enumerate(scorebug_rgbas):
        print(f"scorebug {i} of {len(scorebug_rgbas)}:")
        # prii(scorebug_rgba)
    
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"Writing to directory {out_dir=}")
    
    
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
            # prii(
            #     x=original_rgb_np_u8,
            #     caption="this is the original frame:",
            # )
            # prii(
            #     x=mask_hw_np_u8,
            #     caption="this is the mask:",
            # )

            rgba_hwc_np_u8 = make_rgba_hwc_np_u8_from_rgb_and_alpha(
                rgb=original_rgb_np_u8,
                alpha=mask_hw_np_u8,
            )
            # prii(
            #     x=rgba_hwc_np_u8,
            #     caption="this is the original segmentation annotation:",
            # )

         

            annotation_id = f"{clip_id}_{frame_index:06d}"
            rid = np.random.randint(0, 1_000_000_000_000_000)
            fake_annotation_id = f"{annotation_id}_fake{rid:015d}"

            fake_original_path = out_dir / f"{fake_annotation_id}_original.jpg"
            fake_mask_path = out_dir / f"{fake_annotation_id}_nonfloor.png"

            scorebug_rgba = random.choice(scorebug_rgbas)

            with_scorebug_rgb = feathered_paste_for_images_of_the_same_size(
                bottom_layer_color_np_uint8=original_rgb_np_u8,
                top_layer_rgba_np_uint8=scorebug_rgba,
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




