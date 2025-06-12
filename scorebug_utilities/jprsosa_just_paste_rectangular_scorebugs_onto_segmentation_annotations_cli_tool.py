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
import random
from feathered_paste_for_images_of_the_same_size import (
     feathered_paste_for_images_of_the_same_size
)
from print_green import (
     print_green
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


def jprsosa_just_paste_rectangular_scorebugs_onto_segmentation_annotations_cli_tool():
    """
    Just add random scorebugs to the images in the directory.
    """
    xmin = 464 # good
    xmax = 1460 # good
    ymin = 950  # good
    ymax = 1058  # good

    argp = argparse.ArgumentParser(
        description="jpsosa_just_paste_scorebugs_onto_segmentation_annotations",
        usage=textwrap.dedent(
            """\
            


            jprsosa_just_paste_rectangular_scorebugs_onto_segmentation_annotations \\
            --in_dir \\
            /shared/ind-bos-2024-10-30-hack_floor/.approved \\
            /shared/ind-lal-2023-02-02-mxf_floor/.approved \\
            --scorebug_dir ~/scorebugs \\
            --out_dir /shared/withscorebugs \\
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
        "--scorebug_dir",
        help="a directory with rectangular scorebug images that are 996x108",
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
    out_dir = Path(opt.out_dir)
    scorebug_dir = Path(opt.scorebug_dir).resolve()
    in_dirs = [Path(p).resolve() for p in opt.in_dir]
    
    print_green("Taking (hopefully scorebug-free) segmentation annotations from these input directories:")
    for in_dir in in_dirs:
        print_green(f"    {in_dir}")
    
    

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
    
   
    
        
    scorebug_file_paths = list(scorebug_dir.glob("*.png"))
    scorebug_rgbs = []
    for scorebug_file_path in scorebug_file_paths:
        assert scorebug_file_path.is_file(), f"{scorebug_file_path} is not a file"
        scorebug_rgb = open_as_rgb_hwc_np_u8(
            image_path=scorebug_file_path
        )
        assert scorebug_rgb.shape[0] == 108, f"Scorebug {scorebug_file_path} has height {scorebug_rgb.shape[0]} but must be 108"
        assert scorebug_rgb.shape[1] == 996, f"Scorebug {scorebug_file_path} has width {scorebug_rgb.shape[1]} but must be 996"
        assert scorebug_rgb.shape[2] == 3, f"Scorebug {scorebug_file_path} has {scorebug_rgb.shape[2]} channels but must be 3"
        scorebug_rgbs.append(scorebug_rgb)
  
    for i, scorebug_rgb in enumerate(scorebug_rgbs):
        print(f"scorebug {i} of {len(scorebug_rgbs)}:")
        prii(scorebug_rgb)
    
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"Writing to directory {out_dir=}")
    
    
    num_laps = 10
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

            # rgba_hwc_np_u8 = make_rgba_hwc_np_u8_from_rgb_and_alpha(
            #     rgb=original_rgb_np_u8,
            #     alpha=mask_hw_np_u8,
            # )
            # prii(
            #     x=rgba_hwc_np_u8,
            #     caption="this is the original segmentation annotation:",
            # )

         

            annotation_id = f"{clip_id}_{frame_index:06d}"
            rid = np.random.randint(0, 1_000_000_000_000_000)
            fake_annotation_id = f"{annotation_id}_fake{rid:015d}"

            fake_original_path = out_dir / f"{fake_annotation_id}_original.jpg"
            fake_mask_path = out_dir / f"{fake_annotation_id}_nonfloor.png"

            scorebug_rgb = random.choice(scorebug_rgbs)
            scorebug_rgba = np.zeros(
                shape=(1080, 1920, 4),
                dtype=np.uint8
            )
            
            scorebug_rgba[ymin:ymax, xmin:xmax, 0:3] = scorebug_rgb
            scorebug_rgba[ymin:ymax, xmin:xmax, 3] = 255

            with_scorebug_rgb = feathered_paste_for_images_of_the_same_size(
                bottom_layer_color_np_uint8=original_rgb_np_u8,
                top_layer_rgba_np_uint8=scorebug_rgba,
            )

            with_scorebug_rgb_np_u8 = np.zeros(
                (original_rgb_np_u8.shape[0], original_rgb_np_u8.shape[1], 4),
                dtype=np.uint8
            )
            with_scorebug_rgb_np_u8[:, :, 0:3] = with_scorebug_rgb
            with_scorebug_rgb_np_u8[:, :, 3] = np.maximum(
                scorebug_rgba[:, :, 3],
                mask_hw_np_u8,
            )

            write_rgb_hwc_np_u8_to_jpg(
                rgb_hwc_np_u8=with_scorebug_rgb_np_u8[:, :, 0:3],
                out_abs_file_path=fake_original_path,
                verbose=True
            )
            write_rgba_hwc_np_u8_to_png(
                rgba_hwc_np_u8=with_scorebug_rgb_np_u8,
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

