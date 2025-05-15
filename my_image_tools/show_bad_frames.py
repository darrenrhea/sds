import textwrap
from get_annotation_id_clip_id_and_frame_index_from_mask_file_name import (
     get_annotation_id_clip_id_and_frame_index_from_mask_file_name
)
import argparse
from pric import show_mask
from pathlib import Path
import sys
import shutil

from image_openers import open_image_as_rgb_np_uint8_ignoring_any_alpha
from print_image_in_iterm2 import print_image_in_iterm2
from get_the_large_capacity_shared_directory import get_the_large_capacity_shared_directory




def show_bad_frames_cli():

    argp = argparse.ArgumentParser(
        description=textwrap.dedent(
            f"""\
            show_bad_frames bay-zal-2024-03-15-mxf-yadif fe3
            """
        )
    ) 
    argp.add_argument("clip_id", type=str)
    argp.add_argument("model_id", type=str)
    opt = argp.parse_args()
    clip_id = opt.clip_id
    assert clip_id in ["bay-zal-2024-03-15-mxf-yadif"]
    model_id = opt.model_id
    shared_dir = get_the_large_capacity_shared_directory()
    
    r_dir = Path("~/r").expanduser()


    repo_dir = r_dir / "bay-zal-2024-03-15-mxf-yadif_led"

    clip_id_frame_index_rel_repo_path_pairs = []
    for mask_path in repo_dir.rglob("*_nonfloor.png"):
        (
            _,
            clip_id,
            frame_index
        ) = get_annotation_id_clip_id_and_frame_index_from_mask_file_name(
            mask_file_name=mask_path.name
        )
        
        mask_path_rel_to_r_dir = mask_path.relative_to(r_dir)

        clip_id_frame_index_rel_repo_path_pairs.append(
            (clip_id, frame_index, mask_path_rel_to_r_dir)
        )

   
    bad_frames_dir = Path(
        f"~/bad_frames/{model_id}"
    ).expanduser()
    bad_frames_dir.mkdir(exist_ok=True, parents=False)



    for clip_id, frame_index, rel_r_path in clip_id_frame_index_rel_repo_path_pairs:
        shutil.copy(
            src=shared_dir / "clips" / clip_id / "frames" / f"{clip_id}_{frame_index:06d}_original.jpg",
            dst=bad_frames_dir / f"{clip_id}_{frame_index:06d}_original.jpg"
        )

    

    for clip_id, frame_index, rel_r_path in clip_id_frame_index_rel_repo_path_pairs:
        alpha_path = shared_dir / "inferences" / f"{clip_id}_{frame_index:06d}_{model_id}.png"
        rgb_path =  shared_dir / "clips" / clip_id / "frames" / f"{clip_id}_{frame_index:06d}_original.jpg"

        if not rgb_path.exists():
            print(f"rgb_path does not exist: {rgb_path}")
            sys.exit(1)
        if not alpha_path.exists():
            print(f"alpha_path does not exist: {alpha_path}, you may need to infer model on the bad_frames directory")
            sys.exit(1)
        
        print(f"This is {rel_r_path}/{clip_id}_{frame_index:06d}_nonfloor.png:")
        show_mask(
            rgb_path=rgb_path,
            alpha_path=alpha_path,
            invert=True,
            saveas=None
        )
        show_mask(
            rgb_path=rgb_path,
            alpha_path=alpha_path,
            invert=False,
            saveas=None
        )
        rgb_np_uint8 = open_image_as_rgb_np_uint8_ignoring_any_alpha(
            abs_file_path=rgb_path
        )
        print_image_in_iterm2(
            rgb_np_uint8=rgb_np_uint8
        )

        