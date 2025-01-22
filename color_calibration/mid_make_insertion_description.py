import argparse
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
import pyperclip
from convert_camera_pose_to_jsonable import (
     convert_camera_pose_to_jsonable
)
from pathlib import Path
import subprocess
from sha256_of_file import (
     sha256_of_file
)
from get_video_frame_path_from_clip_id_and_frame_index import (
     get_video_frame_path_from_clip_id_and_frame_index
)
import uuid
from store_file_by_sha256 import (
     store_file_by_sha256
)
import sys
import better_json as bj


def mid_make_insertion_description_cli_tool():
    if len(sys.argv) < 2:
        print(
            "\n"
            "# Pick an ad:\n"
            "ls -1 ~/r/nba_ads\n\n"

            "# Look at it:\n"
            "pri ~/r/nba_ads/ESPN_DAL_LAC_NEXT_ABC.jpg\n\n"
            
            "# Find a frame index where it is displayed amongst the human annotated frames:\n"
            "open ~/r/bos-mia-2024-04-21-mxf_led/.approved\n\n"
            
            "# Then run this:\n\n"
            
            "Usage: python mid_make_insertion_description.py <clip_id> <frame_index> <ad_file_path>\n\n"
            
        )
        sys.exit(1)

    insertion_description_id = str(uuid.uuid4())
    parser = argparse.ArgumentParser()
    parser.add_argument("clip_id", type=str)
    parser.add_argument("frame_index", type=int)
    parser.add_argument("ad_file_path", type=Path)
    args = parser.parse_args()
    clip_id = args.clip_id
    frame_index = args.frame_index
    ad_file_path = args.ad_file_path.resolve()
    assert ad_file_path.is_file()
    store_file_by_sha256(ad_file_path)
    ad_sha256 = sha256_of_file(ad_file_path)

    original_file_path = get_video_frame_path_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )

    camera_pose = get_camera_pose_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index
    )


    gross_ad = ad_file_path.stem

    original_sha256 = sha256_of_file(original_file_path)



    args = [
        "open",
        "-a",
        "Adobe Photoshop 2024",
        f"{original_file_path}"
    ]

    print("Save the image in photoshop by exporting to png to your home folder, then press enter")

    subprocess.run(
        args=args
    )

    print("Press enter to continue once you saved the image in photoshop")
    input()
    
    conservative_mask_png_path = Path.home() / original_file_path.with_suffix(".png").name
    assert conservative_mask_png_path.is_file(), f"png_path {conservative_mask_png_path} not found!  Did you save the image in photoshop?"

    store_file_by_sha256(conservative_mask_png_path)

    conservative_mask_sha256 = sha256_of_file(conservative_mask_png_path)
    
    out_dir = Path(
        "~/r/color_correction_data/insertion_descriptions"
    ).expanduser()
    
    camera_pose_jsonable = convert_camera_pose_to_jsonable(
        camera_pose=camera_pose
    )

    jsonable = dict(
        uuid=insertion_description_id,
        ad_placement_descriptor=dict(
            tl=[-9.9, 30.339, 2.605],
            bl=[-9.9, 30.339, 0.3],
            tr=[9.95, 30.339,    2.605],
            br=[9.80, 30.339,    0.3],
            origin=[-9.81125, 30.339, 0.202239],
            u=[1.0, 0.0, 0.0 ],
            v=[0.0, 0.0, 1.0],
            height=2.458,
            width=19.74,
        ),
        gross_ad=gross_ad,
        domain=dict(
            file_name=ad_file_path.name,
            subrectangle=dict(
                j_min=0,
                j_max=1536,
                i_min=0,
                i_max=192,
            ),
            sha256=ad_sha256,
        ),
        codomain=dict(
            clip_id=clip_id,
            frame_index=frame_index,
            sha256=original_sha256,
            mask_for_regression=dict(
                sha256=conservative_mask_sha256,
            ),
        ),
        camera_pose=camera_pose_jsonable,
    )

    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / f"{insertion_description_id}.json5"
    bj.dump(
        obj=jsonable,
        fp=out_path,
    )

    print(f"bat {out_path}")

    s = f"export insertion_description_id={insertion_description_id}"
    pyperclip.copy(s)
    print("We suggest you run the following command:")
    print(s)
    print("you can just paste since it is on the clipboard")

if __name__ == "__main__":
    mid_make_insertion_description_cli_tool()