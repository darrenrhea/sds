from prii import (
     prii
)
import shutil
import textwrap
from gfposaiwad_get_file_path_of_sha256_assuming_it_was_already_downloaded import (
     gfposaiwad_get_file_path_of_sha256_assuming_it_was_already_downloaded
)

from color_print_json import (
     color_print_json
)
from ltjfthts_load_the_jsonlike_file_that_has_this_sha256 import (
     ltjfthts_load_the_jsonlike_file_that_has_this_sha256
)
from print_green import (
     print_green
)
from pathlib import Path

all_segmentation_annotations_sha256 = "6d7074c40a5aa53286f14e8127d2822f9e5ccb68bee112fa6e43f10f4c6a8485"

all_segmentation_annotations = ltjfthts_load_the_jsonlike_file_that_has_this_sha256(
    sha256_of_the_jsonlike_file=all_segmentation_annotations_sha256,
    check=True
)

out_dir_path = Path(
    "~/gsw_old_data"
).expanduser()
out_dir_path.mkdir(parents=True, exist_ok=True)

count = 0
appropriate_annotations = []
for ann in all_segmentation_annotations:
    clip_id = ann["clip_id"]
    frame_index = ann["frame_index"]
    original_sha256 = ann["original_sha256"]
    mask_sha256 = ann["mask_sha256"]
    info = ann["info"]
    court = info["court"]

    gsw = (
        clip_id.lower().startswith("gsw")
        and clip_id != "gsw"
        and
        court != "gsw_city_2223"
    )
    if gsw:
        count += 1
    
    if not gsw:
        continue
    appropriate_annotations.append(ann)
    
for ann in appropriate_annotations:
    clip_id = ann["clip_id"]
    frame_index = ann["frame_index"]
    original_sha256 = ann["original_sha256"]
    mask_sha256 = ann["mask_sha256"]

    mask_path = gfposaiwad_get_file_path_of_sha256_assuming_it_was_already_downloaded(
        sha256=mask_sha256,
        check=False,
    )
    assert mask_path is not None
    original_path = gfposaiwad_get_file_path_of_sha256_assuming_it_was_already_downloaded(
        sha256=original_sha256,
        check=False,
    )
    assert original_path is not None
    
    color_print_json(
        ann
    )
    prii(original_path)

    ann_id = f"{clip_id}_{frame_index}"
    if original_path.suffix == ".png":
        jpg_or_png = "png"
    elif original_path.suffix == ".jpg":
        jpg_or_png = "jpg"
    else:
         raise ValueError(
            f"original_path.suffix={original_path.suffix} is not .jpg or .png"
        )
    
    if jpg_or_png == "jpg":
        original_name = f"{ann_id}_original.jpg"
    elif jpg_or_png == "png":
        original_name = f"{ann_id}_original.png"
    mask_name = f"{ann_id}_nonfloor.png"
    
    shutil.copy(
        src=original_path,
        dst=Path(out_dir_path) / original_name
    )

    shutil.copy(
        src=mask_path,
        dst=Path(out_dir_path) / mask_name
    )

print_green(
    textwrap.dedent(
        f"""\
        Found {count} gsw annotations
        and stuck them in {out_dir_path}
        """
    )
)

print(
    """\
    rm -rf ~/gsw_old_data
    mkdir -p ~/gsw_old_data
    rsync -rP dl:/home/darren/gsw_old_data/ ~/gsw_old_data/

    ff ~/gsw_old_data/
    """
)
    
