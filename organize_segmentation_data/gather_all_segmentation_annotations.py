from store_file_by_sha256 import (
     store_file_by_sha256
)
from sha256_of_file import (
     sha256_of_file
)
from color_print_json import (
     color_print_json
)
import better_json as bj
from get_clip_id_to_info import (
     get_clip_id_to_info
)
from get_clip_id_and_frame_index_from_mask_file_name import (
     get_clip_id_and_frame_index_from_mask_file_name
)
from maybe_find_sister_original_path_of_this_mask_path import (
     maybe_find_sister_original_path_of_this_mask_path
)
from print_green import (
     print_green
)
from pathlib import Path

mother_dir_str = [
    "/shared/all_training_data/floor_not_floor",  # 1541
    "/mnt/nas/volume1/videos/floor_not_floor",  # 1541
    "/mnt/nas/volume1/videos/segmentation_datasets",  # 960, but with depthmaps, camera poses
][1]

mother_dir = Path(mother_dir_str)

all_abs_file_paths = []
for p in mother_dir.rglob("*"):
    if not p.is_file():
        continue
    all_abs_file_paths.append(p)

all_abs_file_paths = sorted(all_abs_file_paths)
for p in all_abs_file_paths:
    print(p)
print_green(f"That is {len(all_abs_file_paths)} files.")


all_nonfloor_paths = [
    p for p in all_abs_file_paths
    if "nonfloor" in p.name
]
print_green(f"That is {len(all_nonfloor_paths)} nonfloor containing files.")


all_original_paths = [
    p for p in all_abs_file_paths
    if "jpg" in p.name
]
print_green(f"That is {len(all_original_paths)} jpg files.")

remainder = set(all_abs_file_paths) - set(all_original_paths) - set(all_nonfloor_paths)

remainder = sorted(list(remainder))
for p in remainder:
    print(f"rm {p}")


clip_id_to_info = get_clip_id_to_info()

annotations = []
count = 0
for mask_path in all_nonfloor_paths:
    original_path = maybe_find_sister_original_path_of_this_mask_path(
        mask_path=mask_path
    )
    clip_id, frame_index = get_clip_id_and_frame_index_from_mask_file_name(
            file_name=mask_path.name
    )
    assert clip_id in clip_id_to_info, f"clip_id {clip_id} not in clip_id_to_info"
    info = clip_id_to_info[clip_id]
    assert "court" in info, f"info {info} does not have court"
    assert "game_id" in info, f"info {info} does not have game_id"
    game_id = info["game_id"]
    court = info["court"]
    original_sha256 = sha256_of_file(original_path)
    mask_sha256 = sha256_of_file(mask_path)
    store_file_by_sha256(
        file_path=mask_path,
        verbose=True,
    )
    store_file_by_sha256(
        file_path=original_path,
        verbose=True,
    )

    dct = {
        "clip_id": clip_id,
        "frame_index": frame_index,
        "mask_sh256": mask_sha256,
        "original_sha256": original_sha256,
        "court_id": court,
        "game_id": game_id,
        # "mask_path": str(mask_path),
        # "original_path": str(original_path),
    }
    annotations.append(dct)
    count += 1
    print(f"{count=}")

out_file_path = Path("~/all_segmentation_annotations.json").expanduser()
bj.dump(obj=annotations, fp=out_file_path)
print_green(f"bat {out_file_path}")


    