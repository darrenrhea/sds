from sha256_of_file import (
     sha256_of_file
)
from sjaios_save_jsonable_as_its_own_sha256 import (
     sjaios_save_jsonable_as_its_own_sha256
)
from pathlib import Path
from store_file_by_sha256 import (
     store_file_by_sha256
)
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from print_green import (
     print_green
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from extract_all_rgba_frames_from_video_as_png import (
     extract_all_rgba_frames_from_video_as_png
)

shared_dir = get_the_large_capacity_shared_directory()

mother_dir_path = shared_dir / "scorebugs" / "from_video"
mother_dir_path.mkdir(parents=True, exist_ok=True)

folder = Path("~/Downloads/Awecom-2").expanduser()


mxf_file_paths = [
    p
    for p in folder.glob("*.mxf")
]

mxf_sha256s = [
    store_file_by_sha256(mxf_file_path)
    for mxf_file_path in mxf_file_paths
]

mxf_sha256s.sort()


scorebug_sha256s = []
for mxf_sha256 in mxf_sha256s:
    mxf_file_path = get_file_path_of_sha256(
        sha256=mxf_sha256,
        check=True,
    )

    out_dir_abs_path = mother_dir_path / mxf_sha256
    out_dir_abs_path.mkdir(parents=True, exist_ok=True)

    extract_all_rgba_frames_from_video_as_png(
        input_video_abs_file_path=mxf_file_path,
        out_dir_abs_path=out_dir_abs_path,
    )
    print_green(f"Extracted frames to {out_dir_abs_path}")
    for png_file_path in out_dir_abs_path.glob("*.png"):
        # png_sha256 = store_file_by_sha256(png_file_path)
        png_sha256 = sha256_of_file(png_file_path)
        print_green(f"pri {png_sha256}")
        scorebug_sha256s.append(png_sha256)

        
scorebug_sha256s = list(set(scorebug_sha256s))
scorebug_sha256s.sort()

scorebugs = [
    {
        "scorebug_sha256": png_sha256,
    }
    for png_sha256 in scorebug_sha256s
]

print_green("ff -r /Users/darrenrhea/a/scorebugs/from_video/")

jsonable = dict(
    description="BAL scorebugs derived from RGBA mxf videos",
    scorebugs=scorebugs,
)

scorebugs_derived_from_videos_sha256 = sjaios_save_jsonable_as_its_own_sha256(
    obj=jsonable,
    indent=4,
    sort_keys=False
)

print_green(f"{scorebugs_derived_from_videos_sha256=}")