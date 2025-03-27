from print_red import (
     print_red
)
from sjaios_save_jsonable_as_its_own_sha256 import (
     sjaios_save_jsonable_as_its_own_sha256
)
from datfitsfitf_download_all_the_files_in_this_s3_folder_into_this_folder import (
     datfitsfitf_download_all_the_files_in_this_s3_folder_into_this_folder
)
import better_json as bj
from pathlib import Path


# s3_folder = "s3://awecomai-temp/records/"
# s3_folder = "s3://awecomai-temp/fake_bal/"
s3_folder = "s3://awecomai-temp/fake_records/24-25_HOU_CORE/"

# temp_dir_path = Path("~/bal_records").expanduser()  # get_a_temp_dir_path()
temp_dir_path = Path("/shared/hou_core_records").expanduser()  # get_a_temp_dir_path()
temp_dir_path.mkdir(parents=True, exist_ok=True)

src_s3_file_uri_dst_file_path_pairs = (
    datfitsfitf_download_all_the_files_in_this_s3_folder_into_this_folder(
        s3_folder=s3_folder,
        dst_dir_path=temp_dir_path,
        glob_pattern="*",
        max_workers=100,
    )
)

print(f"ls {temp_dir_path}")

lst = []
for s3_file_uri, local_file_path in src_s3_file_uri_dst_file_path_pairs:
    try:
        obj = bj.load(local_file_path)
    except Exception as e:
        print_red(f"Error loading {local_file_path}, whatever.")
        sys.exit(1)

    expected_keys = [
        "clip_id",
        "frame_index",
        "fake_original_sha256",
        "fake_mask_sha256",
    ]

    for key in expected_keys:
        assert key in obj
    
    lst.append(obj)

out_sha256 = sjaios_save_jsonable_as_its_own_sha256(
    obj=lst,
)

print(f"pri {out_sha256}")
