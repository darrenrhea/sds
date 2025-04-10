from SHA256ToS3FileURIResolver import (
     SHA256ToS3FileURIResolver
)
import sys
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
from print_green import (
     print_green
)


# s3_folder = "s3://awecomai-temp/records/"
# s3_folder = "s3://awecomai-temp/fake_bal/"
s3_folders = [
    "s3://awecomai-temp/fake_bal/",
    "s3://awecomai-temp/fake_records/chunk0_2025_04_04/",
    "s3://awecomai-temp/fake_records/chunk1_2025_04_04/",
    "s3://awecomai-temp/fake_records/chunk2_2025_04_04/",
    "s3://awecomai-temp/fake_records/chunk3_2025_04_04/",

    "s3://awecomai-temp/fake_records/g0_chunk0_2025_04_06/",
    "s3://awecomai-temp/fake_records/g0_chunk1_2025_04_06/",
    "s3://awecomai-temp/fake_records/g0_chunk2_2025_04_06/",
    "s3://awecomai-temp/fake_records/g0_chunk3_2025_04_06/",

    "s3://awecomai-temp/fake_records/g1_chunk0_2025_04_06/",
    "s3://awecomai-temp/fake_records/g1_chunk1_2025_04_06/",
    "s3://awecomai-temp/fake_records/g1_chunk2_2025_04_06/",
    "s3://awecomai-temp/fake_records/g1_chunk3_2025_04_06/",
]

sha256_to_s3_file_uri_resolver = SHA256ToS3FileURIResolver()

temp_dir_path = Path("~/a/bal_records").expanduser()
# temp_dir_path = Path("/shared/bal_records").expanduser()
temp_dir_path.mkdir(parents=True, exist_ok=True)

all_src_s3_file_uri_dst_file_path_pairs = []
for s3_folder in s3_folders:
    some_src_s3_file_uri_dst_file_path_pairs = (
        datfitsfitf_download_all_the_files_in_this_s3_folder_into_this_folder(
            s3_folder=s3_folder,
            dst_dir_path=temp_dir_path,
            glob_pattern="*",
            max_workers=100,
        )
    )
    all_src_s3_file_uri_dst_file_path_pairs.extend(
        some_src_s3_file_uri_dst_file_path_pairs
    )

print(f"ls {temp_dir_path}")

lst = []
for s3_file_uri, local_file_path in all_src_s3_file_uri_dst_file_path_pairs:
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

    extant = True
    for shaname in ["fake_original_sha256", "fake_mask_sha256"]:
        sha256 = obj[shaname]
        image_s3_file_uri = sha256_to_s3_file_uri_resolver.get(sha256)
        if image_s3_file_uri is None:
            print_red(f"Error: {shaname}={sha256} not stored in s3")
            extant = False
            break
    if not extant:
        print_red(f"aws s3 rm {s3_file_uri}")
        continue
    
    lst.append(obj)

print_green(f"num fake = {len(lst)}")
out_sha256 = sjaios_save_jsonable_as_its_own_sha256(
    obj=lst,
)

print(f"pri {out_sha256}")
