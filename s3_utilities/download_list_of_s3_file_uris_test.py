from ls_with_glob_for_s3 import (
     ls_with_glob_for_s3
)
import sys
from download_list_of_s3_file_uris import (
     download_list_of_s3_file_uris
)
from pathlib import Path
import boto3


def make_src_dst_pairs(big_test: bool):
    # Specify the local directory where files will be saved
    dst_dir = Path("~/temp").expanduser()
    dst_dir.mkdir(parents=True, exist_ok=True)

    if big_test:
        s3_file_uris = ls_with_glob_for_s3(
            s3_pseudo_folder="s3://awecomai-shared/sha256/",
            glob_pattern="*.png",
            recursive=False  
        )
        
        for x in s3_file_uris:
            print(x)
        
    

        # Define your S3 bucket name and list of files to download
        src_dst_pairs = [
            (x, dst_dir / f"{k}.png")
            for k, x in enumerate(s3_file_uris)
        ]
    else:  # smaller, faster test
        src_dst_pairs = [
            ("s3://awecomai-shared/sha256/ffe28ab39608194cb2436a56e0ed892c83d39579b3f5a651b6a774b0ed95776c.png", Path(dst_dir/"0.png")),
            ("s3://awecomai-shared/sha256/ff28c3e01ee7a92ad4377dbd9db61952b4033d3e528896282bdc7907422ea1e3.png", Path(dst_dir/"1.png")),
            ("s3://awecomai-shared/sha256/ff3db3e6ae92129c44c39222792dc1440fbfa8414022aff0518b5cd67477b241.png", Path(dst_dir/"2.png")),
            ("s3://awecomai-shared/sha256/ff64372d783d669509bfc943fabf08247a95fd9866cd91dff83a07f751e6e6cf.png", Path(dst_dir/"3.png")),
            ("s3://awecomai-shared/sha256/ffcfad1b76fce7bdbf481ff6a74574fd304b1fd458d82410541f2f16a0dec9d5.png", Path(dst_dir/"4.png")),
            ("s3://awecomai-shared/sha256/f8ae3ba34a072c1ef7626c5dd7acf7efb8c6462aed735a52bdfa3e7f02c4962c.png", Path(dst_dir/"5.png")),
            ("s3://awecomai-shared/sha256/f7d60c3ec712538d413550ff7020f14e93d4602ce508c631c84902595bef3bc4.png", Path(dst_dir/"6.png")),
        ]
    
    return src_dst_pairs


def test_download_list_of_s3_file_uris():

    # Initialize the S3 client
    # the boto client is thread safe, so you can pass it out to the threads
    s3_client = boto3.client('s3')

    src_dst_pairs = make_src_dst_pairs(
        big_test=True
    )

    download_list_of_s3_file_uris(
        s3_client=s3_client,
        src_dst_pairs=src_dst_pairs
    )

  
if __name__ == "__main__":
    test_download_list_of_s3_file_uris()