from print_yellow import (
     print_yellow
)
from print_red import (
     print_red
)
from sha256_of_file import (
     sha256_of_file
)
import time
from download_s3_file_uris_to_file_paths import (
     download_s3_file_uris_to_file_paths
)
from glfptsfws_get_local_file_path_to_save_file_with_sha256 import (
     glfptsfws_get_local_file_path_to_save_file_with_sha256
)
from SHA256ToS3FileURIResolver import (
     SHA256ToS3FileURIResolver
)


def download_the_files_with_these_sha256s(
    sha256s_to_download,
    max_workers: int,
    verbose: bool,
):
    """
    We usually describe file assets by their sha256 hash.
    This function will download a bunch of files from s3 as specified by their sha256 hashes.
    """
    sha256_to_s3_file_uri_resolver = SHA256ToS3FileURIResolver()

    
    sha256_src_s3_file_uri_dst_file_path_triplets = []
    for sha256 in sha256s_to_download:
        s3_file_uri = sha256_to_s3_file_uri_resolver.get(sha256)
        assert s3_file_uri is not None, f"ERROR: {sha256=} does not have a corresponding s3 file uri"
        file_name_of_s3_file_uri = s3_file_uri.split("/")[-1]
        recapitulated = file_name_of_s3_file_uri[:64]
        assert recapitulated == sha256, f"ERROR: {recapitulated=} != {sha256=}"
        extension = file_name_of_s3_file_uri[64:]


        local_file_path = glfptsfws_get_local_file_path_to_save_file_with_sha256(
            sha256=sha256,
            extension=extension,
        )
        sha256_src_s3_file_uri_dst_file_path_triplets.append(
            (sha256, s3_file_uri, local_file_path)
        )
    
    # BEGIN remove things we already have from download list:
    start_see_if_we_already_have_a_good_copy = time.time()
    triplets_to_actually_download = []
    for triplet in sha256_src_s3_file_uri_dst_file_path_triplets:
        sha256, s3_file_uri, local_file_path = triplet

        already_have_a_good_copy = False
        if local_file_path.is_file():
            local_sha256 = sha256_of_file(local_file_path)
            if local_sha256 == sha256:
                already_have_a_good_copy = True
            else:
                print_red("ERROR: Already have a file, but its sha256 different")

        
        if not already_have_a_good_copy:
            triplets_to_actually_download.append(triplet)
    
    end_see_if_we_already_have_a_good_copy = time.time()
    duration_see_if_we_already_have_a_good_copy = (
        end_see_if_we_already_have_a_good_copy
        -
        start_see_if_we_already_have_a_good_copy
    )
    print_yellow(f"Took {duration_see_if_we_already_have_a_good_copy} seconds to see if we already have a good copy of the files.")
    # ENDOF remove things we already have from download list

    src_s3_file_uri_dst_file_path_pairs = [
        (s3_file_uri, local_file_path)
        for _, s3_file_uri, local_file_path in triplets_to_actually_download
    ]
    
    # TODO: make the concurrent thing check the shas256s: 
    download_s3_file_uris_to_file_paths(
        src_s3_file_uri_dst_file_path_pairs=\
        src_s3_file_uri_dst_file_path_pairs,
        
        max_workers=\
        max_workers,
        
        verbose=\
        verbose,
    )
