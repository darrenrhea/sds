from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)

def download_the_files_with_these_sha256s(shas_to_download):
    for s in shas_to_download:
        get_file_path_of_sha256(s)
