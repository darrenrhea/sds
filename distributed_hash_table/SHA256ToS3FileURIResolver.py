import time
from print_yellow import (
     print_yellow
)
from is_sha256 import (
     is_sha256
)
from list_all_s3_keys_in_this_bucket_with_this_prefix import (
     list_all_s3_keys_in_this_bucket_with_this_prefix
)
import textwrap
from typing import Optional


class SHA256ToS3FileURIResolver(object):
    def __init__(self):
        self.reindex()
    
    def reindex(self):
        start_time = time.time()
        all_keys = list_all_s3_keys_in_this_bucket_with_this_prefix(
            bucket="awecomai-shared",
            prefix="sha256/"
        )
        stop_time = time.time()
        count = len(all_keys)
        duration = stop_time - start_time
        print_yellow(f"Took {duration=} seconds to index all {count} sha256s available in s3.")

        self.map_from_sha256_to_s3_file_uri = {}

        for key in all_keys:
            assert key.startswith("sha256/"), f"ERROR: {key=} should start with 'sha256/'"

            s3_file_uri = f"s3://awecomai-shared/{key}"
            
            file_name = key.split("/")[-1]
            
            full_length_sha256 = file_name.split(".")[0]
                        
            if len(full_length_sha256) != 64:
                print_yellow(
                    textwrap.dedent(
                        f"""\
                        "WARNING: for
                            {s3_file_uri=}
                        the prefix
                            {full_length_sha256=}
                        is not 64 characters long, skipping.
                        """
                    )
                )
                continue
            if not is_sha256(full_length_sha256):
                print_yellow(
                    textwrap.dedent(
                        f"""\
                        "WARNING: for
                            {s3_file_uri=}
                        the prefix
                            {full_length_sha256=}
                        is not a valid sha256, skipping.
                        """
                    )
                )
                continue
            
            self.map_from_sha256_to_s3_file_uri[full_length_sha256] = s3_file_uri
    
    def get(
        self,
        sha256: str,
    ) -> Optional[str]:
        return self.map_from_sha256_to_s3_file_uri.get(sha256, None)
    
        
        

    


