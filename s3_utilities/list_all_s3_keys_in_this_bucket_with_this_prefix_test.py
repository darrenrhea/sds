from color_print_json import (
     color_print_json
)
from list_all_s3_keys_in_this_bucket_with_this_prefix import (
     list_all_s3_keys_in_this_bucket_with_this_prefix
)


def test_list_objects_with_this_prefix_1():

    bucket = "awecomai-shared"
    prefix = "sha256/af6e2c6e133dc66"

    lst = list_all_s3_keys_in_this_bucket_with_this_prefix(
        bucket=bucket,
        prefix=prefix
    )
    color_print_json(lst)
    
    
if __name__ == "__main__":
    test_list_objects_with_this_prefix_1()