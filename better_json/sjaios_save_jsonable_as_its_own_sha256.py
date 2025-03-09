from save_jsonable import (
     save_jsonable
)
from store_file_by_sha256 import (
     store_file_by_sha256
)
from get_a_temp_file_path import (
     get_a_temp_file_path
)

def sjaios_save_jsonable_as_its_own_sha256(
    obj,
    indent=4,
    sort_keys=False
):
    """
    Save a JSON-serializable object to a file, then store that file by its sha256 hash.
    Return the sha256 hash.
    """
    temp_file_path = get_a_temp_file_path(
        suffix=".json",
    )
    save_jsonable(
        fp=temp_file_path,
        obj=obj,
        indent=indent,
        sort_keys=sort_keys
    )
    sha256 = store_file_by_sha256(
        file_path=temp_file_path
    )
    return sha256
