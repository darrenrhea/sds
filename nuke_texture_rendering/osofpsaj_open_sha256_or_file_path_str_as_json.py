from pathlib import Path

from gfpfwmbasoafoafps_get_file_path_from_what_might_be_a_sha256_of_a_file_or_a_file_path_str import gfpfwmbasoafoafps_get_file_path_from_what_might_be_a_sha256_of_a_file_or_a_file_path_str
import better_json as bj


def osofpsaj_open_sha256_or_file_path_str_as_json(
    sha256_or_local_file_path_str: str
) -> Path:
    """
    We have a lot of JSON-like files that we refer to by their SHA256 hash.
    """
    local_file_path = gfpfwmbasoafoafps_get_file_path_from_what_might_be_a_sha256_of_a_file_or_a_file_path_str(
        sha256_or_local_file_path_str
    )
    jsonable = bj.load(local_file_path)
    return jsonable
