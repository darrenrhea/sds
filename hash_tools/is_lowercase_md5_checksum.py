import re


def is_lowercase_md5_checksum(md5_str: str) -> bool:
    if not len(md5_str) == 32:
        return False
    if not isinstance(md5_str, str):
        return False
    # check is hexidecimal
    if not re.match(r"^[0-9a-f]{32}$", md5_str):
        return False
    return True


assert     is_lowercase_md5_checksum("012345678901234567890123456789ab")
assert not is_lowercase_md5_checksum("012345678901234567890123456789aB")
assert not is_lowercase_md5_checksum("012345678901234567890123456789az")
assert not is_lowercase_md5_checksum("012345678901234567890123456789ab0")
assert not is_lowercase_md5_checksum("012345678901234567890123456789")
