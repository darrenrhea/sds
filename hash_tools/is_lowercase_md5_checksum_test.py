from is_lowercase_md5_checksum import (
     is_lowercase_md5_checksum
)


def test_is_lowercase_md5_checksum_1():
    assert     is_lowercase_md5_checksum("012345678901234567890123456789ab")
    assert not is_lowercase_md5_checksum("012345678901234567890123456789aB")
    assert not is_lowercase_md5_checksum("012345678901234567890123456789az")
    assert not is_lowercase_md5_checksum("012345678901234567890123456789ab0")
    assert not is_lowercase_md5_checksum("012345678901234567890123456789")
