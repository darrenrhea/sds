import pytest
from s3_file_uri_exists import (
     s3_file_uri_exists
)

def test_s3_file_uri_exists_1():
    nonexistent_s3_file_uri = "s3://awecomai-show-n-tell/a/b/c.txt"
    assert not s3_file_uri_exists(nonexistent_s3_file_uri)
    extant_s3_file_uri = "s3://awecomai-shared/sha256/6c6ac7095231d652d61cdc2e3a171991f7a94fc614ae0193e67039320072938a.json"
    assert s3_file_uri_exists(extant_s3_file_uri)

    nonexistent_bucket = "s3://nonexistent-bucket/a/b/c.txt"
    with pytest.raises(Exception):
         assert s3_file_uri_exists(nonexistent_bucket)


if __name__ == '__main__':
    test_s3_file_uri_exists_1()
    print("s3_file_uri_exists PASSED tests.")