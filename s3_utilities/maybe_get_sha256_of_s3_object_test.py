from maybe_get_sha256_of_s3_object import (
     maybe_get_sha256_of_s3_object
)

def test_maybe_get_sha256_of_s3_object_1(s3_object_url: str):
    s3_object_url = "s3://sha-256/cd/63/57/ef/cd6357efdd966de8c0cb2f876cc89ec74ce35f0968e11743987084bd42fb8944.txt"
    ans = maybe_get_sha256_of_s3_object(s3_object_url=s3_object_url)
    print(ans)
   