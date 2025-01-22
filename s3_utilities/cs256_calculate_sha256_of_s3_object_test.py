from cs256_calculate_sha256_of_s3_object import (
     cs256_calculate_sha256_of_s3_object
)



def test_cs256_calculate_sha256_of_s3_object_1():
    """
    printf "kitty" > kitty.txt
    aws s3 cp kitty.txt s3://awecomai-test-videos/nba/Mathieu/kitty.txt
    """
    test_size = "medium"
    if test_size == "small":
        ans = cs256_calculate_sha256_of_s3_object(
            s3_object_url="s3://awecomai-test-videos/nba/Mathieu/kitty.txt"
        )
        assert (
            ans == "67731ff58137eb39713ae30eba33c54c8c1d5418e081428ca815e4e733d64f6d"
        ), f"ERROR: Expected {ans=} to be '67731ff58137eb39713ae30eba33c54c8c1d5418e081428ca815e4e733d64f6d' but it was {ans=}"
    elif test_size == "medium":  # anything less than 5_368_709_120 bytes
        ans = cs256_calculate_sha256_of_s3_object(
            s3_object_url="s3://awecomai-show-n-tell/out_EB_23-24_R01_BAY-BER_hq_fullgame_0322.ts"
        )
        expected  = "81bc3730c82610a0937bf70449dbf2cb7295f86939fdc536c07e2f54c7ec8bc5"
        assert (
            ans == expected
        ), f"ERROR: Expected the sha256 to be {expected=} but it was {ans=}"
    elif test_size == "large":
        ans = cs256_calculate_sha256_of_s3_object(
            s3_object_url="s3://awecomai-test-videos/nba/Mathieu/EB_23-24_R20_BAY-RMB.mxf"
        )
        assert (
            ans == "3be28eecc4877b4c30064522c011095ec6917d412b106dec5aee254e536f9071"
        ), f"ERROR: Expected {ans=} to be 3be28eecc4877b4c30064522c011095ec6917d412b106dec5aee254e536f9071 but it was {ans=}"