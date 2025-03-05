from gaes3fuwts_get_an_extant_s3_file_uri_with_this_sha256 import (
     gaes3fuwts_get_an_extant_s3_file_uri_with_this_sha256
)



def test_gaes3fuwts_get_an_extant_s3_file_uri_with_this_sha256():
    
    sha256 = "af6e2c6e133dc66d1b9a94f336863db3a13aa9b4aa2daa6b4720fa5c29c189db"
    
    answer = gaes3fuwts_get_an_extant_s3_file_uri_with_this_sha256(
        sha256=sha256
    )

    print(answer)
   
if __name__ == "__main__":
    test_gaes3fuwts_get_an_extant_s3_file_uri_with_this_sha256()