from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)



def test_get_file_path_of_sha256_1():
    sha256 = "af6e2c6e133dc66d1b9a94f336863db3a13aa9b4aa2daa6b4720fa5c29c189db"
    file_path = get_file_path_of_sha256(
        sha256=sha256,
        check=True
    )
    print(f"{file_path=!s}")
   
if __name__ == "__main__":
    test_get_file_path_of_sha256_1()