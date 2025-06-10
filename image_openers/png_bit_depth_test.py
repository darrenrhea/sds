from print_green import (
     print_green
)
from png_bit_depth import (
     png_bit_depth
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)

def test_png_bit_depth():  
    test_cases = [
        dict(
            sha256="f33eb08009221f1b25c5ba41f8beae1006d87e225619e0723f6ad71888d79a9d",
            should_be=8
        ),
        dict(
            sha256="5f32b1382067703a4ca4d1f0c0a507f2a97ea2e821afb1ed4e9ff62ba1c9f5c9",
            should_be=16
        ),
    ]
    for test_case in test_cases:
        sha256 = test_case["sha256"]
        expected_bit_depth = test_case["should_be"]
        file_path = get_file_path_of_sha256(sha256=sha256, check=True)
        bit_depth = png_bit_depth(path=file_path)
        assert bit_depth == expected_bit_depth, f"Expected {expected_bit_depth}, got {bit_depth}"
 

if __name__ == "__main__":
    test_png_bit_depth()
    print_green("test_png_bit_depth passed!")