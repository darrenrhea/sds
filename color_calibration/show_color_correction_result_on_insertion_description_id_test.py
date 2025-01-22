from load_color_correction_from_json import (
     load_color_correction_from_json
)
from show_color_correction_result_on_insertion_description_id import (
     show_color_correction_result_on_insertion_description_id
)

from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)

from pathlib import Path



def test_show_color_correction_result_on_insertion_description_id_1():
    
    # color_correction_sha256 = "bd545cba8ac10558b8a5a4eeba40bc3be9f1e809975fd7e6ad38d6a3ac598140"
    color_correction_sha256 = "4edceff5771335b7a64b1507fa1d31f38f5148f71322092c4db5ecd8ec6e985b"
    color_correction_json_path = get_file_path_of_sha256(color_correction_sha256)
    print(f"loading color correction from {color_correction_json_path}")
    
    degree, coefficients = load_color_correction_from_json(
        json_path=color_correction_json_path
    )
    print(coefficients)
    insertion_description_id = "2a04b7dd-8d83-4455-927e-002b16b11128"

    show_color_correction_result_on_insertion_description_id(
        use_linear_light=True,
        degree=degree,
        coefficients=coefficients,
        # give the self_reproducing_insertion_description_id:
        insertion_description_id=insertion_description_id,
        # where to save it for flipflopping
        out_dir=Path.home() / "ccc"
    )

    print("ff ~/ccc")

if __name__ == "__main__":
    test_show_color_correction_result_on_insertion_description_id_1()