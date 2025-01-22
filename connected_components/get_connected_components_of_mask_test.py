from pathlib import Path
from get_connected_components_of_mask import get_connected_components_of_mask



def test_get_connected_components_of_mask_1():
    original_path = Path(
        "~/r/munich4k_importantpeople/anna/DSCF0241_000097.jpg"
    ).expanduser()

    mask_path = Path(
        "~/r/munich4k_importantpeople/anna/DSCF0241_000097_nonfloor.png"
    ).expanduser()

    out_dir_path = Path(
        "temp"
    ).resolve()
        
    get_connected_components_of_mask(
        original_path=original_path,
        mask_path=mask_path,
        out_dir_path=out_dir_path
    )

     

if __name__ == "__main__":
    test_get_connected_components_of_mask_1()
