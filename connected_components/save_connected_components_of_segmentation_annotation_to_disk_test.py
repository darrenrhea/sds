from prii import (
     prii
)
from get_a_temp_dir_path import (
     get_a_temp_dir_path
)
from save_connected_components_of_segmentation_annotation_to_disk import (
     save_connected_components_of_segmentation_annotation_to_disk
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)



def test_save_connected_components_of_segmentation_annotation_to_disk_1():
    original_path = get_file_path_of_sha256(
        "20d2c80a6de836761311d42aedf1697ab4e10c46527e8bee26341267241686fe"
    )
    prii(original_path)

    mask_path = get_file_path_of_sha256(
        "faf7580dea753c61c6466afa6a4c94c5d5bf598fafd709c5d41e35e524949b5f"
    )
    prii(mask_path)

    out_dir_path = get_a_temp_dir_path()
        
    save_connected_components_of_segmentation_annotation_to_disk(
        original_path=original_path,
        mask_path=mask_path,
        out_dir_path=out_dir_path,
        desired="small",
    )

    print(f"ls {out_dir_path}")
    for p in out_dir_path.glob("*.png"):
        prii(p)

     

if __name__ == "__main__":
    test_save_connected_components_of_segmentation_annotation_to_disk_1()
