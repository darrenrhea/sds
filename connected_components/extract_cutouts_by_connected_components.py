from pathlib import Path
from prii import (
     prii
)
from get_a_temp_dir_path import (
     get_a_temp_dir_path
)
from save_connected_components_of_segmentation_annotation_to_disk import (
     save_connected_components_of_segmentation_annotation_to_disk
)
from get_mask_path_from_original_path import (
     get_mask_path_from_original_path
)



def extract_cutouts_by_connected_components():
    folder = Path(
        "~/r/hou-sas-2024-10-17-sdi_floor/.approved"
    ).expanduser()

    for original_path in folder.glob("*_original.jpg"):
        mask_path = get_mask_path_from_original_path(
            original_path=original_path
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
    extract_cutouts_by_connected_components()