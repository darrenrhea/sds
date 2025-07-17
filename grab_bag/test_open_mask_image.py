from pathlib import Path
from print_image_in_iterm2 import print_image_in_iterm2
from open_mask_image import open_mask_image

def test_open_mask_image():
    
    mask_paths = [
        Path("~/alpha_mattes_temp/hou-lac-2023-11-14_243526_nonfloor.png").expanduser().resolve(),
        Path("~/alpha_mattes_temp/hou-lac-2023-11-14_243526_relevance.png").expanduser().resolve(),
    ]

    for mask_path in mask_paths:
        mask = open_mask_image(mask_path=mask_path)
        print_image_in_iterm2(image_path=mask_path)
        print_image_in_iterm2(grayscale_np_uint8=mask)



if __name__ == "__main__":
    test_open_mask_image()