from pathlib import Path
from print_image_in_iterm2 import print_image_in_iterm2
from open_just_the_rgb_part_of_image import open_just_the_rgb_part_of_image
import pytest


def test_open_just_the_rgb_part_of_image():
    
    image_paths = [
        Path("~/alpha_mattes_temp/hou-lac-2023-11-14_243526_nonfloor.png").expanduser().resolve(),
        Path("~/alpha_mattes_temp/hou-lac-2023-11-14_243526.jpg").expanduser().resolve(),
    ]

    for image_path in image_paths:
        mask = open_just_the_rgb_part_of_image(image_path=image_path)
        print_image_in_iterm2(image_path=image_path)


def test_open_just_the_rgb_part_of_image_should_fail_on_grayscale():
    """
    Tests that
    using open_just_the_rgb_part_of_image on a grayscale image should fail.
    """
    # This is a grayscale image, which we do not want silently escalated to an rgb image.
    image_path = Path("~/alpha_mattes_temp/hou-lac-2023-11-14_243526_relevance.png").expanduser().resolve()
    with pytest.raises(AssertionError) as exec_info:
        open_just_the_rgb_part_of_image(image_path=image_path)



if __name__ == "__main__":
    test_open_just_the_rgb_part_of_image()
    test_open_just_the_rgb_part_of_image_should_fail_on_grayscale()