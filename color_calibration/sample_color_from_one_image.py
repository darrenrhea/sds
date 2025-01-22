import sys
from colorama import Fore, Style
from pathlib import Path
import numpy as np
import PIL.Image
from print_image_in_iterm2 import print_image_in_iterm2
from finitely_many_clicks_on_one_image import finitely_many_clicks_on_one_image


def sample_color_from_one_image(
    image_file_path: Path,
    instructions: str
):
    """
    Given the path to an image,
    this will pop up a full size window and instruct the user
    to click several times according to the instructions,
    then pull the color from those click locations and display it.
    """
    if not image_file_path.exists():
        print(f"{Fore.RED}ERROR: {image_file_path} does not exist!{Style.RESET_ALL}")
        sys.exit(1)

    image_pil = PIL.Image.open(str(image_file_path))
    image_np = np.array(image_pil)

    assert (
        image_np.dtype == np.uint8
    ), f"ERROR somehow dtype of image_np from {image_file_path} is NOT np.uint8?!"

    jsonable = finitely_many_clicks_on_one_image(
        image_file_path=image_file_path,
        instructions=instructions
    )

    samples = []
    for dct in jsonable:
        sample_id = f"s{np.random.randint(1000000000, 9999999999)}"
        x_pixel = dct["x_pixel"]
        y_pixel = dct["y_pixel"]
        r = int(image_np[y_pixel, x_pixel, 0])
        g = int(image_np[y_pixel, x_pixel, 1])
        b = int(image_np[y_pixel, x_pixel, 2])
        print(f"sample_id {sample_id} color at {x_pixel=}, {y_pixel=} is {r=}, {b=}, {g=}")
        radius = 10
        small = image_np[
            (y_pixel-radius):(y_pixel+radius+1),
            (x_pixel-radius):(x_pixel+radius+1),
            :
        ].copy()
        small[radius, radius, :] = 0
        larger = np.kron(
            small,
            np.ones(
                shape=(15, 15, 1),
                dtype=np.uint8
            )
        )

        print_image_in_iterm2(rgb_np_uint8=larger)
        sample = dict(
            sample_id=sample_id,
            image_id=image_file_path.name,
            x=x_pixel,
            y=y_pixel,
            r=r,
            g=g,
            b=b
        )
        samples.append(sample)

    who_to_murder_string = input("Give space-delimited list of samples to discard. Return for no one ")

    if who_to_murder_string == "":
        print("Discarded no samples.")
        who_to_murder = []
    else:
        who_to_murder = who_to_murder_string.split(" ")
        print("Discarded the samples:")
        for who in who_to_murder:
            print(f"   {who}")
    
    winnowed_samples = []
    for sample in samples:
        if sample["sample_id"] in who_to_murder:
            continue
        winnowed_samples.append(sample)
    
    return winnowed_samples

