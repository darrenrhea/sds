from prii_histogram import (
     prii_histogram
)
from use_polynomial_regression_coefficients import (
     use_polynomial_regression_coefficients
)
from colorama import Fore, Style
import pyperclip
from save_color_correction_as_json import (
     save_color_correction_as_json
)
from show_color_correction_result_on_insertion_description_id import (
     show_color_correction_result_on_insertion_description_id
)
from get_color_correction_polynomial_coefficients_from_from_to_mapping_array_f64 import (
     get_color_correction_polynomial_coefficients_from_from_to_mapping_array_f64
)
from get_from_to_mapping_array_f64_from_insertion_description_id import (
     get_from_to_mapping_array_f64_from_insertion_description_id
)
from pathlib import Path
import numpy as np



def fccfi_fit_color_correction_from_insertions():
    """
    Let's do
    ESPN_MIL_IND_FRI
    because it has a black background and red yellow blue and green and white.

    A "self_reproducing_insertion_description" is a bundle of:
    an original video frame
    its camera pose
    A mask that selects a large part of the LED board, but which
    definitely does not contain people nor other objects occlude the LED board.
    A 2-dimensional ad jpg that they sent us, that was been displayed on the the LED board at that time the video frame was taken.
    Maybe a description of a subrectangle of that image,
    because actually we insert only a subrectangle of the image.
    3D world coordinates of the 4 corners of the LED board that tell us where to insert.
    """
    
    degree = 1
    use_linear_light = True

    # insertion_description_id = "6a9a25fb-9fbc-4fc2-9e14-cb6105b3d249"
    # insertion_description_id = "41bb4d0b-d2cf-4d15-8482-abb6493520ba"
    insertion_description_ids = [
        "2a04b7dd-8d83-4455-927e-002b16b11128",
        # "0e4d1982-baeb-4b04-8c18-774f3bce4084", 
        # "60820db1-9028-4566-86a0-84d6056168fb",
        # "fd217680-07d7-4f0f-a8ff-8a2bdf2a25e4",
        # "48964eb8-573e-4455-9db6-e6874c66ef62",
        # "e8d4d8d2-0409-4eb1-a993-f0a7d8ce58ab",
    ]

    # sometimes you want to force certain pairs hard:
    add_from_to_pairs_manually = True
    if add_from_to_pairs_manually:
        described_additional_from_to_pairs = [
            dict(
                from_=[30, 89, 50],
                to=[35, 89, 65],
                explanation="Emerald green of Milwaukee Bucks",
                weight=1.0,
            ),
            dict(
                from_=[227, 0, 28],
                to=[221, 45, 76],
                explanation="Emerald green of Milwaukee Bucks",
                weight=1.0,
            ),
            dict(
                from_=[0, 0, 0],
                to=[44, 50, 49],
                explanation="Emerald green of Milwaukee Bucks",
                weight=1.0,
            ),
            dict(
                from_=[255, 255, 255],
                to=[243, 255, 255],
                explanation="White of Different Here",
                weight=0.0,
            ),
            dict(
                from_=[32, 32, 37],
                to=[0, 12, 50],
                explanation="",
                weight=0.0,
            ),
            dict(
                from_=[195, 29, 41],
                to=[133, 33, 79],
                explanation="",
                weight=0.0,
            ),
            dict(
                from_=[29, 29, 29],
                to=[65, 69, 66],
                explanation="",
                weight=0.0,
            ),
            dict(
                from_=[0, 0, 0],
                to=[69, 63, 58],
                explanation="ESPN+ background color",
                weight=0.0,
            ),
            dict(
                from_=[188, 32, 39],
                to=[165, 34, 44],
                explanation="ESPN+ background color",
                weight=0.0,
            ),
        ]
    else:  # no, do it purely from insertions
        described_additional_from_to_pairs = []
        


    if len(described_additional_from_to_pairs) == 0:
        print(f"{Fore.YELLOW}No additional_from_to_pairs were given.{Style.RESET_ALL}")
        additional_from_to_pairs = None
    else:
        additional_from_to_pairs_list_of_lists = []
        for described_additional_from_to_pair in described_additional_from_to_pairs:
            from_ = described_additional_from_to_pair["from_"]
            to = described_additional_from_to_pair["to"]
            weight = described_additional_from_to_pair["weight"]
            assert isinstance(weight, float)
            assert 0.0 <= weight <= 1.0
            assert isinstance(from_, list)
            assert isinstance(to, list)
            assert len(from_) == 3
            assert len(to) == 3
            assert all(isinstance(x, int) for x in from_)
            assert all(isinstance(x, int) for x in to)
            assert all(0.0 <= x  and x <= 255.0 for x in from_)
            num_repeats = int(np.round(weight * 10000))
            for _ in range(num_repeats):
                additional_from_to_pairs_list_of_lists.append([from_, to])
            
        additional_from_to_pairs = np.array(
            additional_from_to_pairs_list_of_lists,
            dtype=np.float64
        )
        if use_linear_light:
            additional_from_to_pairs /= 255.0
            additional_from_to_pairs = additional_from_to_pairs**2.2
        else:
            additional_from_to_pairs /= 255.0

    
    from_to_mapping_array_f64s = []
    for insertion_description_id in insertion_description_ids:
        from_to_mapping_array_f64 = get_from_to_mapping_array_f64_from_insertion_description_id(
            insertion_description_id=insertion_description_id,
            use_linear_light=use_linear_light,
        )
        from_to_mapping_array_f64s.append(from_to_mapping_array_f64)
    
    from_to_mapping_array_f64_without_additional_pairs = np.concatenate(from_to_mapping_array_f64s, axis=0)
    
    if additional_from_to_pairs is not None:
        from_to_mapping_array_f64s.append(additional_from_to_pairs)

    from_to_mapping_array_f64 = np.concatenate(
        (
            from_to_mapping_array_f64_without_additional_pairs,
            additional_from_to_pairs
        ),
        axis=0
    )

    print(f"{from_to_mapping_array_f64.shape=}")
    print(f"{from_to_mapping_array_f64=}")

    coefficients = get_color_correction_polynomial_coefficients_from_from_to_mapping_array_f64(
        degree=degree,
        from_to_mapping_array_f64=from_to_mapping_array_f64,
    )
    print(f"{coefficients=}")

    input_vectors = from_to_mapping_array_f64_without_additional_pairs[:, 0, :]
    output_should_be = from_to_mapping_array_f64_without_additional_pairs[:, 1, :]

    print(f"{np.mean(input_vectors, axis=0)=}")
    print(f"{np.mean(output_should_be, axis=0)=}")

    corrected = use_polynomial_regression_coefficients(
        degree=degree,
        coefficients=coefficients,
        input_vectors=input_vectors
    )

    errors = np.sum((corrected - output_should_be)**2, axis=1)**0.5
    assert errors.ndim == 1
    prii_histogram(x=errors)

    inlier = np.nonzero(errors < 0.2)[0]
    print(f"{inlier.shape=}")

    reduced_from_to_mapping_array_f64_without_additional_pairs = from_to_mapping_array_f64_without_additional_pairs[inlier, :, :]
    print(f"{reduced_from_to_mapping_array_f64_without_additional_pairs.shape=}")

    reduced_from_to_mapping_array_f64 = np.concatenate(
        (
            reduced_from_to_mapping_array_f64_without_additional_pairs,
            additional_from_to_pairs
        ),
        axis=0
    )

    coefficients = get_color_correction_polynomial_coefficients_from_from_to_mapping_array_f64(
        degree=degree,
        from_to_mapping_array_f64=reduced_from_to_mapping_array_f64,
    )

    input_vectors = from_to_mapping_array_f64[:, 0, :]
    output_should_be = from_to_mapping_array_f64[:, 1, :]

    corrected = use_polynomial_regression_coefficients(
        degree=degree,
        coefficients=coefficients,
        input_vectors=input_vectors
    )

    errors = np.sum((corrected - output_should_be)**2, axis=1)**0.5
    assert errors.ndim == 1
    prii_histogram(x=errors)


    color_correction_out_path = Path.home() / "color_correction.json"
    save_color_correction_as_json(
        degree=degree,
        coefficients=coefficients,
        out_path=color_correction_out_path
    )
    print(f"saved color correction to {color_correction_out_path}")

    out_dir = Path(
        "~/color_corrected"
    ).expanduser()

    out_dir.mkdir(exist_ok=True, parents=True)

    for insertion_description_id in insertion_description_ids:
        show_color_correction_result_on_insertion_description_id(
            use_linear_light=use_linear_light,
            degree=degree,
            coefficients=coefficients,
            insertion_description_id=insertion_description_id,
            out_dir=out_dir,
        )
   

    s = "flipflop ~/color_corrected"
    pyperclip.copy(s)
    print("We suggest you run the following command:")
    print(s)
    print("you can just paste it since it is already on the clipboard")
 
    print(f"saved color correction to {color_correction_out_path}")
    print("You might want to save it a la:")
    print(f"store_file_by_sha256 {color_correction_out_path}")
 
if __name__ == "__main__":
    fccfi_fit_color_correction_from_insertions()