from setuptools import setup

setup(
    name="segmentation_utils",
    version="0.1.3",
    py_modules=[
        "SegmentationDataset",
        "randomization_utils",
        "get_numpy_arrays_of_croppings_and_their_masks",
        "cut_this_many_interesting_subrectangles_from_annotated_image",
        "annotated_data",
        "cover_subrectangles",
        "open_binary_mask_attempt",
        "image_openers",
    ],
    license="MIT",
    long_description=(
        "Tools for people working on image segmentation and pngs"
    ),
    install_requires=[
        "pillow",
        "colorama",
        "print_image_in_iterm2",
    ],
    entry_points={
        "console_scripts": [
            "flip_alpha_channel = flip_alpha_channel:main",
            "cast_judgement = cast_judgement:main",
            "transfer_alpha = transfer_alpha:main",
            "transfer_flip_alpha = transfer_flip_alpha:main",
            "confirm_binary = confirm_alpha_of_rgba_png_is_binary:main",
            "pri01grayscale = image_displayers:mainpri01grayscale",
            "threshold_alpha = threshold_alpha_of_rgba_png:main",
            "frames = get_frames:main",
            "show_mask = show_mask:main",
            "show_new_mask = show_new_mask:main",
            "add_alphas = add_alphas:main",

        ]
    },
)
