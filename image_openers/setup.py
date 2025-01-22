from setuptools import setup

setup(
    name="image_openers",
    version="0.1.0",
    py_modules=[
        "get_bgra_hwc_np_u8",
        "image_openers",
        "make_bgra_from_original_and_mask_paths",
        "make_rgba_from_original_and_mask_paths",
        "open_a_grayscale_png_barfing_if_it_is_not_grayscale",
        "open_alpha_channel_image_as_a_single_channel_grayscale_image",
        "open_as_grayscale_and_yet_still_hwc_rgb_np_uint8",
        "open_as_grayscale_regardless",
        "open_as_hwc_rgb_np_uint8",
        "open_as_hwc_rgba_np_uint8",
        "open_image_as_rgb_np_uint8_ignoring_any_alpha",
        "pil_to_jpg_to_bytes",
        "pil_to_png_to_bytes",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "colorama",
    ],
    entry_points={
        "console_scripts": [
        ],
    },
)
