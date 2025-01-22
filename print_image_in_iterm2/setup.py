from setuptools import setup

setup(
    name="print_image_in_iterm2",
    version="0.1.0",
    py_modules=[
        "convert_to_rgb_hwc_np_u8",
        "convert_to_rgba_hwc_np_u8",
        "get_to_image_pil_from_one_of_these",
        "prii",
        "prii_linear_f32",
        "prii_named_xy_points_on_image",
        "prii_named_xy_points_on_image_with_auto_zoom",
        "prii_nonlinear_f32",
        "print_data_image_in_iterm2",
        "print_image_in_iterm2",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "pri = pri_cli_tool:pri_cli_tool",
        ],
    },
)
