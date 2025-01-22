from setuptools import setup

setup(
    name="image_writers",
    version="0.1.0",
    py_modules=[
        "image_writers",
        "write_grayscale_hw_np_u8_to_png",
        "write_rgb_and_alpha_to_png",
        "write_rgb_hwc_np_u8_to_png",
        "write_rgba_hwc_np_u8_to_png",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[],
    entry_points={
        "console_scripts": [
        ],
    },
)
