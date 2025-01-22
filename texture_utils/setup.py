from setuptools import setup

setup(
    name="texture_utils",
    version="0.1.0",
    py_modules=[
        "get_rgba_hwc_np_f32_from_texture_id",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
        ],
    },
)
