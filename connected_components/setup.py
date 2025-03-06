from setuptools import setup

setup(
    name="connected_components",
    version="0.1.0",
    py_modules=[
        "RectangleSummer",
        "find_indices_of_the_two_closest_boxes",
        "get_connected_components_of_mask",
        "horizontal_distance_between_boxes",
        "make_cutouts",
        "make_smaller",
        "old",
        "polygon_to_mask",
        "shrink",
        "union_two_boxes_together",
        "union_until_only_3_boxes_remain",
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
