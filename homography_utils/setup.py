from setuptools import setup

setup(
    name="homography_utils",
    version="0.1.0",
    py_modules=[
        "convert_felix_format_to_json",
        "find_homography_from_2d_correspondences",
        "homographically_warp_advertisement",
        "homography_texture_rendering",
        "homography_texture_rendering_demo",
        "homography_utils",
        "interpolate_keyframes",
        "landmarks_from_cameras_for_homographies",
        "prepare_ads",
        "recover_SO3_from_shadow",
        "smoothly_connect_homography_charts",
        "spherical_homography_utils",
        "tripod_homography",
        "write_camera_parameters_from_homographies",
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
