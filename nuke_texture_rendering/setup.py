from setuptools import setup

setup(
    name="nuke_texture_rendering",
    version="0.1.0",
    py_modules=[
        "forward_project_world_points",
        "get_t_hit",
        "insert_ad_into_camera_posed_original_video_frame",
        "insert_ads_faster",
        "insert_quads_into_camera_posed_image_behind_mask",
        "least_positive_element_and_index",
        "least_positive_element_and_index_over_last_axis",
        "nuke_texture_rendering",
        "nuke_texture_rendering_demo",
        "nuke_texture_rendering_one_off",
        "render_ad_on_subregion",
        "render_ad_texture_on_quad",
        "render_ads_on_subregion",
        "rip_world_points",
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
