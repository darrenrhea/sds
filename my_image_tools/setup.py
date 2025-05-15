from setuptools import setup

setup(
    name="my_image_tools",
    version="0.1.0",
    py_modules=[
        "add_alphas",
        "cast_judgement",
        "compare_two_segmentation_answers",
        "confirm_alpha_of_rgba_png_is_binary",
        "diff_images",
        "extract_alpha_channel",
        "flip_alpha_channel",
        "get_frames",
        "image_displayers",
        "pric",
        "pridiff",
        "see_alpha_channel_differences",
        "show_bad_frames",
        "show_mask",
        "show_new_mask",
        "threshold_alpha_of_rgba_png",
        "transfer_alpha_channel",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "extract_alpha = extract_alpha_cli_tool:extract_alpha_cli_tool",
            "pric = pric_cli_tool:pric_cli_tool",
            "pridiff = pridiff_cli_tool:pridiff_cli_tool",
            "pridiffalpha = pridiffalpha_cli_tool:pridiffalpha_cli_tool",
            "transfer_alpha = transfer_alpha_cli_tool:transfer_alpha_cli_tool",
        ],
    },
)
